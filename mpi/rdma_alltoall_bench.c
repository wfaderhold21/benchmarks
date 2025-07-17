#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>

#include <mpi.h>
#include <infiniband/verbs.h>

#define NR_ITER     110
#define SKIP        10
#define MAX_MSG_SIZE (1024 * 1024)  // 1MB
#define MIN_MSG_SIZE 8               // 8 bytes

// RDMA context structures
struct rdma_context {
    struct ibv_device *device;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp **qps;  // Array of QPs, one per connection
    struct ibv_port_attr port_attr;
    uint32_t *qp_nums;    // Array of QP numbers
    uint16_t lid;
    uint8_t port_num;
    void *buffer;
    size_t buffer_size;
    int num_qps;
};

struct rdma_connection {
    struct rdma_context *ctx;
    uint32_t remote_qp_num;
    uint16_t remote_lid;
    uint8_t remote_port_num;
    union ibv_gid remote_gid;
    uint32_t remote_rkey;  // Remote memory key for RDMA operations
    uintptr_t remote_buffer_addr;  // Remote buffer address
    int qp_index;  // Index of the QP used for this connection
};

static struct rdma_context *rdma_ctx = NULL;
static struct rdma_connection *connections = NULL;
static int num_ranks, my_rank;

// Initialize RDMA context
static int init_rdma_context(struct rdma_context *ctx, size_t buffer_size) {
    struct ibv_device **dev_list;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    
    // Get device list
    dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        fprintf(stderr, "Failed to get IB devices list\n");
        return -1;
    }
    
    // Use first available device
    ctx->device = dev_list[0];
    if (!ctx->device) {
        fprintf(stderr, "No IB devices found\n");
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    // Open device context
    ctx->context = ibv_open_device(ctx->device);
    if (!ctx->context) {
        fprintf(stderr, "Failed to open IB device context\n");
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    // Allocate protection domain
    ctx->pd = ibv_alloc_pd(ctx->context);
    if (!ctx->pd) {
        fprintf(stderr, "Failed to allocate protection domain\n");
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    // Allocate buffer and register memory region
    ctx->buffer = malloc(buffer_size);
    if (!ctx->buffer) {
        fprintf(stderr, "Failed to allocate buffer\n");
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    ctx->mr = ibv_reg_mr(ctx->pd, ctx->buffer, buffer_size,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE);
    if (!ctx->mr) {
        fprintf(stderr, "Failed to register memory region\n");
        free(ctx->buffer);
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    // Create completion queue
    ctx->cq = ibv_create_cq(ctx->context, 10, NULL, NULL, 0);
    if (!ctx->cq) {
        fprintf(stderr, "Failed to create completion queue\n");
        ibv_dereg_mr(ctx->mr);
        free(ctx->buffer);
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    // Create multiple queue pairs (one per connection)
    ctx->num_qps = num_ranks - 1;  // We don't need a QP for ourselves
    ctx->qps = malloc(ctx->num_qps * sizeof(struct ibv_qp*));
    ctx->qp_nums = malloc(ctx->num_qps * sizeof(uint32_t));
    
    if (!ctx->qps || !ctx->qp_nums) {
        fprintf(stderr, "Failed to allocate QP arrays\n");
        ibv_destroy_cq(ctx->cq);
        ibv_dereg_mr(ctx->mr);
        free(ctx->buffer);
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    for (int i = 0; i < ctx->num_qps; i++) {
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.send_cq = ctx->cq;
        qp_init_attr.recv_cq = ctx->cq;
        qp_init_attr.cap.max_send_wr = 10;
        qp_init_attr.cap.max_recv_wr = 10;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = 0;
        
        ctx->qps[i] = ibv_create_qp(ctx->pd, &qp_init_attr);
        if (!ctx->qps[i]) {
            fprintf(stderr, "Failed to create queue pair %d\n", i);
            // Cleanup previously created QPs
            for (int j = 0; j < i; j++) {
                ibv_destroy_qp(ctx->qps[j]);
            }
            free(ctx->qps);
            free(ctx->qp_nums);
            ibv_destroy_cq(ctx->cq);
            ibv_dereg_mr(ctx->mr);
            free(ctx->buffer);
            ibv_dealloc_pd(ctx->pd);
            ibv_close_device(ctx->context);
            ibv_free_device_list(dev_list);
            return -1;
        }
        ctx->qp_nums[i] = ctx->qps[i]->qp_num;
    }
    
    // Query port attributes
    if (ibv_query_port(ctx->context, 1, &ctx->port_attr) != 0) {
        fprintf(stderr, "Failed to query port attributes\n");
        ibv_destroy_qp(ctx->qp);
        ibv_destroy_cq(ctx->cq);
        ibv_dereg_mr(ctx->mr);
        free(ctx->buffer);
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    ctx->qp_num = ctx->qp->qp_num;
    ctx->lid = ctx->port_attr.lid;
    ctx->port_num = 1;
    ctx->buffer_size = buffer_size;
    
    ibv_free_device_list(dev_list);
    return 0;
}

// Exchange QP information and memory keys between ranks
static int exchange_qp_info() {
    // First, exchange basic info to determine buffer sizes
    struct {
        uint16_t lid;
        uint8_t port_num;
        union ibv_gid gid;
        uint32_t rkey;  // Remote key for memory access
        uintptr_t buffer_addr;  // Buffer address for RDMA operations
        int num_qps;
    } basic_info, *basic_remote_info;
    
    basic_info.lid = rdma_ctx->lid;
    basic_info.port_num = rdma_ctx->port_num;
    basic_info.rkey = rdma_ctx->mr->rkey;  // Our memory region's remote key
    basic_info.buffer_addr = (uintptr_t)rdma_ctx->buffer;  // Our buffer address
    basic_info.num_qps = rdma_ctx->num_qps;
    
    // Get local GID
    if (ibv_query_gid(rdma_ctx->context, rdma_ctx->port_num, 0, &basic_info.gid) != 0) {
        fprintf(stderr, "Failed to query GID\n");
        return -1;
    }
    
    // Allocate array for remote basic info
    basic_remote_info = malloc(num_ranks * sizeof(*basic_remote_info));
    if (!basic_remote_info) {
        fprintf(stderr, "Failed to allocate remote basic info array\n");
        return -1;
    }
    
    // Exchange basic information
    MPI_Allgather(&basic_info, sizeof(basic_info), MPI_BYTE,
                  basic_remote_info, sizeof(basic_info), MPI_BYTE, MPI_COMM_WORLD);
    
    // Now exchange QP numbers using separate messages
    uint32_t *local_qp_nums = malloc(rdma_ctx->num_qps * sizeof(uint32_t));
    uint32_t *all_qp_nums = malloc(num_ranks * rdma_ctx->num_qps * sizeof(uint32_t));
    
    if (!local_qp_nums || !all_qp_nums) {
        fprintf(stderr, "Failed to allocate QP numbers arrays\n");
        free(basic_remote_info);
        free(local_qp_nums);
        free(all_qp_nums);
        return -1;
    }
    
    // Copy our QP numbers
    for (int i = 0; i < rdma_ctx->num_qps; i++) {
        local_qp_nums[i] = rdma_ctx->qp_nums[i];
    }
    
    // Exchange QP numbers
    MPI_Allgather(local_qp_nums, rdma_ctx->num_qps, MPI_UINT32_T,
                  all_qp_nums, rdma_ctx->num_qps, MPI_UINT32_T, MPI_COMM_WORLD);
    
    // Initialize connections
    connections = malloc(num_ranks * sizeof(*connections));
    if (!connections) {
        fprintf(stderr, "Failed to allocate connections array\n");
        free(basic_remote_info);
        free(local_qp_nums);
        free(all_qp_nums);
        return -1;
    }
    
    for (int i = 0; i < num_ranks; i++) {
        connections[i].ctx = rdma_ctx;
        connections[i].remote_lid = basic_remote_info[i].lid;
        connections[i].remote_port_num = basic_remote_info[i].port_num;
        connections[i].remote_gid = basic_remote_info[i].gid;
        connections[i].remote_rkey = basic_remote_info[i].rkey;  // Store remote memory key
        connections[i].remote_buffer_addr = basic_remote_info[i].buffer_addr;  // Store remote buffer address
        
        // Find the appropriate QP for this connection
        if (i < my_rank) {
            connections[i].qp_index = i;
            connections[i].remote_qp_num = all_qp_nums[i * rdma_ctx->num_qps + (my_rank - 1)];
        } else if (i > my_rank) {
            connections[i].qp_index = i - 1;
            connections[i].remote_qp_num = all_qp_nums[i * rdma_ctx->num_qps + my_rank];
        } else {
            connections[i].qp_index = -1;  // No QP needed for self
            connections[i].remote_qp_num = 0;
        }
    }
    
    free(basic_remote_info);
    free(local_qp_nums);
    free(all_qp_nums);
    return 0;
}

// Connect QPs
static int connect_qps() {
    struct ibv_qp_attr attr;
    int flags;
    
    for (int i = 0; i < num_ranks; i++) {
        if (i == my_rank) continue;
        
        int qp_idx = connections[i].qp_index;
        if (qp_idx < 0) continue;  // Skip self
        
        // Move QP to INIT state
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = rdma_ctx->port_num;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        
        flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(rdma_ctx->qps[qp_idx], &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP %d to INIT state\n", qp_idx);
            return -1;
        }
        
        // Move QP to RTR state
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = connections[i].remote_qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = connections[i].remote_lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = rdma_ctx->port_num;
        attr.ah_attr.grh.dgid = connections[i].remote_gid;
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.traffic_class = 0;
        
        flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        if (ibv_modify_qp(rdma_ctx->qps[qp_idx], &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP %d to RTR state\n", qp_idx);
            return -1;
        }
        
        // Move QP to RTS state
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        
        flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
        if (ibv_modify_qp(rdma_ctx->qps[qp_idx], &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP %d to RTS state\n", qp_idx);
            return -1;
        }
    }
    
    return 0;
}

// Perform RDMA write operation
static int rdma_write(int target_rank, void *local_addr, void *remote_addr, size_t size) {
    struct ibv_send_wr wr, *bad_wr;
    struct ibv_sge sge;
    struct ibv_wc wc;
    int ret;
    
    // Get the QP index for this connection
    int qp_idx = connections[target_rank].qp_index;
    if (qp_idx < 0) {
        fprintf(stderr, "Invalid QP index for rank %d\n", target_rank);
        return -1;
    }
    
    // Prepare scatter-gather element
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local_addr;
    sge.length = size;
    sge.lkey = rdma_ctx->mr->lkey;
    
    // Prepare work request
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uintptr_t)target_rank;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = (uintptr_t)remote_addr;
    wr.wr.rdma.rkey = connections[target_rank].remote_rkey;  // Use remote memory key
    
    // Post send request
    ret = ibv_post_send(rdma_ctx->qps[qp_idx], &wr, &bad_wr);
    if (ret != 0) {
        fprintf(stderr, "Failed to post send request to QP %d\n", qp_idx);
        return -1;
    }
    
    // Wait for completion
    do {
        ret = ibv_poll_cq(rdma_ctx->cq, 1, &wc);
    } while (ret == 0);
    
    if (ret < 0) {
        fprintf(stderr, "Failed to poll completion queue\n");
        return -1;
    }
    
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "Work completion failed with status %d (QP %d)\n", wc.status, qp_idx);
        return -1;
    }
    
    return 0;
}

// Perform alltoall using RDMA writes
static void rdma_alltoall(void *sendbuf, void *recvbuf, size_t msg_size) {
    size_t offset;
    char *send_ptr = (char *)sendbuf;
    char *recv_ptr = (char *)recvbuf;
    
    // Calculate offsets for each rank
    for (int i = 0; i < num_ranks; i++) {
        offset = i * msg_size;
        
        // Send data to rank i
        if (i != my_rank) {
            // Calculate remote buffer address using the exchanged buffer address
            void *remote_buffer = (void *)(connections[i].remote_buffer_addr + my_rank * msg_size);
            rdma_write(i, send_ptr + offset, remote_buffer, msg_size);
        } else {
            // Local copy
            memcpy(recv_ptr + offset, send_ptr + offset, msg_size);
        }
    }
    
    // Copy received data from RDMA buffer to receive buffer
    for (int i = 0; i < num_ranks; i++) {
        if (i != my_rank) {
            offset = i * msg_size;
            memcpy(recv_ptr + offset, 
                   rdma_ctx->buffer + i * msg_size, msg_size);
        }
    }
}

// Cleanup RDMA resources
static void cleanup_rdma() {
    if (connections) {
        free(connections);
        connections = NULL;
    }
    
    if (rdma_ctx) {
        if (rdma_ctx->qps) {
            for (int i = 0; i < rdma_ctx->num_qps; i++) {
                if (rdma_ctx->qps[i]) {
                    ibv_destroy_qp(rdma_ctx->qps[i]);
                }
            }
            free(rdma_ctx->qps);
        }
        if (rdma_ctx->qp_nums) {
            free(rdma_ctx->qp_nums);
        }
        if (rdma_ctx->cq) {
            ibv_destroy_cq(rdma_ctx->cq);
        }
        if (rdma_ctx->mr) {
            ibv_dereg_mr(rdma_ctx->mr);
        }
        if (rdma_ctx->pd) {
            ibv_dealloc_pd(rdma_ctx->pd);
        }
        if (rdma_ctx->context) {
            ibv_close_device(rdma_ctx->context);
        }
        if (rdma_ctx->buffer) {
            free(rdma_ctx->buffer);
        }
        free(rdma_ctx);
        rdma_ctx = NULL;
    }
}

int main(int argc, char **argv) {
    int ret;
    double start_time, end_time, min_time, max_time, avg_time;
    double bandwidth, total_bandwidth;
    size_t msg_size;
    void *sendbuf, *recvbuf;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // Allocate RDMA context
    rdma_ctx = malloc(sizeof(*rdma_ctx));
    if (!rdma_ctx) {
        fprintf(stderr, "Failed to allocate RDMA context\n");
        MPI_Finalize();
        return -1;
    }
    
    // Initialize RDMA context with buffer large enough for alltoall
    size_t buffer_size = num_ranks * MAX_MSG_SIZE;
    ret = init_rdma_context(rdma_ctx, buffer_size);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        cleanup_rdma();
        MPI_Finalize();
        return -1;
    }
    
    // Exchange QP information
    ret = exchange_qp_info();
    if (ret != 0) {
        fprintf(stderr, "Failed to exchange QP information\n");
        cleanup_rdma();
        MPI_Finalize();
        return -1;
    }
    
    // Connect QPs
    ret = connect_qps();
    if (ret != 0) {
        fprintf(stderr, "Failed to connect QPs\n");
        cleanup_rdma();
        MPI_Finalize();
        return -1;
    }
    
    // Allocate send and receive buffers
    sendbuf = malloc(MAX_MSG_SIZE * num_ranks);
    recvbuf = malloc(MAX_MSG_SIZE * num_ranks);
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Failed to allocate send/receive buffers\n");
        cleanup_rdma();
        MPI_Finalize();
        return -1;
    }
    
    // Initialize send buffer with rank-specific data
    for (int i = 0; i < num_ranks; i++) {
        char *ptr = (char *)sendbuf + i * MAX_MSG_SIZE;
        for (size_t j = 0; j < MAX_MSG_SIZE; j++) {
            ptr[j] = (char)(my_rank * 100 + i);
        }
    }
    
    // Print header
    if (my_rank == 0) {
        printf("%-12s %-12s %-15s %-15s %-15s %-15s\n", 
               "Message Size", "Total Size", "Bandwidth (MB/s)", 
               "Avg Latency (us)", "Min Latency (us)", "Max Latency (us)");
        printf("--------------------------------------------------------------------------------\n");
    }
    
    // Benchmark different message sizes
    for (msg_size = MIN_MSG_SIZE; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        min_time = 1e9;
        max_time = 0.0;
        avg_time = 0.0;
        
        // Warmup iterations
        for (int iter = 0; iter < SKIP; iter++) {
            MPI_Barrier(MPI_COMM_WORLD);
            rdma_alltoall(sendbuf, recvbuf, msg_size);
        }
        
        // Benchmark iterations
        for (int iter = 0; iter < NR_ITER; iter++) {
            MPI_Barrier(MPI_COMM_WORLD);
            
            start_time = MPI_Wtime();
            rdma_alltoall(sendbuf, recvbuf, msg_size);
            end_time = MPI_Wtime();
            
            double iter_time = (end_time - start_time) * 1e6; // Convert to microseconds
            avg_time += iter_time;
            
            if (iter_time < min_time) min_time = iter_time;
            if (iter_time > max_time) max_time = iter_time;
        }
        
        avg_time /= NR_ITER;
        
        // Calculate bandwidth
        bandwidth = (msg_size * num_ranks * num_ranks) / (avg_time * 1e-6) / (1024 * 1024); // MB/s
        
        // Reduce statistics across ranks
        double global_avg, global_min, global_max, global_bw;
        MPI_Allreduce(&avg_time, &global_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&min_time, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_time, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&bandwidth, &global_bw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        global_avg /= num_ranks;
        global_bw /= num_ranks;
        
        // Print results
        if (my_rank == 0) {
            printf("%-12zu %-12zu %-15.2f %-15.2f %-15.2f %-15.2f\n",
                   msg_size, msg_size * num_ranks * num_ranks,
                   global_bw, global_avg, global_min, global_max);
        }
    }
    
    // Cleanup
    free(sendbuf);
    free(recvbuf);
    cleanup_rdma();
    MPI_Finalize();
    
    return 0;
} 