#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <mpi.h>
#include <infiniband/verbs.h>

// RDMA context structures
struct rdma_context {
    struct ibv_device *device;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_port_attr port_attr;
    uint16_t lid;
    uint8_t port_num;
    void *buffer;
    size_t buffer_size;
};

struct rdma_connection {
    struct rdma_context *ctx;
    uint32_t remote_qp_num;
    uint16_t remote_lid;
    uint8_t remote_port_num;
    uint32_t remote_rkey;
    uintptr_t remote_buffer_addr;
};

static struct rdma_context *rdma_ctx = NULL;
static struct rdma_connection *connections = NULL;
static int num_ranks, my_rank;

// Initialize RDMA context
static int init_rdma_context(struct rdma_context *ctx, size_t buffer_size) {
    struct ibv_device **dev_list;
    struct ibv_qp_init_attr qp_init_attr;
    
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
    
    fprintf(stderr, "Rank %d: Using device: %s\n", my_rank, 
            ibv_get_device_name(ctx->device) ? ibv_get_device_name(ctx->device) : "<unknown>");
    
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
    
    // Create queue pair
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.send_cq = ctx->cq;
    qp_init_attr.recv_cq = ctx->cq;
    qp_init_attr.cap.max_send_wr = 10;
    qp_init_attr.cap.max_recv_wr = 10;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.cap.max_inline_data = 0;
    
    ctx->qp = ibv_create_qp(ctx->pd, &qp_init_attr);
    if (!ctx->qp) {
        fprintf(stderr, "Failed to create queue pair\n");
        ibv_destroy_cq(ctx->cq);
        ibv_dereg_mr(ctx->mr);
        free(ctx->buffer);
        ibv_dealloc_pd(ctx->pd);
        ibv_close_device(ctx->context);
        ibv_free_device_list(dev_list);
        return -1;
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
    
    ctx->lid = ctx->port_attr.lid;
    ctx->port_num = 1;
    ctx->buffer_size = buffer_size;
    
    ibv_free_device_list(dev_list);
    return 0;
}

// Exchange QP information
static int exchange_qp_info() {
    struct {
        uint16_t lid;
        uint8_t port_num;
        uint32_t rkey;
        uintptr_t buffer_addr;
        uint32_t qp_num;
    } info, *remote_info;
    
    info.lid = rdma_ctx->lid;
    info.port_num = rdma_ctx->port_num;
    info.rkey = rdma_ctx->mr->rkey;
    info.buffer_addr = (uintptr_t)rdma_ctx->buffer;
    info.qp_num = rdma_ctx->qp->qp_num;
    
    remote_info = malloc(num_ranks * sizeof(*remote_info));
    if (!remote_info) {
        fprintf(stderr, "Failed to allocate remote info array\n");
        return -1;
    }
    
    MPI_Allgather(&info, sizeof(info), MPI_BYTE,
                  remote_info, sizeof(info), MPI_BYTE, MPI_COMM_WORLD);
    
    connections = malloc(num_ranks * sizeof(*connections));
    if (!connections) {
        fprintf(stderr, "Failed to allocate connections array\n");
        free(remote_info);
        return -1;
    }
    
    for (int i = 0; i < num_ranks; i++) {
        connections[i].ctx = rdma_ctx;
        connections[i].remote_lid = remote_info[i].lid;
        connections[i].remote_port_num = remote_info[i].port_num;
        connections[i].remote_rkey = remote_info[i].rkey;
        connections[i].remote_buffer_addr = remote_info[i].buffer_addr;
        connections[i].remote_qp_num = remote_info[i].qp_num;
    }
    
    free(remote_info);
    return 0;
}

// Connect QPs
static int connect_qps() {
    struct ibv_qp_attr attr;
    int flags;
    
    for (int i = 0; i < num_ranks; i++) {
        if (i == my_rank) continue;  // Skip self
        
        fprintf(stderr, "Rank %d: Connecting to rank %d\n", my_rank, i);
        
        // Move QP to INIT state
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = rdma_ctx->port_num;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        
        flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(rdma_ctx->qp, &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP to INIT state\n");
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
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = connections[i].remote_lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = rdma_ctx->port_num;
        
        flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        if (ibv_modify_qp(rdma_ctx->qp, &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP to RTR state\n");
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
        
        flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
        if (ibv_modify_qp(rdma_ctx->qp, &attr, flags) != 0) {
            fprintf(stderr, "Failed to modify QP to RTS state\n");
            return -1;
        }
        
        fprintf(stderr, "Rank %d: Successfully connected to rank %d\n", my_rank, i);
    }
    
    return 0;
}

// Perform RDMA write
static int rdma_write(int target_rank, void *local_addr, void *remote_addr, size_t size) {
    struct ibv_send_wr wr, *bad_wr;
    struct ibv_sge sge;
    struct ibv_wc wc;
    int ret;
    
    fprintf(stderr, "Rank %d: RDMA write to rank %d, size %zu\n", my_rank, target_rank, size);
    fprintf(stderr, "Rank %d: Local addr: %p, Remote addr: %p\n", my_rank, local_addr, remote_addr);
    fprintf(stderr, "Rank %d: Local lkey: %u, Remote rkey: %u\n", my_rank, rdma_ctx->mr->lkey, connections[target_rank].remote_rkey);
    
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
    wr.wr.rdma.rkey = connections[target_rank].remote_rkey;
    
    // Post send request
    ret = ibv_post_send(rdma_ctx->qp, &wr, &bad_wr);
    if (ret != 0) {
        fprintf(stderr, "Failed to post send request: %s\n", strerror(errno));
        return -1;
    }
    
    fprintf(stderr, "Rank %d: RDMA write posted successfully\n", my_rank);
    return 0;
}

// Cleanup RDMA resources
static void cleanup_rdma() {
    if (connections) {
        free(connections);
        connections = NULL;
    }
    
    if (rdma_ctx) {
        if (rdma_ctx->qp) {
            ibv_destroy_qp(rdma_ctx->qp);
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
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    if (num_ranks != 2) {
        fprintf(stderr, "This test requires exactly 2 ranks\n");
        MPI_Finalize();
        return -1;
    }
    
    fprintf(stderr, "Rank %d: Starting RDMA test\n", my_rank);
    
    // Allocate RDMA context
    rdma_ctx = malloc(sizeof(*rdma_ctx));
    if (!rdma_ctx) {
        fprintf(stderr, "Failed to allocate RDMA context\n");
        MPI_Finalize();
        return -1;
    }
    
    // Initialize RDMA context
    ret = init_rdma_context(rdma_ctx, 1024 * 1024);  // 1MB buffer
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Post receive work requests
    fprintf(stderr, "Rank %d: Posting receive work requests\n", my_rank);
    for (int i = 0; i < num_ranks; i++) {
        if (i == my_rank) continue;  // Skip self
        
        struct ibv_recv_wr wr, *bad_wr;
        struct ibv_sge sge;
        
        // Prepare scatter-gather element for receive
        memset(&sge, 0, sizeof(sge));
        sge.addr = (uintptr_t)rdma_ctx->buffer;
        sge.length = rdma_ctx->buffer_size;
        sge.lkey = rdma_ctx->mr->lkey;
        
        // Prepare receive work request
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = (uintptr_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        
        // Post receive work request
        int ret = ibv_post_recv(rdma_ctx->qp, &wr, &bad_wr);
        if (ret != 0) {
            fprintf(stderr, "Rank %d: Failed to post receive WR: %s\n", my_rank, strerror(errno));
            return -1;
        }
        
        fprintf(stderr, "Rank %d: Posted receive WR for rank %d\n", my_rank, i);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test RDMA write
    if (my_rank == 0) {
        char *test_data = (char *)rdma_ctx->buffer + 1024;
        strcpy(test_data, "Hello from rank 0!");
        size_t test_size = strlen(test_data) + 1;
        
        void *remote_buffer = (void *)(connections[1].remote_buffer_addr + 1024);
        ret = rdma_write(1, test_data, remote_buffer, test_size);
        
        if (ret == 0) {
            fprintf(stderr, "Rank 0: RDMA test successful\n");
        } else {
            fprintf(stderr, "Rank 0: RDMA test failed\n");
        }
    } else if (my_rank == 1) {
        char *test_data = (char *)rdma_ctx->buffer + 1024;
        strcpy(test_data, "Hello from rank 1!");
        size_t test_size = strlen(test_data) + 1;
        
        void *remote_buffer = (void *)(connections[0].remote_buffer_addr + 1024);
        ret = rdma_write(0, test_data, remote_buffer, test_size);
        
        if (ret == 0) {
            fprintf(stderr, "Rank 1: RDMA test successful\n");
        } else {
            fprintf(stderr, "Rank 1: RDMA test failed\n");
        }
    }
    
    // Add a small delay to ensure RDMA operations complete
    usleep(100000);  // 100ms delay
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Check received data
    if (my_rank == 0) {
        char *received = (char *)rdma_ctx->buffer + 1024;
        fprintf(stderr, "Rank 0: Received: '%s'\n", received);
    } else if (my_rank == 1) {
        char *received = (char *)rdma_ctx->buffer + 1024;
        fprintf(stderr, "Rank 1: Received: '%s'\n", received);
    }
    
    // Cleanup
    cleanup_rdma();
    MPI_Finalize();
    
    return 0;
} 