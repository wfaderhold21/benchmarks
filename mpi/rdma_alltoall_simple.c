#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <pthread.h>

#include <infiniband/verbs.h>

#define NR_ITER     110
#define SKIP        10
#define MAX_MSG_SIZE (1024 * 1024)  // 1MB
#define MIN_MSG_SIZE 8               // 8 bytes
#define NUM_RANKS    4               // Simulate 4 ranks

// RDMA context structures
struct rdma_context {
    struct ibv_device *device;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_port_attr port_attr;
    uint32_t qp_num;
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
    union ibv_gid remote_gid;
};

static struct rdma_context *rdma_ctx = NULL;
static struct rdma_connection *connections = NULL;
static int num_ranks = NUM_RANKS, my_rank = 0;

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
    
    ctx->qp_num = ctx->qp->qp_num;
    ctx->lid = ctx->port_attr.lid;
    ctx->port_num = 1;
    ctx->buffer_size = buffer_size;
    
    ibv_free_device_list(dev_list);
    return 0;
}

// Simulate RDMA write operation (for testing without actual network)
static int rdma_write(int target_rank, void *local_addr, void *remote_addr, size_t size) {
    // In a real implementation, this would perform actual RDMA write
    // For testing, we'll just simulate the operation
    printf("RDMA Write: Rank %d -> Rank %d, Size: %zu bytes\n", my_rank, target_rank, size);
    
    // Simulate network delay
    usleep(100); // 100 microseconds delay
    
    return 0;
}

// Perform alltoall using RDMA writes
static void rdma_alltoall(void *sendbuf, void *recvbuf, size_t msg_size) {
    size_t offset;
    char *send_ptr = (char *)sendbuf;
    char *recv_ptr = (char *)recvbuf;
    
    printf("Alltoall: Message size = %zu bytes\n", msg_size);
    
    // Calculate offsets for each rank
    for (int i = 0; i < num_ranks; i++) {
        offset = i * msg_size;
        
        // Send data to rank i
        if (i != my_rank) {
            rdma_write(i, send_ptr + offset, 
                      rdma_ctx->buffer + my_rank * msg_size, msg_size);
        } else {
            // Local copy
            memcpy(recv_ptr + offset, send_ptr + offset, msg_size);
            printf("Local copy: Rank %d\n", my_rank);
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
    double start_time, end_time, min_time, max_time, avg_time;
    double bandwidth;
    size_t msg_size;
    void *sendbuf, *recvbuf;
    
    printf("RDMA Alltoall Benchmark (Simplified Version)\n");
    printf("Simulating %d ranks\n", num_ranks);
    
    // Allocate RDMA context
    rdma_ctx = malloc(sizeof(*rdma_ctx));
    if (!rdma_ctx) {
        fprintf(stderr, "Failed to allocate RDMA context\n");
        return -1;
    }
    
    // Initialize RDMA context with buffer large enough for alltoall
    size_t buffer_size = num_ranks * MAX_MSG_SIZE;
    ret = init_rdma_context(rdma_ctx, buffer_size);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        cleanup_rdma();
        return -1;
    }
    
    printf("RDMA context initialized successfully\n");
    
    // Allocate send and receive buffers
    sendbuf = malloc(MAX_MSG_SIZE * num_ranks);
    recvbuf = malloc(MAX_MSG_SIZE * num_ranks);
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Failed to allocate send/receive buffers\n");
        cleanup_rdma();
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
    printf("%-12s %-12s %-15s %-15s %-15s %-15s\n", 
           "Message Size", "Total Size", "Bandwidth (MB/s)", 
           "Avg Latency (us)", "Min Latency (us)", "Max Latency (us)");
    printf("--------------------------------------------------------------------------------\n");
    
    // Benchmark different message sizes
    for (msg_size = MIN_MSG_SIZE; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        min_time = 1e9;
        max_time = 0.0;
        avg_time = 0.0;
        
        // Warmup iterations
        for (int iter = 0; iter < SKIP; iter++) {
            rdma_alltoall(sendbuf, recvbuf, msg_size);
        }
        
        // Benchmark iterations
        for (int iter = 0; iter < NR_ITER; iter++) {
            start_time = (double)clock() / CLOCKS_PER_SEC * 1e6; // Convert to microseconds
            rdma_alltoall(sendbuf, recvbuf, msg_size);
            end_time = (double)clock() / CLOCKS_PER_SEC * 1e6;
            
            double iter_time = end_time - start_time;
            avg_time += iter_time;
            
            if (iter_time < min_time) min_time = iter_time;
            if (iter_time > max_time) max_time = iter_time;
        }
        
        avg_time /= NR_ITER;
        
        // Calculate bandwidth
        bandwidth = (msg_size * num_ranks * num_ranks) / (avg_time * 1e-6) / (1024 * 1024); // MB/s
        
        // Print results
        printf("%-12zu %-12zu %-15.2f %-15.2f %-15.2f %-15.2f\n",
               msg_size, msg_size * num_ranks * num_ranks,
               bandwidth, avg_time, min_time, max_time);
    }
    
    // Cleanup
    free(sendbuf);
    free(recvbuf);
    cleanup_rdma();
    
    printf("Benchmark completed successfully\n");
    return 0;
} 