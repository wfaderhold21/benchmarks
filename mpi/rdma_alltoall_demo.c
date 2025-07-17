#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>

#define NR_ITER     110
#define SKIP        10
#define MAX_MSG_SIZE (1024 * 1024)  // 1MB
#define MIN_MSG_SIZE 8               // 8 bytes
#define NUM_RANKS    4               // Simulate 4 ranks

// Simulated RDMA context structures
struct rdma_context {
    void *buffer;
    size_t buffer_size;
    int rank;
    int num_ranks;
};

struct rdma_connection {
    struct rdma_context *ctx;
    int remote_rank;
};

static struct rdma_context *rdma_ctx = NULL;
static struct rdma_connection *connections = NULL;
static int num_ranks = NUM_RANKS, my_rank = 0;

// Initialize simulated RDMA context
static int init_rdma_context(struct rdma_context *ctx, size_t buffer_size, int rank, int nranks) {
    ctx->buffer = malloc(buffer_size);
    if (!ctx->buffer) {
        fprintf(stderr, "Failed to allocate buffer\n");
        return -1;
    }
    
    ctx->buffer_size = buffer_size;
    ctx->rank = rank;
    ctx->num_ranks = nranks;
    
    return 0;
}

// Simulate RDMA write operation
static int rdma_write(int target_rank, void *local_addr, void *remote_addr, size_t size) {
    // In a real implementation, this would perform actual RDMA write
    // For demonstration, we'll just simulate the operation
    printf("RDMA Write: Rank %d -> Rank %d, Size: %zu bytes\n", my_rank, target_rank, size);
    
    // Simulate network delay (microseconds)
    usleep(50 + (size / 1024)); // Delay proportional to message size
    
    return 0;
}

// Perform alltoall using simulated RDMA writes
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

// Cleanup simulated RDMA resources
static void cleanup_rdma() {
    if (connections) {
        free(connections);
        connections = NULL;
    }
    
    if (rdma_ctx) {
        if (rdma_ctx->buffer) {
            free(rdma_ctx->buffer);
        }
        free(rdma_ctx);
        rdma_ctx = NULL;
    }
}

// Get current time in microseconds
static double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

int main(int argc, char **argv) {
    int ret;
    double start_time, end_time, min_time, max_time, avg_time;
    double bandwidth;
    size_t msg_size;
    void *sendbuf, *recvbuf;
    
    printf("RDMA Alltoall Benchmark (Demonstration Version)\n");
    printf("Simulating %d ranks with RDMA write operations\n", num_ranks);
    printf("This demonstrates the concept without requiring actual RDMA hardware\n\n");
    
    // Allocate RDMA context
    rdma_ctx = malloc(sizeof(*rdma_ctx));
    if (!rdma_ctx) {
        fprintf(stderr, "Failed to allocate RDMA context\n");
        return -1;
    }
    
    // Initialize RDMA context with buffer large enough for alltoall
    size_t buffer_size = num_ranks * MAX_MSG_SIZE;
    ret = init_rdma_context(rdma_ctx, buffer_size, my_rank, num_ranks);
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
            start_time = get_time_us();
            rdma_alltoall(sendbuf, recvbuf, msg_size);
            end_time = get_time_us();
            
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
    
    printf("\nBenchmark completed successfully\n");
    printf("Note: This is a simulation. Real RDMA performance would be much higher.\n");
    return 0;
} 