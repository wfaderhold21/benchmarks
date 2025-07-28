#ifndef SHMEM_COMMON_H
#define SHMEM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <shmem.h>
#include <shmemx.h>

/* Common constants */
#define MAX_ITER    100000
#define WARMUP      1000
#define DEFAULT_SKIP 10
#define MIN_MSG_SIZE 1
#define MAX_MSG_SIZE (1 << 20)  /* 1MB */

/* Color codes for output formatting */
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"

/* Timing utilities */
static inline double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

#define TIME() getMicrosecondTimeStamp()

/* Memory size formatting */
static inline void format_size(size_t size, char* buffer, size_t buffer_size) {
    if (size < 1024) {
        snprintf(buffer, buffer_size, "%lu B", size);
    } else if (size < 1024 * 1024) {
        snprintf(buffer, buffer_size, "%.2f KB", (double)size / 1024);
    } else {
        snprintf(buffer, buffer_size, "%.2f MB", (double)size / (1024 * 1024));
    }
}

/* Results structure for latency/bandwidth measurements */
typedef struct {
    double latency_us;
    double bandwidth_mbps;
    double message_rate;
    size_t message_size;
    int iterations;
} benchmark_result_t;

/* Test configuration structure */
typedef struct {
    int min_size;
    int max_size;
    int iterations;
    int warmup;
    int skip;
    int validate;
} test_config_t;

/* Default test configuration */
static const test_config_t default_config = {
    .min_size = MIN_MSG_SIZE,
    .max_size = MAX_MSG_SIZE,
    .iterations = MAX_ITER,
    .warmup = WARMUP,
    .skip = DEFAULT_SKIP,
    .validate = 1
};

/* Benchmark result printing */
static inline void print_benchmark_header(const char* test_name) {
    int my_pe = shmem_my_pe();
    if (my_pe == 0) {
        printf("\n" COLOR_CYAN "===============================================\n");
        printf(" %s BENCHMARK RESULTS\n", test_name);
        printf("===============================================" COLOR_RESET "\n");
        printf("%-12s %-12s %-15s %-15s\n", 
               "Size", "Latency(us)", "Bandwidth(MB/s)", "Msg Rate(M/s)");
        printf("---------------------------------------------------------------\n");
    }
}

static inline void print_benchmark_result(const benchmark_result_t* result) {
    char size_str[32];
    format_size(result->message_size, size_str, sizeof(size_str));
    
    printf("%-12s %-12.2f %-15.2f %-15.2f\n",
           size_str,
           result->latency_us,
           result->bandwidth_mbps,
           result->message_rate);
}

/* Memory allocation with hints if available */
static inline void* shmem_malloc_aligned(size_t size) {
#ifdef WITH_HINTS
    return shmemx_malloc_with_hint(size, SHMEM_HINT_NEAR_NIC_MEM);
#else
    return shmem_malloc(size);
#endif
}

/* Calculate iterations based on message size */
static inline int calculate_iterations(size_t msg_size, int base_iterations) {
    int iterations;
    if (msg_size <= 1024) {
        iterations = base_iterations;
    } else if (msg_size <= 64 * 1024) {
        iterations = base_iterations / 2;
    } else {
        iterations = base_iterations / 10;
    }
    /* Ensure minimum iterations to avoid division by zero */
    return iterations > 0 ? iterations : 1;
}

/* Validation helpers */
static inline int validate_data(const char* expected, const char* actual, size_t size) {
    return memcmp(expected, actual, size) == 0;
}

static inline void init_data(char* buffer, size_t size, char pattern) {
    memset(buffer, pattern, size);
}

#endif /* SHMEM_COMMON_H */ 