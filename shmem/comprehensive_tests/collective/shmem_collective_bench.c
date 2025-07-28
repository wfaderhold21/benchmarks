#include "../common/common.h"

#define SYNC_SIZE SHMEM_COLLECT_SYNC_SIZE
#define MAX_COLLECTIVE_SIZE (1 << 16)  /* 64KB max for collectives */

/* Work arrays for collectives */
static long pSync[SHMEM_ALLTOALL_SYNC_SIZE];
static long pSync_collect[SHMEM_COLLECT_SYNC_SIZE];
static long pSync_reduce[SHMEM_REDUCE_SYNC_SIZE];

static void init_sync_arrays() {
    for (int i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
    for (int i = 0; i < SHMEM_COLLECT_SYNC_SIZE; i++) {
        pSync_collect[i] = SHMEM_SYNC_VALUE;
    }
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync_reduce[i] = SHMEM_SYNC_VALUE;
    }
}

static void print_collective_header(const char* operation) {
    int my_pe = shmem_my_pe();
    if (my_pe == 0) {
        printf("\n" COLOR_CYAN "===============================================\n");
        printf(" SHMEM %s COLLECTIVE BENCHMARK\n", operation);
        printf("===============================================" COLOR_RESET "\n");
        printf("%-12s %-15s %-15s %-12s\n", 
               "Size", "Latency(us)", "Bandwidth(MB/s)", "Iterations");
        printf("---------------------------------------------------------------\n");
    }
}

static void benchmark_alltoall(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    print_collective_header("ALLTOALL");
    
    for (size_t nelems = 1; nelems <= MAX_COLLECTIVE_SIZE / sizeof(long) / n_pes; nelems *= 2) {
        size_t total_size = nelems * n_pes * sizeof(long);
        int iterations = calculate_iterations(total_size, config->iterations / 10);
        
        long* source = (long*)shmem_malloc_aligned(total_size);
        long* dest = (long*)shmem_malloc_aligned(total_size);
        
        if (!source || !dest) {
            fprintf(stderr, "Memory allocation failed\n");
            shmem_global_exit(1);
        }
        
        /* Initialize data */
        for (size_t i = 0; i < nelems * n_pes; i++) {
            source[i] = my_pe * 1000 + i;
        }
        memset(dest, 0, total_size);
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup / 10; i++) {
            shmem_alltoall64(dest, source, nelems, 0, 0, n_pes, pSync);
        }
        shmem_barrier_all();
        
        /* Benchmark */
        double start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_alltoall64(dest, source, nelems, 0, 0, n_pes, pSync);
        }
        double end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / iterations;
        double bandwidth_mbps = (double)(total_size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        
        /* Basic validation */
        if (config->validate && my_pe == 0) {
            int valid = 1;
            for (int pe = 0; pe < n_pes && valid; pe++) {
                for (size_t elem = 0; elem < nelems && valid; elem++) {
                    long expected = pe * 1000 + (my_pe * nelems + elem);
                    if (dest[pe * nelems + elem] != expected) {
                        valid = 0;
                    }
                }
            }
            if (!valid) {
                printf(COLOR_RED "WARNING: Validation failed for alltoall\n" COLOR_RESET);
            }
        }
        
        if (my_pe == 0) {
            char size_str[32];
            format_size(total_size, size_str, sizeof(size_str));
            printf("%-12s %-15.2f %-15.2f %-12d\n",
                   size_str, latency_us, bandwidth_mbps, iterations);
        }
        
        shmem_free(source);
        shmem_free(dest);
    }
}

static void benchmark_collect(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    print_collective_header("COLLECT");
    
    for (size_t nelems = 1; nelems <= MAX_COLLECTIVE_SIZE / sizeof(long) / n_pes; nelems *= 2) {
        size_t source_size = nelems * sizeof(long);
        size_t dest_size = nelems * n_pes * sizeof(long);
        int iterations = calculate_iterations(dest_size, config->iterations / 10);
        
        long* source = (long*)shmem_malloc_aligned(source_size);
        long* dest = (long*)shmem_malloc_aligned(dest_size);
        
        if (!source || !dest) {
            fprintf(stderr, "Memory allocation failed\n");
            shmem_global_exit(1);
        }
        
        /* Initialize data */
        for (size_t i = 0; i < nelems; i++) {
            source[i] = my_pe * 1000 + i;
        }
        memset(dest, 0, dest_size);
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup / 10; i++) {
            shmem_collect64(dest, source, nelems, 0, 0, n_pes, pSync_collect);
        }
        shmem_barrier_all();
        
        /* Benchmark */
        double start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_collect64(dest, source, nelems, 0, 0, n_pes, pSync_collect);
        }
        double end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / iterations;
        double bandwidth_mbps = (double)(dest_size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        
        if (my_pe == 0) {
            char size_str[32];
            format_size(dest_size, size_str, sizeof(size_str));
            printf("%-12s %-15.2f %-15.2f %-12d\n",
                   size_str, latency_us, bandwidth_mbps, iterations);
        }
        
        shmem_free(source);
        shmem_free(dest);
    }
}

static void benchmark_fcollect(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    print_collective_header("FCOLLECT");
    
    for (size_t nelems = 1; nelems <= MAX_COLLECTIVE_SIZE / sizeof(long) / n_pes; nelems *= 2) {
        size_t source_size = nelems * sizeof(long);
        size_t dest_size = nelems * n_pes * sizeof(long);
        int iterations = calculate_iterations(dest_size, config->iterations / 10);
        
        long* source = (long*)shmem_malloc_aligned(source_size);
        long* dest = (long*)shmem_malloc_aligned(dest_size);
        
        if (!source || !dest) {
            fprintf(stderr, "Memory allocation failed\n");
            shmem_global_exit(1);
        }
        
        /* Initialize data */
        for (size_t i = 0; i < nelems; i++) {
            source[i] = my_pe * 1000 + i;
        }
        memset(dest, 0, dest_size);
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup / 10; i++) {
            shmem_fcollect64(dest, source, nelems, 0, 0, n_pes, pSync_collect);
        }
        shmem_barrier_all();
        
        /* Benchmark */
        double start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_fcollect64(dest, source, nelems, 0, 0, n_pes, pSync_collect);
        }
        double end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / iterations;
        double bandwidth_mbps = (double)(dest_size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        
        if (my_pe == 0) {
            char size_str[32];
            format_size(dest_size, size_str, sizeof(size_str));
            printf("%-12s %-15.2f %-15.2f %-12d\n",
                   size_str, latency_us, bandwidth_mbps, iterations);
        }
        
        shmem_free(source);
        shmem_free(dest);
    }
}

static void benchmark_broadcast(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    print_collective_header("BROADCAST");
    
    for (size_t nelems = 1; nelems <= MAX_COLLECTIVE_SIZE / sizeof(long); nelems *= 2) {
        size_t buffer_size = nelems * sizeof(long);
        int iterations = calculate_iterations(buffer_size, config->iterations / 10);
        
        long* buffer = (long*)shmem_malloc_aligned(buffer_size);
        
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed\n");
            shmem_global_exit(1);
        }
        
        /* Initialize data on root */
        if (my_pe == 0) {
            for (size_t i = 0; i < nelems; i++) {
                buffer[i] = 12345 + i;
            }
        } else {
            memset(buffer, 0, buffer_size);
        }
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup / 10; i++) {
            shmem_broadcast64(buffer, buffer, nelems, 0, 0, 0, n_pes, pSync_reduce);
        }
        shmem_barrier_all();
        
        /* Benchmark */
        double start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_broadcast64(buffer, buffer, nelems, 0, 0, 0, n_pes, pSync_reduce);
        }
        double end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / iterations;
        double bandwidth_mbps = (double)(buffer_size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        
        /* Validation */
        if (config->validate) {
            int valid = 1;
            for (size_t i = 0; i < nelems && valid; i++) {
                if (buffer[i] != (long)(12345 + i)) {
                    valid = 0;
                }
            }
            if (!valid && my_pe == 0) {
                printf(COLOR_RED "WARNING: Validation failed for broadcast\n" COLOR_RESET);
            }
        }
        
        if (my_pe == 0) {
            char size_str[32];
            format_size(buffer_size, size_str, sizeof(size_str));
            printf("%-12s %-15.2f %-15.2f %-12d\n",
                   size_str, latency_us, bandwidth_mbps, iterations);
        }
        
        shmem_free(buffer);
    }
}

static void benchmark_reduce(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    print_collective_header("REDUCE (SUM)");
    
    for (size_t nelems = 1; nelems <= (size_t)MAX_COLLECTIVE_SIZE / sizeof(long); nelems *= 2) {
        size_t buffer_size = nelems * sizeof(long);
        int iterations = calculate_iterations(buffer_size, config->iterations / 10);
        
        long* source = (long*)shmem_malloc_aligned(buffer_size);
        long* dest = (long*)shmem_malloc_aligned(buffer_size);
        long* pWrk = (long*)shmem_malloc_aligned(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(long));
        
        if (!source || !dest || !pWrk) {
            fprintf(stderr, "Memory allocation failed\n");
            shmem_global_exit(1);
        }
        
        /* Initialize data */
        for (size_t i = 0; i < nelems; i++) {
            source[i] = my_pe + 1; /* Each PE contributes (pe + 1) */
        }
        memset(dest, 0, buffer_size);
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup / 10; i++) {
            shmem_long_sum_to_all(dest, source, (int)nelems, 0, 0, n_pes, pWrk, pSync_reduce);
        }
        shmem_barrier_all();
        
        /* Benchmark */
        double start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_long_sum_to_all(dest, source, (int)nelems, 0, 0, n_pes, pWrk, pSync_reduce);
        }
        double end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / iterations;
        double bandwidth_mbps = (double)(buffer_size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        
        /* Validation - sum should be n_pes * (n_pes + 1) / 2 */
        if (config->validate) {
            long expected_sum = (long)n_pes * (n_pes + 1) / 2;
            int valid = 1;
            for (size_t i = 0; i < nelems && valid; i++) {
                if (dest[i] != expected_sum) {
                    valid = 0;
                }
            }
            if (!valid && my_pe == 0) {
                printf(COLOR_RED "WARNING: Validation failed for reduce (expected: %ld, got: %ld)\n" COLOR_RESET, 
                       expected_sum, dest[0]);
            }
        }
        
        if (my_pe == 0) {
            char size_str[32];
            format_size(buffer_size, size_str, sizeof(size_str));
            printf("%-12s %-15.2f %-15.2f %-12d\n",
                   size_str, latency_us, bandwidth_mbps, iterations);
        }
        
        shmem_free(source);
        shmem_free(dest);
        shmem_free(pWrk);
    }
}

static void benchmark_barrier(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    
    if (my_pe == 0) {
        printf("\n" COLOR_MAGENTA "BARRIER LATENCY TEST" COLOR_RESET "\n");
        printf("---------------------------------------------------------------\n");
    }
    
    const int iterations = 10000;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_barrier_all();
    }
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_barrier_all();
    }
    double end_time = TIME();
    
    double total_time_us = end_time - start_time;
    double latency_us = total_time_us / iterations;
    
    if (my_pe == 0) {
        printf("Barrier Latency: %.3f us\n", latency_us);
        printf("Barriers per second: %.2f K\n", 1e6 / latency_us / 1000);
    }
}

int main(int argc, char* argv[]) {
    shmem_init();
    
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "SHMEM Collective Operations Comprehensive Benchmark\n");
        printf("Running on %d PEs\n" COLOR_RESET, n_pes);
    }
    
    if (n_pes < 2) {
        if (my_pe == 0) {
            fprintf(stderr, COLOR_RED "This benchmark requires at least 2 PEs\n" COLOR_RESET);
        }
        shmem_finalize();
        return 1;
    }
    
    /* Initialize sync arrays */
    init_sync_arrays();
    shmem_barrier_all();
    
    test_config_t config = default_config;
    config.max_size = MAX_COLLECTIVE_SIZE;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            config.max_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-validate") == 0) {
            config.validate = 0;
        }
    }
    
    /* Run all collective benchmarks */
    benchmark_alltoall(&config);
    benchmark_collect(&config);
    benchmark_fcollect(&config);
    benchmark_broadcast(&config);
    benchmark_reduce(&config);
    benchmark_barrier(&config);
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "\nSHMEM Collective Operations Benchmark Complete\n" COLOR_RESET);
    }
    
    shmem_finalize();
    return 0;
} 