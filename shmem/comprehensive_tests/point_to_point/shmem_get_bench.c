#include "../common/common.h"

static void benchmark_get_latency_bandwidth(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int source_pe = (my_pe + 1) % n_pes;
    
    char* local_buf = (char*)shmem_malloc_aligned(config->max_size);
    char* remote_buf = (char*)shmem_malloc_aligned(config->max_size);
    
    if (!local_buf || !remote_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    if (my_pe == 0) {
        print_benchmark_header("SHMEM GET LATENCY/BANDWIDTH");
    }
    
    for (size_t size = (size_t)config->min_size; size <= (size_t)config->max_size; size *= 2) {
        int iterations = calculate_iterations(size, config->iterations);
        double start_time, end_time;
        benchmark_result_t result = {0};
        
        /* Initialize data - source PE sets up data to be fetched */
        init_data(remote_buf, size, 'S' + source_pe);
        init_data(local_buf, size, 'L');
        
        shmem_barrier_all();
        
        /* Warmup */
        for (int i = 0; i < config->warmup; i++) {
            shmem_getmem(local_buf, remote_buf, size, source_pe);
        }
        shmem_barrier_all();
        
        /* Actual benchmark */
        start_time = TIME();
        for (int i = 0; i < iterations; i++) {
            shmem_getmem(local_buf, remote_buf, size, source_pe);
        }
        end_time = TIME();
        
        shmem_barrier_all();
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        result.latency_us = total_time_us / iterations;
        result.bandwidth_mbps = (double)(size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        result.message_rate = iterations / (total_time_us / 1e6) / 1e6; /* Million messages per second */
        result.message_size = size;
        result.iterations = iterations;
        
        /* Validation - check if fetched data matches expected */
        if (config->validate) {
            char expected_pattern = 'S' + source_pe;
            int valid = 1;
            for (size_t i = 0; i < size; i++) {
                if (local_buf[i] != expected_pattern) {
                    valid = 0;
                    break;
                }
            }
            if (!valid && my_pe == 0) {
                printf(COLOR_RED "WARNING: Data validation failed for size %zu" COLOR_RESET "\n", size);
            }
        }
        
        /* Print results from PE 0 */
        if (my_pe == 0) {
            print_benchmark_result(&result);
        }
    }
    
    shmem_free(local_buf);
    shmem_free(remote_buf);
}

static void benchmark_get_message_rate(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int source_pe = (my_pe + 1) % n_pes;
    
    /* For message rate, use small fixed-size messages */
    const int rate_iterations = 1000000; /* High iteration count for message rate */
    
    long* local_buf = (long*)shmem_malloc_aligned(sizeof(long));
    long* remote_buf = (long*)shmem_malloc_aligned(sizeof(long));
    
    if (!local_buf || !remote_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    if (my_pe == 0) {
        printf("\n" COLOR_YELLOW "SHMEM GET MESSAGE RATE (8-byte messages)" COLOR_RESET "\n");
        printf("---------------------------------------------------------------\n");
    }
    
    *remote_buf = 0xFEDCBA9876543210LL + source_pe;
    *local_buf = 0;
    
    shmem_barrier_all();
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_get(local_buf, remote_buf, 1, source_pe);
    }
    shmem_barrier_all();
    
    /* Message rate benchmark */
    double start_time = TIME();
    for (int i = 0; i < rate_iterations; i++) {
        shmem_long_get(local_buf, remote_buf, 1, source_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    double message_rate = rate_iterations / (total_time_us / 1e6) / 1e6; /* Million messages per second */
    double latency_us = total_time_us / rate_iterations;
    
    if (my_pe == 0) {
        printf("Message Rate: %.2f M messages/sec\n", message_rate);
        printf("Average Latency: %.2f us\n", latency_us);
        printf("Total Messages: %d\n", rate_iterations);
        printf("Total Time: %.2f ms\n", total_time_us / 1000);
    }
    
    shmem_free(local_buf);
    shmem_free(remote_buf);
}

static void benchmark_get_bi_directional(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (n_pes < 2) return;
    
    /* Only run on PE pairs for bi-directional testing */
    if (my_pe >= 2) return;
    
    int partner_pe = 1 - my_pe; /* PE 0 <-> PE 1 */
    const size_t test_size = 1024; /* Fixed size for bi-directional test */
    const int test_iterations = 10000;
    
    char* local_buf = (char*)shmem_malloc_aligned(test_size);
    char* remote_buf = (char*)shmem_malloc_aligned(test_size);
    
    if (!local_buf || !remote_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    if (my_pe == 0) {
        printf("\n" COLOR_MAGENTA "SHMEM GET BI-DIRECTIONAL TEST (1KB messages)" COLOR_RESET "\n");
        printf("---------------------------------------------------------------\n");
    }
    
    /* Initialize data */
    init_data(remote_buf, test_size, 'R' + my_pe);
    init_data(local_buf, test_size, 'L');
    
    shmem_barrier_all();
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_getmem(local_buf, remote_buf, test_size, partner_pe);
    }
    shmem_barrier_all();
    
    /* Bi-directional test - both PEs get simultaneously */
    double start_time = TIME();
    for (int i = 0; i < test_iterations; i++) {
        shmem_getmem(local_buf, remote_buf, test_size, partner_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    double latency_us = total_time_us / test_iterations;
    double bandwidth_mbps = (double)(test_size * test_iterations) / (total_time_us / 1e6) / (1024 * 1024);
    
    if (my_pe == 0) {
        printf("PE %d <-> PE %d Latency: %.2f us\n", my_pe, partner_pe, latency_us);
        printf("PE %d <-> PE %d Bandwidth: %.2f MB/s\n", my_pe, partner_pe, bandwidth_mbps);
    }
    
    shmem_free(local_buf);
    shmem_free(remote_buf);
}

int main(int argc, char* argv[]) {
    shmem_init();
    
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "SHMEM GET Comprehensive Benchmark\n");
        printf("Running on %d PEs\n" COLOR_RESET, n_pes);
    }
    
    if (n_pes < 2) {
        if (my_pe == 0) {
            fprintf(stderr, COLOR_RED "This benchmark requires at least 2 PEs\n" COLOR_RESET);
        }
        shmem_finalize();
        return 1;
    }
    
    test_config_t config = default_config;
    
    /* Parse command line arguments if needed */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            config.max_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-validate") == 0) {
            config.validate = 0;
        }
    }
    
    /* Run benchmarks */
    benchmark_get_latency_bandwidth(&config);
    benchmark_get_message_rate(&config);
    benchmark_get_bi_directional(&config);
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "\nSHMEM GET Benchmark Complete\n" COLOR_RESET);
    }
    
    shmem_finalize();
    return 0;
} 