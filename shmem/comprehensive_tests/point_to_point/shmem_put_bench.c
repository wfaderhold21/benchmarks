#include "../common/common.h"

static void benchmark_put_latency_bandwidth(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    char* send_buf = (char*)shmem_malloc_aligned(config->max_size);
    char* recv_buf = (char*)shmem_malloc_aligned(config->max_size);
    
    if (!send_buf || !recv_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    if (my_pe == 0) {
        print_benchmark_header("SHMEM PUT LATENCY/BANDWIDTH");
    }
    
    for (size_t size = (size_t)config->min_size; size <= (size_t)config->max_size; size *= 2) {
        int iterations = calculate_iterations(size, config->iterations);
        double start_time, end_time;
        benchmark_result_t result = {0};
        
        /* Initialize data */
        init_data(send_buf, size, 'A' + my_pe);
        init_data(recv_buf, size, 'B');
        
        safe_barrier_all("put_bench_initialization");
        
        /* Warmup */
        SAFE_LOOP_WITH_TIMEOUT(i, 0, config->warmup, "put_bench_warmup")
            shmem_putmem(recv_buf, send_buf, size, target_pe);
        END_SAFE_LOOP()
        shmem_quiet();
        safe_barrier_all("put_bench_warmup_complete");
        
        /* Actual benchmark */
        start_time = TIME();
        SAFE_LOOP_WITH_TIMEOUT(j, 0, iterations, "put_bench_main")
            shmem_putmem(recv_buf, send_buf, size, target_pe);
        END_SAFE_LOOP()
        shmem_quiet();
        end_time = TIME();
        
        safe_barrier_all("put_bench_complete");
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        result.latency_us = total_time_us / iterations;
        result.bandwidth_mbps = (double)(size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        result.message_rate = iterations / (total_time_us / 1e6) / 1e6; /* Million messages per second */
        result.message_size = size;
        result.iterations = iterations;
        
        /* Validation - check if data was received correctly */
        if (config->validate) {
            /* After PUT operations, recv_buf should contain data from the PE that put to us */
            int source_pe = (my_pe - 1 + n_pes) % n_pes;  /* PE that puts to us */
            char expected_pattern = 'A' + source_pe;       /* Pattern from source PE */

            int valid = 1;
            for (size_t i = 0; i < size; i++) {
                if (recv_buf[i] != expected_pattern) {
                    valid = 0;
                    break;
                }
            }

            /* Record validation result */
            record_validation_result(valid, "PUT", size, my_pe);

            if (!valid) {
                printf(COLOR_RED "[PE %d] VALIDATION FAILED for size %zu: expected 0x%02x, got 0x%02x" COLOR_RESET "\n", 
                       my_pe, size, (unsigned char)expected_pattern, (unsigned char)recv_buf[0]);
            }
        }
        
        /* Print results from PE 0 */
        if (my_pe == 0) {
            print_benchmark_result(&result);
        }
    }
    
    shmem_free(send_buf);
    shmem_free(recv_buf);
}

static void benchmark_put_message_rate(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    /* For message rate, use small fixed-size messages */
    const int rate_iterations = 1000000; /* High iteration count for message rate */
    
    long* send_buf = (long*)shmem_malloc_aligned(sizeof(long));
    long* recv_buf = (long*)shmem_malloc_aligned(sizeof(long));
    
    if (!send_buf || !recv_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    if (my_pe == 0) {
        printf("\n" COLOR_YELLOW "SHMEM PUT MESSAGE RATE (8-byte messages)" COLOR_RESET "\n");
        printf("---------------------------------------------------------------\n");
    }
    
    *send_buf = 0x123456789ABCDEF0LL + my_pe;
    *recv_buf = 0;
    
    safe_barrier_all("put_message_rate_init");
    
    /* Warmup */
    SAFE_LOOP_WITH_TIMEOUT(i, 0, config->warmup, "put_message_rate_warmup")
        shmem_long_put(recv_buf, send_buf, 1, target_pe);
    END_SAFE_LOOP()
    shmem_quiet();
    safe_barrier_all("put_message_rate_warmup_complete");
    
    /* Message rate benchmark */
    double start_time = TIME();
    SAFE_LOOP_WITH_TIMEOUT(j, 0, rate_iterations, "put_message_rate_main")
        shmem_long_put(recv_buf, send_buf, 1, target_pe);
    END_SAFE_LOOP()
    shmem_quiet();
    double end_time = TIME();
    
    safe_barrier_all("put_message_rate_complete");
    
    double total_time_us = end_time - start_time;
    double message_rate = rate_iterations / (total_time_us / 1e6) / 1e6; /* Million messages per second */
    double latency_us = total_time_us / rate_iterations;
    
    if (my_pe == 0) {
        printf("Message Rate: %.2f M messages/sec\n", message_rate);
        printf("Average Latency: %.2f us\n", latency_us);
        printf("Total Messages: %d\n", rate_iterations);
        printf("Total Time: %.2f ms\n", total_time_us / 1000);
    }
    
    shmem_free(send_buf);
    shmem_free(recv_buf);
}

int main(int argc, char* argv[]) {
    shmem_init();
    
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "SHMEM PUT Comprehensive Benchmark\n");
        printf("Running on %d PEs\n" COLOR_RESET, n_pes);
    }
    
    if (n_pes < 2) {
        if (my_pe == 0) {
            fprintf(stderr, COLOR_RED "This benchmark requires at least 2 PEs\n" COLOR_RESET);
        }
        shmem_finalize();
        return 1;
    }
    
    /* Verify PE connectivity before starting benchmarks */
    if (!verify_pe_connectivity()) {
        if (my_pe == 0) {
            fprintf(stderr, COLOR_RED "PE connectivity test failed\n" COLOR_RESET);
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
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            config.timeout_seconds = atoi(argv[++i]);
        }
    }
    
    /* Initialize validation tracking */
    reset_validation_counters();

    /* Start test timer */
    start_test_timer();

    /* Run benchmarks with timeout protection */
    if (my_pe == 0) {
        printf(COLOR_BLUE "[INFO] Starting benchmarks with %d second timeout\n" COLOR_RESET,
               config.timeout_seconds);
    }

    benchmark_put_latency_bandwidth(&config);
    benchmark_put_message_rate(&config);
    
    /* Print validation summary */
    print_validation_summary();

    /* Check for validation failures and exit accordingly */
    int exit_code = 0;
    if (has_validation_failures()) {
        if (my_pe == 0) {
            printf(COLOR_RED "\nSHMEM PUT Benchmark FAILED due to validation errors\n" COLOR_RESET);
        }
        exit_code = 1;
    } else {
        if (my_pe == 0) {
            printf(COLOR_GREEN "\nSHMEM PUT Benchmark Complete\n" COLOR_RESET);
        }
    }
    
    shmem_finalize();
    return exit_code;
}
