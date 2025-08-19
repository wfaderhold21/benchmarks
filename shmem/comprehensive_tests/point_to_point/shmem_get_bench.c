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
        
        /* Initialize data - each PE sets up its own buffer with its PE ID */
        init_data(remote_buf, size, 'R' + my_pe);  /* Remote buffer identifies its owner */
        init_data(local_buf, size, 'L');
        
        safe_barrier_all("get_bench_initialization");
        
        /* Warmup */
        SAFE_LOOP_WITH_TIMEOUT(i, 0, config->warmup, "get_bench_warmup")
            shmem_getmem(local_buf, remote_buf, size, source_pe);
        END_SAFE_LOOP()
        safe_barrier_all("get_bench_warmup_complete");
        
        /* Actual benchmark */
        start_time = TIME();
        SAFE_LOOP_WITH_TIMEOUT(j, 0, iterations, "get_bench_main")
            shmem_getmem(local_buf, remote_buf, size, source_pe);
        END_SAFE_LOOP()
        end_time = TIME();
        
        safe_barrier_all("get_bench_complete");
        
        /* Calculate metrics */
        double total_time_us = end_time - start_time;
        result.latency_us = total_time_us / iterations;
        result.bandwidth_mbps = (double)(size * iterations) / (total_time_us / 1e6) / (1024 * 1024);
        result.message_rate = iterations / (total_time_us / 1e6) / 1e6; /* Million messages per second */
        result.message_size = size;
        result.iterations = iterations;
        
        /* Validation - check if fetched data matches source PE's pattern */
        if (config->validate) {
            char expected_pattern = 'R' + source_pe;  /* Should match source PE's remote buffer */
            int valid = 1;
            for (size_t i = 0; i < size; i++) {
                if (local_buf[i] != expected_pattern) {
                    valid = 0;
                    break;
                }
            }

            /* Record validation result */
            record_validation_result(valid, "GET", size, my_pe);

            if (!valid) {
                printf(COLOR_RED "[PE %d] VALIDATION FAILED for size %zu: expected 0x%02x, got 0x%02x" COLOR_RESET "\n", 
                       my_pe, size, (unsigned char)expected_pattern, (unsigned char)local_buf[0]);
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
    
    safe_barrier_all("get_message_rate_init");
    
    /* Warmup */
    SAFE_LOOP_WITH_TIMEOUT(i, 0, config->warmup, "get_message_rate_warmup")
        shmem_long_get(local_buf, remote_buf, 1, source_pe);
    END_SAFE_LOOP()
    safe_barrier_all("get_message_rate_warmup_complete");
    
    /* Message rate benchmark */
    double start_time = TIME();
    SAFE_LOOP_WITH_TIMEOUT(j, 0, rate_iterations, "get_message_rate_main")
        shmem_long_get(local_buf, remote_buf, 1, source_pe);
    END_SAFE_LOOP()
    double end_time = TIME();
    
    safe_barrier_all("get_message_rate_complete");
    
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
    
    /* Only PE 0 and PE 1 do the actual bi-directional test work,
       but ALL PEs must participate in barriers to avoid deadlock */
    int do_bidirectional_work = (my_pe < 2);
    
    int partner_pe = (my_pe < 2) ? (1 - my_pe) : 0; /* PE 0 <-> PE 1, others use safe value */
    const size_t test_size = 1024; /* Fixed size for bi-directional test */
    const int test_iterations = 10000;

    char* local_buf = (char*)shmem_malloc_aligned(test_size);
    char* remote_buf = (char*)shmem_malloc_aligned(test_size);
    
    if (!local_buf || !remote_buf) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }

    /* Initialize data */
    init_data(remote_buf, test_size, 'R' + my_pe);
    init_data(local_buf, test_size, 'L');
    
    if (my_pe == 0) {
        printf("\n" COLOR_MAGENTA "SHMEM GET BI-DIRECTIONAL TEST (1KB messages)" COLOR_RESET "\n");
        printf("---------------------------------------------------------------\n");
    }
    
    safe_barrier_all("get_bi_directional_init");
    
    /* Warmup - only active PEs do SHMEM operations */
    if (do_bidirectional_work) {
        SAFE_LOOP_WITH_TIMEOUT(i, 0, config->warmup, "get_bi_directional_warmup")
            shmem_getmem(local_buf, remote_buf, test_size, partner_pe);
        END_SAFE_LOOP()
    }
    safe_barrier_all("get_bi_directional_warmup_complete");
    
    /* Sequential bi-directional test to avoid deadlocks */
    double start_time = TIME();

    /* PE 0 goes first */
    if (my_pe == 0) {
        SAFE_LOOP_WITH_TIMEOUT(j, 0, test_iterations, "get_bi_directional_pe0")
            shmem_getmem(local_buf, remote_buf, test_size, partner_pe);
        END_SAFE_LOOP()
    }

    /* All PEs synchronize before PE 1 starts */
    safe_barrier_all("get_bi_directional_pe0_complete");

    /* PE 1 goes second */
    if (my_pe == 1) {
        SAFE_LOOP_WITH_TIMEOUT(j, 0, test_iterations, "get_bi_directional_pe1")
            shmem_getmem(local_buf, remote_buf, test_size, partner_pe);
        END_SAFE_LOOP()
    }

    double end_time = TIME();
    safe_barrier_all("get_bi_directional_complete");

    /* Only PE 0 calculates and prints results */
    if (my_pe == 0) {
        double total_time_us = end_time - start_time;
        double latency_us = total_time_us / test_iterations;
        double bandwidth_mbps = (double)(test_size * test_iterations) / (total_time_us / 1e6) / (1024 * 1024);

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

    benchmark_get_latency_bandwidth(&config);
    benchmark_get_message_rate(&config);
    benchmark_get_bi_directional(&config);

    /* Print validation summary */
    print_validation_summary();

    /* Check for validation failures and exit accordingly */
    int exit_code = 0;
    if (has_validation_failures()) {
        if (my_pe == 0) {
            printf(COLOR_RED "\nSHMEM GET Benchmark FAILED due to validation errors\n" COLOR_RESET);
        }
        exit_code = 1;
    } else {
        if (my_pe == 0) {
            printf(COLOR_GREEN "\nSHMEM GET Benchmark Complete\n" COLOR_RESET);
        }
    }

    shmem_finalize();
    return exit_code;
}
