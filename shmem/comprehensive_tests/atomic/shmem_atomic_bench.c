#include "../common/common.h"

typedef struct {
    const char* name;
    double latency_us;
    double ops_per_sec;
    int iterations;
} atomic_result_t;

static void print_atomic_header() {
    int my_pe = shmem_my_pe();
    if (my_pe == 0) {
        printf("\n" COLOR_CYAN "===============================================\n");
        printf(" SHMEM ATOMIC OPERATIONS BENCHMARK RESULTS\n");
        printf("===============================================" COLOR_RESET "\n");
        printf("%-20s %-15s %-15s %-10s\n", 
               "Operation", "Latency(us)", "Ops/sec(M)", "Iterations");
        printf("---------------------------------------------------------------\n");
    }
}

static void print_atomic_result(const atomic_result_t* result) {
    printf("%-20s %-15.3f %-15.3f %-10d\n",
           result->name,
           result->latency_us,
           result->ops_per_sec / 1e6,
           result->iterations);
}

static void benchmark_atomic_fetch_add(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_counter = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_counter) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_counter = 0;
    shmem_barrier_all();
    
    const int iterations = 100000;
    atomic_result_t result = {0};
    result.name = "atomic_fetch_add";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_fetch_add(remote_counter, 1, target_pe);
    }
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_long_atomic_fetch_add(remote_counter, 1, target_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_counter);
}

static void benchmark_atomic_add(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_counter = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_counter) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_counter = 0;
    shmem_barrier_all();
    
    const int iterations = 100000;
    atomic_result_t result = {0};
    result.name = "atomic_add";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_add(remote_counter, 1, target_pe);
    }
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_long_atomic_add(remote_counter, 1, target_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_counter);
}

static void benchmark_atomic_compare_swap(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_var = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_var) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_var = 0;
    shmem_barrier_all();
    
    const int iterations = 50000; /* Lower iterations due to higher latency */
    atomic_result_t result = {0};
    result.name = "atomic_compare_swap";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_compare_swap(remote_var, i, i + 1, target_pe);
    }
    shmem_barrier_all();
    
    /* Reset for benchmark */
    *remote_var = 0;
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        /* Alternate between values to ensure some swaps succeed */
        long expected = i % 2;
        long new_val = (i + 1) % 2;
        shmem_long_atomic_compare_swap(remote_var, expected, new_val, target_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_var);
}

static void benchmark_atomic_swap(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_var = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_var) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_var = 0;
    shmem_barrier_all();
    
    const int iterations = 100000;
    atomic_result_t result = {0};
    result.name = "atomic_swap";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_swap(remote_var, i, target_pe);
    }
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_long_atomic_swap(remote_var, i, target_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_var);
}

static void benchmark_atomic_fetch(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_var = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_var) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_var = 0x123456789ABCDEF0LL;
    shmem_barrier_all();
    
    const int iterations = 100000;
    atomic_result_t result = {0};
    result.name = "atomic_fetch";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_fetch(remote_var, target_pe);
    }
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        volatile long val = shmem_long_atomic_fetch(remote_var, target_pe);
        (void)val; /* Prevent optimization */
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_var);
}

static void benchmark_atomic_inc(const test_config_t* config) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    int target_pe = (my_pe + 1) % n_pes;
    
    long* remote_counter = (long*)shmem_malloc_aligned(sizeof(long));
    if (!remote_counter) {
        fprintf(stderr, "Memory allocation failed\n");
        shmem_global_exit(1);
    }
    
    *remote_counter = 0;
    shmem_barrier_all();
    
    const int iterations = 100000;
    atomic_result_t result = {0};
    result.name = "atomic_inc";
    result.iterations = iterations;
    
    /* Warmup */
    for (int i = 0; i < config->warmup; i++) {
        shmem_long_atomic_inc(remote_counter, target_pe);
    }
    shmem_barrier_all();
    
    /* Benchmark */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_long_atomic_inc(remote_counter, target_pe);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    result.latency_us = total_time_us / iterations;
    result.ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        print_atomic_result(&result);
    }
    
    shmem_free(remote_counter);
}

static void benchmark_contention_test(const test_config_t* config __attribute__((unused))) {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (n_pes < 4) return; /* Need at least 4 PEs for meaningful contention */
    
    /* Shared counter that all PEs will contend for */
    static long shared_counter = 0;
    const int iterations = 10000;
    
    if (my_pe == 0) {
        printf("\n" COLOR_BLUE "ATOMIC CONTENTION TEST (%d PEs)" COLOR_RESET "\n", n_pes);
        printf("---------------------------------------------------------------\n");
    }
    
    shmem_barrier_all();
    
    /* All PEs increment the same counter on PE 0 */
    double start_time = TIME();
    for (int i = 0; i < iterations; i++) {
        shmem_long_atomic_fetch_add(&shared_counter, 1, 0);
    }
    double end_time = TIME();
    
    shmem_barrier_all();
    
    double total_time_us = end_time - start_time;
    double avg_latency = total_time_us / iterations;
    double ops_per_sec = iterations / (total_time_us / 1e6);
    
    if (my_pe == 0) {
        printf("Contended Operations per PE: %d\n", iterations);
        printf("Average Latency per PE: %.3f us\n", avg_latency);
        printf("Operations per second per PE: %.2f M\n", ops_per_sec / 1e6);
        printf("Final counter value: %ld (expected: %ld)\n", 
               shared_counter, (long)iterations * n_pes);
        
        if (shared_counter == (long)iterations * n_pes) {
            printf(COLOR_GREEN "PASS: Counter value is correct\n" COLOR_RESET);
        } else {
            printf(COLOR_RED "FAIL: Counter value is incorrect\n" COLOR_RESET);
        }
    }
}

int main(int argc, char* argv[]) {
    shmem_init();
    
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "SHMEM Atomic Operations Comprehensive Benchmark\n");
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
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup = atoi(argv[++i]);
        }
    }
    
    print_atomic_header();
    
    /* Run all atomic operation benchmarks */
    benchmark_atomic_fetch_add(&config);
    benchmark_atomic_add(&config);
    benchmark_atomic_compare_swap(&config);
    benchmark_atomic_swap(&config);
    benchmark_atomic_fetch(&config);
    benchmark_atomic_inc(&config);
    
    /* Run contention test */
    benchmark_contention_test(&config);
    
    if (my_pe == 0) {
        printf(COLOR_GREEN "\nSHMEM Atomic Operations Benchmark Complete\n" COLOR_RESET);
    }
    
    shmem_finalize();
    return 0;
} 