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

/* Timeout and safety constants */
#define MAX_TEST_TIME_SECONDS    120    /* 2 minutes max per test */
#define MAX_TEST_TIME_US         (MAX_TEST_TIME_SECONDS * 1000000)
#define TIMEOUT_CHECK_INTERVAL   1000   /* Check timeout every N iterations */
#define HEALTH_CHECK_INTERVAL    5000   /* PE health check interval */

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

/* Global timeout tracking */
static double g_test_start_time = 0.0;

/* Global validation tracking */
static int g_validation_failures = 0;
static int g_total_validations = 0;

/* Timeout and health check utilities */
static inline void start_test_timer() {
    g_test_start_time = TIME();
}

static inline int check_test_timeout(const char* test_name) {
    if (g_test_start_time == 0.0) return 0; /* Timer not started */

    double elapsed = TIME() - g_test_start_time;
    if (elapsed > MAX_TEST_TIME_US) {
        int my_pe = shmem_my_pe();
        fprintf(stderr, COLOR_RED "[PE %d] TIMEOUT: %s exceeded %.1f seconds\n" COLOR_RESET,
                my_pe, test_name, elapsed / 1e6);
        fflush(stderr);
        shmem_global_exit(1);
        return 1;
    }
    return 0;
}

/* PE health and connectivity check */
static inline int verify_pe_connectivity() {
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();

    /* Simple connectivity test using a shared counter */
    static long connectivity_test = 0;

    if (my_pe == 0) {
        connectivity_test = 0;
        printf(COLOR_BLUE "[INFO] Verifying connectivity of %d PEs...\n" COLOR_RESET, n_pes);
    }

    shmem_barrier_all();

    /* Each PE increments the counter on PE 0 */
    shmem_long_atomic_inc(&connectivity_test, 0);

    shmem_barrier_all();

    /* PE 0 checks if all PEs participated */
    if (my_pe == 0) {
        if (connectivity_test == n_pes) {
            printf(COLOR_GREEN "[SUCCESS] All %d PEs are connected and responsive\n" COLOR_RESET, n_pes);
            return 1;
        } else {
            printf(COLOR_RED "[ERROR] PE connectivity test failed: expected %d, got %ld\n" COLOR_RESET, 
                   n_pes, connectivity_test);
            return 0;
        }
    }
    return 1;
}

/* Safer barrier with timeout */
static inline void safe_barrier_all(const char* location) {
    check_test_timeout(location);
    shmem_barrier_all();
}

/* Validation tracking functions */
static inline void reset_validation_counters() {
    g_validation_failures = 0;
    g_total_validations = 0;
}

static inline int record_validation_result(int is_valid, const char* test_name, size_t size, int pe) {
    g_total_validations++;
    if (!is_valid) {
        g_validation_failures++;
        return 0;
    }
    return 1;
}

static inline void print_validation_summary() {
    int my_pe = shmem_my_pe();
    if (my_pe == 0) {
        if (g_validation_failures == 0) {
            printf(COLOR_GREEN "[VALIDATION] All %d validation checks PASSED\n" COLOR_RESET, g_total_validations);
        } else {
            printf(COLOR_RED "[VALIDATION] %d/%d validation checks FAILED (%.1f%% failure rate)\n" COLOR_RESET,
                   g_validation_failures, g_total_validations, 
                   (100.0 * g_validation_failures) / g_total_validations);
        }
    }
}

static inline int has_validation_failures() {
    return g_validation_failures > 0;
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
    int timeout_seconds;       /* Per-test timeout */
    int enable_health_checks;  /* Enable PE health monitoring */
} test_config_t;

/* Default test configuration */
static const test_config_t default_config = {
    .min_size = MIN_MSG_SIZE,
    .max_size = MAX_MSG_SIZE,
    .iterations = MAX_ITER,
    .warmup = WARMUP,
    .skip = DEFAULT_SKIP,
    .validate = 1,
    .timeout_seconds = MAX_TEST_TIME_SECONDS,
    .enable_health_checks = 1
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

/* Enhanced benchmark runner with timeout protection */
static inline void run_benchmark_with_timeout(
    const char* test_name,
    void (*benchmark_func)(const test_config_t*),
    const test_config_t* config) {

    int my_pe = shmem_my_pe();

    if (my_pe == 0) {
        printf(COLOR_BLUE "[INFO] Starting %s with %d second timeout\n" COLOR_RESET,
               test_name, config->timeout_seconds);
    }

    /* Start timeout timer */
    start_test_timer();

    /* Run the benchmark */
    benchmark_func(config);

    if (my_pe == 0) {
        printf(COLOR_GREEN "[SUCCESS] %s completed successfully\n" COLOR_RESET, test_name);
    }
}

/* Emergency exit function for timeout situations */
static inline void emergency_exit(const char* reason) {
    int my_pe = shmem_my_pe();
    fprintf(stderr, COLOR_RED "[PE %d] EMERGENCY EXIT: %s\n" COLOR_RESET, my_pe, reason);
    fflush(stderr);

    /* Try graceful exit first */
    shmem_global_exit(1);

    /* If that fails, force exit */
    exit(1);
}

/* Macro for safer loops with timeout checking */
#define SAFE_LOOP_WITH_TIMEOUT(iter_var, start_val, end_val, test_name) \
    for (int iter_var = start_val; iter_var < end_val; iter_var++) { \
        if ((iter_var % TIMEOUT_CHECK_INTERVAL) == 0) { \
            if (check_test_timeout(test_name)) break; \
        }

#define END_SAFE_LOOP() }

#endif /* SHMEM_COMMON_H */
