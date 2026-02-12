#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <shmem.h>

/* Sync and work arrays for reductions */
static long pSync_reduce[SHMEM_REDUCE_SYNC_SIZE];
static long pWrk_reduce[SHMEM_REDUCE_MIN_WRKDATA_SIZE];

/* Function to get current memory usage in KB from /proc/self/status */
static long get_memory_usage_kb(void) {
    FILE *file;
    char line[256];
    long vm_rss = 0;
    
    file = fopen("/proc/self/status", "r");
    if (file == NULL) {
        return -1;
    }
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %ld", &vm_rss);
            break;
        }
    }
    
    fclose(file);
    return vm_rss;
}

/* Function to format memory size for display */
static void format_memory_size(long kb, char *buffer, size_t buffer_size) {
    if (kb < 1024) {
        snprintf(buffer, buffer_size, "%ld KB", kb);
    } else if (kb < 1024 * 1024) {
        snprintf(buffer, buffer_size, "%.2f MB", (double)kb / 1024.0);
    } else {
        snprintf(buffer, buffer_size, "%.2f GB", (double)kb / (1024.0 * 1024.0));
    }
}

/* Function to get current time in microseconds */
static double get_time_us(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        return -1.0;
    }
    return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

/* Function to format time for display */
static void format_time(double us, char *buffer, size_t buffer_size) {
    if (us < 1000.0) {
        snprintf(buffer, buffer_size, "%.2f us", us);
    } else if (us < 1000000.0) {
        snprintf(buffer, buffer_size, "%.2f ms", us / 1000.0);
    } else {
        snprintf(buffer, buffer_size, "%.2f s", us / 1000000.0);
    }
}

/* Initialize sync arrays for reductions */
static void init_sync_arrays(void) {
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync_reduce[i] = SHMEM_SYNC_VALUE;
    }
}

/* Helper function to reset sync arrays before a reduction */
static void reset_sync_for_reduction(void) {
    init_sync_arrays();
    shmem_barrier_all();
}

/* Print statistics for a memory measurement */
static void print_memory_stats(const char *label, long max_val, long min_val, long sum_val, int n_pes) {
    long avg_val = sum_val / n_pes;
    char max_str[64], min_str[64], avg_str[64];
    
    format_memory_size(max_val, max_str, sizeof(max_str));
    format_memory_size(min_val, min_str, sizeof(min_str));
    format_memory_size(avg_val, avg_str, sizeof(avg_str));
    
    printf("  %-30s Max: %-12s Min: %-12s Avg: %-12s\n", 
           label, max_str, min_str, avg_str);
}

/* Print statistics for a timing measurement */
static void print_timing_stats(const char *label, double max_val, double min_val, double sum_val, int n_pes) {
    double avg_val = sum_val / n_pes;
    char max_str[64], min_str[64], avg_str[64];
    
    format_time(max_val, max_str, sizeof(max_str));
    format_time(min_val, min_str, sizeof(min_str));
    format_time(avg_val, avg_str, sizeof(avg_str));
    
    printf("  %-30s Max: %-12s Min: %-12s Avg: %-12s\n", 
           label, max_str, min_str, avg_str);
}

int main(int argc, char *argv[]) {
    /* Static variables for symmetric memory (required for OpenSHMEM reductions) */
    static long mem_before_init = 0;
    static long mem_after_init = 0;
    static long mem_after_barrier = 0;
    static long mem_init_overhead = 0;
    static long mem_barrier_overhead = 0;
    
    /* Timing variables - use long for reductions (convert microseconds to long) */
    static double time_before_init = 0.0;
    static double time_after_init = 0.0;
    static double init_duration = 0.0;
    static long init_duration_us = 0;  /* Convert to long for reductions */
    
    /* Reduction results - static for symmetric memory */
    static long max_before, min_before, sum_before;
    static long max_after_init, min_after_init, sum_after_init;
    static long max_init_overhead, min_init_overhead, sum_init_overhead;
    static long max_after_barrier, min_after_barrier, sum_after_barrier;
    static long max_barrier_overhead, min_barrier_overhead, sum_barrier_overhead;
    static long max_init_duration, min_init_duration, sum_init_duration;
    
    /* Temporary result variable for reductions - static for symmetric memory */
    static long temp_result;
    
    /* Measure memory and time before shmem_init */
    mem_before_init = get_memory_usage_kb();
    if (mem_before_init < 0) {
        fprintf(stderr, "Error: Failed to read memory usage before init\n");
        return 1;
    }
    
    time_before_init = get_time_us();
    if (time_before_init < 0.0) {
        fprintf(stderr, "Error: Failed to get time before init\n");
        return 1;
    }
    
    /* Initialize OpenSHMEM */
    shmem_init();
    
    time_after_init = get_time_us();
    if (time_after_init < 0.0) {
        fprintf(stderr, "Error: Failed to get time after init\n");
        shmem_finalize();
        return 1;
    }
    
    init_duration = time_after_init - time_before_init;
    init_duration_us = (long)init_duration;
    
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();
    
    /* Initialize sync arrays */
    init_sync_arrays();
    
    /* Measure memory after shmem_init */
    mem_after_init = get_memory_usage_kb();
    if (mem_after_init < 0) {
        fprintf(stderr, "[PE %d] Error: Failed to read memory usage after init\n", my_pe);
        shmem_finalize();
        return 1;
    }
    
    mem_init_overhead = mem_after_init - mem_before_init;
    
    /* Perform a barrier to ensure all PEs have initialized */
    shmem_barrier_all();
    
    /* Measure memory after barrier */
    mem_after_barrier = get_memory_usage_kb();
    if (mem_after_barrier < 0) {
        fprintf(stderr, "[PE %d] Error: Failed to read memory usage after barrier\n", my_pe);
        shmem_finalize();
        return 1;
    }
    
    mem_barrier_overhead = mem_after_barrier - mem_after_init;
    
    /* Memory before init - max, min, sum */
    reset_sync_for_reduction();
    temp_result = mem_before_init;
    shmem_long_max_to_all(&temp_result, &mem_before_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_before = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_before_init;
    shmem_long_min_to_all(&temp_result, &mem_before_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_before = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_before_init;
    shmem_long_sum_to_all(&temp_result, &mem_before_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_before = temp_result;
    
    /* Memory after init - max, min, sum */
    reset_sync_for_reduction();
    temp_result = mem_after_init;
    shmem_long_max_to_all(&temp_result, &mem_after_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_after_init = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_after_init;
    shmem_long_min_to_all(&temp_result, &mem_after_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_after_init = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_after_init;
    shmem_long_sum_to_all(&temp_result, &mem_after_init, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_after_init = temp_result;
    
    /* Init overhead - max, min, sum */
    reset_sync_for_reduction();
    temp_result = mem_init_overhead;
    shmem_long_max_to_all(&temp_result, &mem_init_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_init_overhead = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_init_overhead;
    shmem_long_min_to_all(&temp_result, &mem_init_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_init_overhead = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_init_overhead;
    shmem_long_sum_to_all(&temp_result, &mem_init_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_init_overhead = temp_result;
    
    /* Memory after barrier - max, min, sum */
    reset_sync_for_reduction();
    temp_result = mem_after_barrier;
    shmem_long_max_to_all(&temp_result, &mem_after_barrier, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_after_barrier = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_after_barrier;
    shmem_long_min_to_all(&temp_result, &mem_after_barrier, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_after_barrier = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_after_barrier;
    shmem_long_sum_to_all(&temp_result, &mem_after_barrier, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_after_barrier = temp_result;
    
    /* Barrier overhead - max, min, sum */
    reset_sync_for_reduction();
    temp_result = mem_barrier_overhead;
    shmem_long_max_to_all(&temp_result, &mem_barrier_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_barrier_overhead = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_barrier_overhead;
    shmem_long_min_to_all(&temp_result, &mem_barrier_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_barrier_overhead = temp_result;
    
    reset_sync_for_reduction();
    temp_result = mem_barrier_overhead;
    shmem_long_sum_to_all(&temp_result, &mem_barrier_overhead, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_barrier_overhead = temp_result;
    
    /* shmem_init duration - max, min, sum */
    reset_sync_for_reduction();
    temp_result = init_duration_us;
    shmem_long_max_to_all(&temp_result, &init_duration_us, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    max_init_duration = temp_result;
    
    reset_sync_for_reduction();
    temp_result = init_duration_us;
    shmem_long_min_to_all(&temp_result, &init_duration_us, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    min_init_duration = temp_result;
    
    reset_sync_for_reduction();
    temp_result = init_duration_us;
    shmem_long_sum_to_all(&temp_result, &init_duration_us, 1, 0, 0, n_pes, pWrk_reduce, pSync_reduce);
    sum_init_duration = temp_result;
    
    /* Print aggregated results from PE 0 */
    if (my_pe == 0) {
        printf("\n");
        printf("===============================================\n");
        printf("SHMEM Memory Usage Benchmark\n");
        printf("Running on %d PEs\n", n_pes);
        printf("===============================================\n");
        printf("\n");
        printf("Memory Statistics (across all PEs):\n");
        printf("  %-30s %-12s %-12s %-12s\n", "Measurement", "Max", "Min", "Avg");
        printf("  -------------------------------------------------------------\n");
        
        print_memory_stats("Memory before shmem_init", max_before, min_before, sum_before, n_pes);
        print_memory_stats("Memory after shmem_init", max_after_init, min_after_init, sum_after_init, n_pes);
        print_memory_stats("shmem_init overhead", max_init_overhead, min_init_overhead, sum_init_overhead, n_pes);
        print_memory_stats("Memory after barrier", max_after_barrier, min_after_barrier, sum_after_barrier, n_pes);
        print_memory_stats("Barrier overhead", max_barrier_overhead, min_barrier_overhead, sum_barrier_overhead, n_pes);
        
        printf("\n");
        printf("Timing Statistics (across all PEs):\n");
        printf("  %-30s %-12s %-12s %-12s\n", "Measurement", "Max", "Min", "Avg");
        printf("  -------------------------------------------------------------\n");
        
        print_timing_stats("shmem_init duration", (double)max_init_duration, (double)min_init_duration, (double)sum_init_duration, n_pes);
        
        printf("\n");
        printf("===============================================\n");
        printf("\n");
    }
    
    shmem_finalize();
    return 0;
}
