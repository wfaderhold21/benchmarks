#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shmem.h>
#include <shmemx.h>
#include <sys/time.h>

#define DEFAULT_NELEMS 1024
#define DEFAULT_ITERATIONS 100

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int my_pe, n_pes;
    int64_t *odd_src, *odd_dst;
    int64_t *even_src, *even_dst;
    size_t nelems;
    int iterations;
    shmem_req_h odd_req, even_req;
    double start_time, end_time;
    int i;

    // Initialize OpenSHMEM
    shmem_init();

    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

    // Parse command line arguments
    if (argc > 1) {
        nelems = atoi(argv[1]);
    } else {
        nelems = DEFAULT_NELEMS;
    }

    if (argc > 2) {
        iterations = atoi(argv[2]);
    } else {
        iterations = DEFAULT_ITERATIONS;
    }

    // Allocate buffers
    odd_src = (int64_t *)shmem_malloc(nelems * sizeof(int64_t));
    odd_dst = (int64_t *)shmem_malloc(nelems * sizeof(int64_t));
    even_src = (int64_t *)shmem_malloc(nelems * sizeof(int64_t));
    even_dst = (int64_t *)shmem_malloc(nelems * sizeof(int64_t));

    if (!odd_src || !odd_dst || !even_src || !even_dst) {
        fprintf(stderr, "Failed to allocate memory\n");
        shmem_finalize();
        return 1;
    }

    // Initialize source buffers with some data
    for (i = 0; i < nelems; i++) {
        odd_src[i] = my_pe * 1000 + i;
        even_src[i] = my_pe * 2000 + i;
    }
    odd_req = SHMEM_REQ_INVALID;
    even_req = SHMEM_REQ_INVALID;

    // Synchronize all PEs before starting the benchmark
    shmem_sync_all();

    // Start timing
    start_time = get_time();

    // Main benchmark loop
    for (i = 0; i < iterations; i++) {
        if (i % 2) {
            if (odd_req != SHMEM_REQ_INVALID) {
                shmem_req_wait(&odd_req);
            }
            shmemx_alltoall_nb(SHMEM_TEAM_WORLD, odd_dst, odd_src, nelems, &odd_req);
        } else {
            if (even_req != SHMEM_REQ_INVALID) {
                shmem_req_wait(&even_req);
            }
            shmemx_alltoall_nb(SHMEM_TEAM_WORLD, even_dst, even_src, nelems, &even_req);
        }
#ifndef WITHOUT_SYNC
        if (i > 0 && (i % 2 == 0)) {
            shmem_sync_all();
        }
#endif
    }
    shmem_req_wait(&odd_req);
    shmem_req_wait(&even_req);

    // End timing
    end_time = get_time();
    shmem_barrier_all();

    // Print results
    if (my_pe == 0) {
        printf("Double Buffering Benchmark Results:\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Elements per PE: %zu\n", nelems);
        printf("Number of iterations: %d\n", iterations);
        printf("Total time: %f seconds\n", end_time - start_time);
        printf("Average time per iteration: %f microseconds\n", 
               ((end_time - start_time) / iterations) * 1000000.0);
    }

    // Cleanup
    shmem_free(odd_src);
    shmem_free(odd_dst);
    shmem_free(even_src);
    shmem_free(even_dst);

    shmem_finalize();
    return 0;
} 