#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <shmem.h>
#include <shmemx.h>
#include <sys/time.h>

#define DEFAULT_NELEMS 1024
#define DEFAULT_ITERATIONS 10000

static double get_time() {
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

void
compute_on_host_()
{
    int x=0;
    int i = 0, j = 0;
    for (i = 0; i < 25; i++) {
        for (j = 0; j < 25; j++) {
            x = x + i*j;
        }
    }
}

static inline void
do_compute_cpu_(double target_usec)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_usec) {
        t1 = get_time();
        compute_on_host_();
        t2 = get_time();
        time_elapsed += (t2-t1);
    }
}

int main(int argc, char **argv) {
    int my_pe, n_pes;
    int64_t *odd_src, *odd_dst;
    int64_t *even_src, *even_dst;
    size_t nelems;
    int iterations;
    shmem_req_h odd_req, even_req;
    double start_time, end_time;
    double comm_time1, comm_time2;
    double total_comm_time = 0.0;
    double compute_time;
    double latency, latency_avg;
    int i;
    long pSync_a2a_odd[_SHMEM_ALLTOALL_SYNC_SIZE];
    long pSync_a2a_even[_SHMEM_ALLTOALL_SYNC_SIZE];

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
    for (int i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync_a2a_even[i] = SHMEM_SYNC_VALUE;
        pSync_a2a_odd[i] = SHMEM_SYNC_VALUE;
    }

    // Synchronize all PEs before starting the benchmark
    shmem_sync_all();

    for (int i = 0; i < iterations; i++) {
        shmem_alltoall64(even_dst, even_src, nelems, 0, 0, n_pes, pSync_a2a_even);
        shmem_barrier_all();
    }
    shmem_barrier_all();
    start_time = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        shmem_alltoall64(even_dst, even_src, nelems, 0, 0, n_pes, pSync_a2a_even);
        shmem_barrier_all();
    }
    end_time = MPI_Wtime();
    latency = end_time - start_time;
    MPI_Allreduce(&latency, &latency_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    latency_avg = latency_avg / n_pes;
    compute_time = 1500; //(0.9 * (latency_avg * 1e6));
    
    // Start timing
    start_time = MPI_Wtime();

    #ifdef WITH_BLOCKING
        comm_time1 = MPI_Wtime();
        shmem_alltoall64(even_dst, even_src, nelems, 0, 0, n_pes, pSync_a2a_even);
        comm_time2 = MPI_Wtime();
        total_comm_time += comm_time2 - comm_time1;
    #else
        comm_time1 = MPI_Wtime();
        shmemx_alltoallmem_nb(SHMEM_TEAM_WORLD, even_dst, even_src, nelems * sizeof(int64_t), &even_req);
        comm_time2 = MPI_Wtime();
        total_comm_time += comm_time2 - comm_time1;

    #endif
    do_compute_cpu_(compute_time);

    // Main benchmark loop
    for (i = 1; i < iterations; i++) {
    #ifdef WITH_BLOCKING
        comm_time1 = MPI_Wtime();
        shmem_barrier_all();
        comm_time2 = MPI_Wtime();
        total_comm_time += comm_time2 - comm_time1;

        comm_time1 = MPI_Wtime();
        if (i & 1) {
            shmem_alltoall64(odd_dst, odd_src, nelems, 0, 0, n_pes, pSync_a2a_odd);
        } else {
            shmem_alltoall64(even_dst, even_src, nelems, 0, 0, n_pes, pSync_a2a_even);
        }
        comm_time2 = MPI_Wtime();
        total_comm_time += comm_time2 - comm_time1;
    #else
        comm_time1 = MPI_Wtime();
        if (i & 1) { //odd_req != SHMEM_REQ_INVALID) {
            shmem_req_wait(&even_req);
        } else { //even_req != SHMEM_REQ_INVALID) {
            shmem_req_wait(&odd_req);
        }

        #ifndef WITHOUT_SYNC
        shmem_sync_all();
        #endif
        if (i & 1) {
            shmemx_alltoallmem_nb(SHMEM_TEAM_WORLD, odd_dst, odd_src, nelems * sizeof(int64_t), &odd_req);
        } else {
            shmemx_alltoallmem_nb(SHMEM_TEAM_WORLD, even_dst, even_src, nelems * sizeof(int64_t), &even_req);
        }
        comm_time2 = MPI_Wtime();
        total_comm_time += comm_time2 - comm_time1;

    #endif

    // do compute
        do_compute_cpu_(compute_time);
    }
#ifdef WITH_BLOCKING
    comm_time1 = MPI_Wtime();
    shmem_barrier_all();
    comm_time2 = MPI_Wtime();
    total_comm_time += comm_time2 - comm_time1;
#else
    comm_time1 = MPI_Wtime();
    shmem_req_wait(&odd_req);
    shmem_req_wait(&even_req);
    #ifndef WITHOUT_SYNC
        shmem_sync_all();
    #endif
    comm_time2 = MPI_Wtime();
    total_comm_time += comm_time2 - comm_time1;
#endif

    // End timing
    end_time = MPI_Wtime();
    shmem_barrier_all();

    double total_time = end_time - start_time;
    double avg_time;
    MPI_Allreduce(&total_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_time = avg_time / n_pes;

    // Print results
    if (my_pe == 0) {
        printf("Double Buffering Benchmark Results:\n");
        printf("Number of PEs: %d\n", n_pes);
        printf("Elements per PE: %zu\n", nelems);
        printf("Number of iterations: %d\n", iterations);
        printf("Compute time: %f s\n", compute_time / 1e6);
        printf("Total time: %f seconds\n", avg_time);
        printf("Total comm time: %f seconds\n", total_comm_time);
        printf("Average comm time per iter: %f us\n", 1e6 * (total_comm_time / iterations));
        printf("Average time per iteration: %f us\n", 
               ((end_time - start_time) / iterations) * 1e6);
    }

    shmem_finalize();
    return 0;
} 
