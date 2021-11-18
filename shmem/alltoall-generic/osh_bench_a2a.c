/*
 *  This benchmark measures bandwidth and latency for a2a calls in openshmem. 
 *
 *  Meant to be used with OSHMEM
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <shmem.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>

#define NR_ITER     1000
#define SKIP        100

int main(int argc, char ** argv)
{
    shmem_init();
    int me = shmem_my_pe();
    int npes = shmem_n_pes();
    int count = 32768;
    long * pSync;
    long * pSync2;
    long * pSync3;
    double * pWrk;
    static long val = 9999;
    int64_t* dest = (int64_t*) shmem_malloc(count * npes * sizeof(int64_t));
    int64_t* source = (int64_t*) shmem_malloc(count * npes * sizeof(int64_t));
    static double min_latency, max_latency;
    static double src_buff, dest_buff;
    static double total_time = 0.0;
    static double start, end, total = 0.0;
    int ppn = 1;

    if (argc > 1) {
        ppn = atoi(argv[1]);
    }

    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    pSync2 = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    pSync3 = (long *) shmem_malloc(sizeof(long) * SHMEM_REDUCE_SYNC_SIZE);
    pWrk = (double *) shmem_malloc(sizeof(double) * SHMEM_REDUCE_MIN_WRKDATA_SIZE);

    for (int i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
        pSync2[i] = SHMEM_SYNC_VALUE;
        pSync3[i] = SHMEM_SYNC_VALUE;
    }

    /* assign source values */
    for (int pe = 0; pe < npes; pe++) {
        for (int i = 0; i < count; i++) {
            source[(pe * count) + i] = ((me + 1) * 10) + i;
            dest[(pe * count) + i] = 9999;
        }
    }

    if (me == 0) {
        printf("%-10s%15s%13s%13s%13s%13s%13s\n", "Size", 
                                              "Bandwidth MB/s", 
                                              "Agg MB/s",
                                              "Total BW",
                                              "Avg Latency", 
                                              "Min Latency", 
                                              "Max Latency");
    }

    for (int k = 1; k <= count; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        static double total_bw = 0, min = 0;
        min_latency = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;
        
        /* alltoall */
        for (int i = 0; i < NR_ITER; i++) {
            long * a_psync = (i % 2) ? pSync : pSync2;
            double b_start, b_end;
            start = MPI_Wtime();
            shmem_alltoall64(dest, source, k, 0, 0, npes, a_psync);
            end = MPI_Wtime();

            if (i > SKIP) {
                double time = end - start;
                total += time;// - (b_end - b_start);
                if (time < min_latency) {
                    min = time;
                } 
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            shmem_barrier_all();
        }

        shmem_double_min_to_all(&min_latency, &min, 1, 0, 0, npes, pWrk, pSync2);
        shmem_barrier_all();
        shmem_double_sum_to_all(&total_time, &total, 1, 0, 0, npes, pWrk, pSync);
        total_time = total_time / npes;
        total_bw = (npes * (k * sizeof(uint64_t))) / min_latency;
        bandwidth = (npes * (k * sizeof(uint64_t)) * (NR_ITER - SKIP)) / (total_time);
        src_buff = bandwidth;
        shmem_barrier_all();
        shmem_double_sum_to_all(&dest_buff, &src_buff, 1, 0, 0, npes, pWrk, pSync3); 
        agg_bandwidth = dest_buff;
        if (me == 0) {
            printf("%-10ld", k * sizeof(uint64_t));
            printf("%15.2f", bandwidth / (1024 * 1024));
            printf("%13.2f", agg_bandwidth / (1024 * 1024));
            printf("%13.2f", total_bw / (1024 * 1024));
            printf("%13.2f", (total_time * 1e6) / ((NR_ITER - SKIP)));
            printf("%13.2f", min_latency * 1e6);
            printf("%13.2f\n", max_latency * 1e6);
        }
    }

    shmem_free(dest);
    shmem_free(source);
    shmem_finalize();
    return 0;
}
