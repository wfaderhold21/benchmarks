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

#define NR_ITER     10000
#define SKIP        1000

int main(void)
{
    shmem_init();
    int me = shmem_my_pe();
    int npes = shmem_n_pes();
    int count = 524288;
    long * pSync;
    long * pSync2;
    double * pWrk;
    int64_t* dest = (int64_t*) shmem_malloc(count * npes * sizeof(int64_t));
    int64_t* source = (int64_t*) shmem_malloc(count * npes * sizeof(int64_t));
    double min_latency, max_latency;
    static double src_buff, dest_buff;
    double start, end, total = 0.0;

    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    pSync2 = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    pWrk = (double *) shmem_malloc(sizeof(double) * SHMEM_REDUCE_MIN_WRKDATA_SIZE);

    for (int i = 0; i < npes; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
        pSync2[i] = SHMEM_SYNC_VALUE;
    }

    /* assign source values */
    for (int pe = 0; pe < npes; pe++) {
        for (int i = 0; i < count; i++) {
            source[(pe * count) + i] = ((me + 1) * 10) + i;
            dest[(pe * count) + i] = 9999;
        }
    }

    if (me == 0) {
        printf("%-10s%15s%13s%13s%13s%13s\n", "Size", 
                                              "Bandwidth MB/s", 
                                              "Agg BW MB/s", 
                                              "Avg Latency", 
                                              "Min Latency", 
                                              "Max Latency");
    }

    for (int k = 1; k <= 524288; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
       /* wait for all PEs to initialize source/dest */
        shmem_barrier_all();

        min_latency = (double) INT_MAX;
        max_latency = (double) INT_MIN;

        /* alltoall */
        for (int i = 0; i < NR_ITER; i++) {
            long * pSync3 = (i & 2) ? pSync : pSync2;
            start = MPI_Wtime();
            shmem_alltoall64(dest, source, k, 0, 0, npes, pSync3);
            end = MPI_Wtime();

            if (i > SKIP) {
                double time = end - start;
                total += end - start;
                if (time < min_latency) {
                    min_latency = time;
                } 
                if (time > max_latency) {
                    max_latency = time;
                }
            }
        }
        bandwidth = (npes / 1e6 * (NR_ITER - SKIP) * k * sizeof(uint64_t)) / (total);
        src_buff = bandwidth;
        shmem_double_sum_to_all(&dest_buff, &src_buff, 1, 0, 0, npes, pWrk, pSync); 
        agg_bandwidth = dest_buff;

        if (me == 0) {
            printf("%-10ld", k * sizeof(uint64_t));
            printf("%15.2f", bandwidth);
            printf("%13.2f", agg_bandwidth);
            printf("%13.2f", (total * 1e6) / ((NR_ITER - SKIP)));
            printf("%13.2f", min_latency * 1e6);
            printf("%13.2f\n", max_latency * 1e6);
        }
    }

    shmem_free(dest);
    shmem_free(source);
    shmem_finalize();
    return 0;
}
