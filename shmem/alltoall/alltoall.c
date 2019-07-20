#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <shmem.h>
#include <shmemx.h>

#define MAX_ITER    100
#define SKIP        10

/*
 * Blatant copy from OSU...
 */
double getMicrosecondTimeStamp()
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

#define TIME()    getMicrosecondTimeStamp()

/* arbitrary data larger than 64 bits */
struct data {
    char data;
};
typedef struct data data_t;

/*
 * Linear all to all implementation
 */
double myalltoall(void * dest, const void * source, size_t nelems, size_t selems, 
                int PE_start, int logPE_stride, int PE_size, long * pSync)
{
    int stride = 1 << logPE_stride;
    int i = PE_start;
    int mystarting_index = nelems * ((shmem_my_pe() - PE_start) / stride);
    double start = TIME(), end = 0;

    for (; i < PE_size; i += stride) {
        shmem_putmem((void *)((char *) dest + selems * mystarting_index), source, nelems * selems, i);
    }
    shmem_quiet();
    end = TIME();
    shmem_sync(PE_start, logPE_stride, PE_size, pSync);
    return end - start;
}

int main(void) 
{
    int nr_elems;
    int my_pe, n_pes;
    long * pSync;
    data_t * data;
    data_t * alldata;
    int i = 0;

    shmem_init();
    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

#if WITH_HINTS
    pSync = (long *) shmemx_malloc_with_hint(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE, SHMEM_HINT_NEAR_NIC_MEM);
#else
    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
#endif
    for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    for (i = 0; i <= 23; i++) {
        nr_elems = (1 << i);
        double f_start = 0, f_end = 0, i_time = 0;
        double latency = 0, bandwidth = 0;
        double size = 0;
        int j = 0, k = 0;

#if WITH_HINTS
        data = (data_t *) shmem_malloc(sizeof(data_t) * nr_elems);
        alldata = (data_t *) shmemx_malloc_with_hint(sizeof(data_t) * nr_elems * n_pes, SHMEM_HINT_NEAR_NIC_MEM);
#else
        data = (data_t *) shmem_malloc(sizeof(data_t) * nr_elems);
        alldata = (data_t *) shmem_malloc(sizeof(data_t) * nr_elems * n_pes);
#endif
        for (j = 0; j < nr_elems; j++) {
            data[j].data = (long) my_pe * i + j;
        }               

        for (k = 0; k < MAX_ITER + SKIP; k++) {
            double iter = 0;
            if (k == SKIP) {
                f_start = TIME();
            }
            iter = myalltoall(alldata, data, nr_elems, sizeof(data_t), 0, 0, n_pes, pSync);
            if (k >= SKIP) {
                i_time += iter;
            }
        }
        f_end = TIME();
        latency = ((f_end - f_start)) / MAX_ITER; // in us
        size = (1.0 * n_pes * MAX_ITER * nr_elems); // MB
        bandwidth = size / (i_time / 1e6);
//        bandwidth = (n_pes * (MAX_ITER * (nr_elems * sizeof(data_t)))) * (1 / (f_end - f_start) / 1000000); // MBs

        if (shmem_my_pe() == 0) {
        #ifdef ALLTOALL_DEBUG
            printf("completed iteration %d\n", i);
            for (j = 0; j < n_pes; j++) {
                int k = 0;
                printf("%d: ", j);
                for (; k < nr_elems; k++) {
                    printf("%d ", alldata[j*nr_elems + k].data);
                }
                printf("\n");
            }
        #endif

            if (nr_elems < 1024) {
                printf("** Time with size %lu:\n", sizeof(data_t) * nr_elems);
            } else if (nr_elems < (1024 * 1024)) {
                printf("** Time with size %lu kB:\n", nr_elems / 1024);
            } else {
                printf("** Time with size %lu MB:\n", nr_elems / (1024 * 1024));
            }
            //printf("\tsize: %g, i_time: %g, total: %g\n", size, i_time, f_end - f_start);
            printf("\tAvg Latency: %g us\n", latency);
            printf("\tBandwidth: %g MB/s\n", bandwidth / (1024 * 1024));
        }

        shmem_free(data);
        shmem_free(alldata);
    }
    shmem_finalize();
    return 0;
}

