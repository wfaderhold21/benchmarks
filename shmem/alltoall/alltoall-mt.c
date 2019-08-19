#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <shmem.h>
#include <shmemx.h>

#define MAX_UPDATES 1000
#define MAX_ITER    100
#define SKIP        10

#define MAX_SHIFT 22
#define MAX_SIZE ((1 << MAX_SHIFT))

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

#define data_t char

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
    long * pSync =NULL;
    double *pWork = NULL;
    char * data;
    char * alldata;
    int i = 0, j = 0;
    long hint; 

    shmem_init();
    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

    #if NEAR==1
    hint = SHMEM_HINT_NUMA_0;
    #elif FAR==1
    hint = SHMEM_HINT_NUMA_1;
    #elif LOCAL==1
    hint = SHMEM_HINT_LOCAL;
    #elif INTERLEAVE==1
    hint = SHMEM_HINT_INTERLEAVE;
    #endif

    double *src_buff = NULL, *dest_buff = NULL, agg_bw;
    src_buff = shmem_malloc(sizeof(double));
    dest_buff = shmem_malloc(sizeof(double));


	int nthreads, thread_id;
#pragma omp parallel private(nthreads, thread_id)
{
	thread_id = omp_get_thread_num();
	nthreads = omp_get_num_threads();
#ifdef NEAR
if (0 == thread_id) {
#else 
if (nthreads-1 == thread_id) {
#endif 

#if WITH_HINTS
    pSync = (long *) shmemx_malloc_with_hint(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE, hint);
    pWork = (double *) shmem_malloc(sizeof(double) * SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    data = (char *) shmemx_malloc_with_hint(sizeof(data_t) * MAX_SIZE, hint);
    alldata = (char *) shmemx_malloc_with_hint(sizeof(data_t) * MAX_SIZE * n_pes , hint);
#else
    pWork = (double *) shmem_malloc(sizeof(double) * SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    data = (data_t *) shmem_malloc(sizeof(data_t) * MAX_SIZE);
    alldata = (data_t *) shmem_malloc(sizeof(data_t) * MAX_SIZE * n_pes);
#endif
    for (j = 0; j < nr_elems; j++) {
        data[j] = (char) my_pe * i + j;
        alldata[j] = (char) rand() % 255; 
    }               

    for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
    printf("data: %p, alldata: %p\n", data, alldata);
}

#pragma omp barrier 
if (0 == thread_id) 
	shmem_barrier_all();
// local memory update operation
if (0 == thread_id) {
    for (i = 0; i <= MAX_SHIFT; i++) {
        double start = 0, end = 0;
        int j = 0, k = 0;
        double avg_lat = 0.0;
        size_t size = 1 << i;
        // write "random" data to the buffer
        for (j = 0; j < MAX_UPDATES; j++) {
            if (j == 0) {
                start = TIME();
            }
            memcpy(data, alldata, size);
        }
        end = TIME(); 
        avg_lat = (end - start) / MAX_UPDATES;
        if (my_pe == 0) {
            if (size < 1024) {
                printf("** Memory updates with size %lu:\n", size);
            } else if (size < (1024 * 1024)) {
                printf("** Memory updates with size %lu kB:\n", size / 1024);
            } else {
                printf("** Memory updates with size %lu MB:\n", size / (1024 * 1024));
            }
            printf("\tAvg Latency: %g us\n", avg_lat);
        }    
    }
}
        
#pragma omp barrier

if (0 == thread_id) 
	shmem_barrier_all();

if (0 == thread_id) {
    for (i = 0; i <= MAX_SHIFT; i++) {
        nr_elems = (1 << i);
        double f_start = 0, f_end = 0, i_time = 0;
        double latency = 0, bandwidth = 0;
        double size = 0;
        int j = 0, k = 0;

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
        size = (1.0 * n_pes * MAX_ITER * nr_elems); // bytes
        bandwidth = size / (i_time / 1e6);

        *src_buff = bandwidth;
        shmem_double_sum_to_all(dest_buff, src_buff, 1,0,0,n_pes,pWork,pSync);
        agg_bw = *dest_buff;

			
        if (shmem_my_pe() == 0) {
            if (nr_elems < 1024) {
                printf("** Time with size %lu:\n", sizeof(data_t) * nr_elems);
            } else if (nr_elems < (1024 * 1024)) {
                printf("** Time with size %lu kB:\n", nr_elems / 1024);
            } else {
                printf("** Time with size %lu MB:\n", nr_elems / (1024 * 1024));
            }
            printf("\tAvg Latency: %g us\n", latency);
            printf("\tBandwidth: %g MB/s\n", bandwidth / (1024 * 1024));
            fprintf(stdout,"\tAggreate bandwidth %g MB/s\n", agg_bw/(1024 * 1024));
        }
    }
    shmem_barrier_all();
}

#pragma omp barrier
}
    shmem_finalize();
    return 0;
}

