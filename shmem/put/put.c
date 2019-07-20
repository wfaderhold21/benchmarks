#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <shmem.h>
#include <shmemx.h>

#define MAX_ITER    100000
#define WARMUP      1000

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

int main(void) 
{
    int nr_elems;
    int my_pe, n_pes;
    char * data;
    char * rbuf;
    int i = 0;

    shmem_init();
    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

#if WITH_HINTS
        data = (char *) shmemx_malloc_with_hint((1<<23)*100, SHMEM_HINT_NEAR_NIC_MEM);
        rbuf = (char *) shmemx_malloc_with_hint((1<<23)*100, SHMEM_HINT_NEAR_NIC_MEM);
#else
        data = (char *) shmem_malloc((1<<23)*100);
        rbuf = (char *) shmem_malloc((1<<23)*100);
#endif


    for (i = 0; i <= 23; i++) {
        nr_elems = (1 << i);
        double f_start = 0, f_end = 0, m_time = 0;
        double latency = 0, bandwidth = 0;
        double size = 0;
        int j = 0, k = 0;
        int iterations = (1<<23) / nr_elems + 100;

        for (j = 0; j < nr_elems; j++) {
            data[j] = 'a';
            rbuf[j] = 'b';
        }               

        shmem_barrier_all();
        for (k = 0; k < WARMUP; i++) {
            shmem_putmem(&rbuf[k], &data[k], 1, 0);
        }
        shmem_barrier_all();

        if (shmem_my_pe() == 3) {
            f_start = TIME();
            for (k = 0; k < iterations; k++) {
                shmem_putmem(&rbuf[k * nr_elems], &data[k * nr_elems], nr_elems, 0);
            }
            shmem_quiet();
            f_end = TIME();
        }
        shmem_barrier_all();

        latency = (iterations * 1e6) / (f_end - f_start); // in us
        size = (1.0 * iterations * nr_elems); // MB
        bandwidth = size / ((f_end - f_start) / 1e6);

        if (shmem_my_pe() == 3) {
            if (nr_elems < 1024) {
                printf("** Time with size %lu:\n", sizeof(char) * nr_elems);
            } else if (nr_elems < (1024 * 1024)) {
                printf("** Time with size %lu kB:\n", nr_elems / 1024);
            } else {
                printf("** Time with size %lu MB:\n", nr_elems / (1024 * 1024));
            }
            printf("\tMessage rate: %g m/s\n", latency);
            printf("\tBandwidth: %g MB/s\n", bandwidth / (1024 * 1024));
        }
    }
    shmem_free(data);
    shmem_free(rbuf);
    shmem_finalize();
    return 0;
}

