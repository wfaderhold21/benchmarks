#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/time.h>
#include <time.h>
#include <shmem.h>
#include <shmemx.h>

#include "common.h"

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



static inline struct timespec mydifftime(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if((end.tv_nsec-start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

struct params {
    float ** a;
    float ** b;
    float * c;
    float * d;
    float * c_a;
    float * c_b;
    int up, down, j;
    int stop;
    int num_pes;
    int mype;
};
typedef struct params params_t;

void foo(float ** a, float ** b, float * c, float * d, int start, int stop, int nr_rows)
{
    int i = 0, j = 0;

    for (i = start; i < stop; i++) {
        for (j = 1; j < M - 1; j++) {
            if (i == 0 && start == 0) {
                b[i][j] = 0.2 * 
                    (c[j] + a[i][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1]);
            } else if (i == nr_rows - 1 && stop == nr_rows) {
                b[i][j] = 0.2 * 
                    (a[i - 1][j] + a[i][j] + d[j] + a[i][j + 1] + a[i][j - 1]);
            } else {
                b[i][j] = 0.2 *
                    (a[i - 1][j] + a[i][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1]);
            }
        }
    }
}

int main(int argc, char ** argv) {
    float ** a, ** b, * c, *d;
    float * c_a, * c_b;
    int i = 0;
    int j = 0, k = 0;
    int me, npes, up, down;
    int start, stop;
    double time1,time2,result;
    params_t param;
    int nr_rows = 0;

    shmem_init();

    me = shmem_my_pe();
    npes = shmem_n_pes();
   
    nr_rows = M / npes;
    a = (float **) shmemx_malloc_with_hint(sizeof(float *) * (nr_rows), SHMEM_HINT_LOW_LAT_MEM);
    b = (float **) shmemx_malloc_with_hint(sizeof(float *) * (nr_rows), SHMEM_HINT_LOW_LAT_MEM);
    c = (float *) shmemx_malloc_with_hint(sizeof(float) * M, SHMEM_HINT_NEAR_NIC_MEM);
    d = (float *) shmemx_malloc_with_hint(sizeof(float) * M, SHMEM_HINT_NEAR_NIC_MEM);

    for (j = 0; j < (nr_rows); j++) {
        a[j] = (float *) shmemx_malloc_with_hint(sizeof(float) * M, SHMEM_HINT_LOW_LAT_MEM);
        b[j] = (float *) shmemx_malloc_with_hint(sizeof(float) * M, SHMEM_HINT_LOW_LAT_MEM);
        memset(b[j], 0, sizeof(float) * M);
        memset(a[j], 0, sizeof(float) * M);
    }

    // initialize my elements
    for (j = 0; j < nr_rows; j++) {
        a[j][0] = 1;
        a[j][M - 1] = 1;
    }

    up = (me == 0) ? -1 : me - 1;
    down = (me == (npes - 1)) ? -1 : me + 1;
    
    start = (me == 0) ? 1 : 0;
    stop = (me != npes - 1 && npes > 1) ? nr_rows : nr_rows - 1;

    shmem_barrier_all();
    time1 = TIME();
    for (i = 0; i < 1024; i++) {
        int l = 0;
        
        if (me != 0) {
            shmem_float_get(c, a[nr_rows - 1], M, up);
        } 
        if (me != (npes -1)) {
            shmem_float_get(d, a[0], M, down);
        }

        // compute
        foo(a, b, c, d, start, stop, nr_rows);
        
        // copy over
        for (l = start; l < stop; l++) {
            for (k = 1; k < M - 1; k++) {
                a[l][k] = b[l][k];
            }
        }
        shmem_barrier_all();
    }
    time2 = TIME();

#ifdef DEBUG
    for (i = 0; i < npes; i++) {
        if (me == i) {
            printf("[debug output of b]\n");
            for (j = 0; j < nr_rows; j++) {
                for (k = 0; k < M; k++) {
                    printf("%5.5g ", b[j][k]);
                }
                printf("\n");
            }
            printf("\n");
            fflush(stdout);
        }
        shmem_barrier_all();
    }
#endif
    result = time2 - time1;
    if (me == 0) {
        printf("timing: %g sec\n", result / 1000000);
    }


    shmem_free(a);
    shmem_free(b);
    shmem_free(c);
    shmem_free(d);

    shmem_finalize();

    return 0;
}
