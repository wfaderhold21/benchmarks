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

void foo(params_t *);

static int rows_for_pe(int pe, int npes)
{
    int base = M / npes;
    int rem = M % npes;
    return base + (pe < rem);
}

static int first_row_for_pe(int pe, int npes)
{
    int base = M / npes;
    int rem = M % npes;
    return pe * base + (pe < rem ? pe : rem);
}

int main(int argc, char ** argv) {
    float ** a, ** b, * c, *d;
    float * c_a, * c_b;
    int i = 0;
    int j = 0, k = 0;
    int me, npes, up, down;
    int start, stop;
    double time1,time2,result;
    //struct timespec time1;
    //struct timespec time2;
    //struct timespec result;
    params_t param;
    int nr_rows, first_row, max_rows;

    shmem_init();

    me = shmem_my_pe();
    npes = shmem_n_pes();
   
    nr_rows = rows_for_pe(me, npes);
    first_row = first_row_for_pe(me, npes);
    max_rows = (M + npes - 1) / npes;
    printf("[%d] hello world %d of %d (allocating %d rows)\n", me, me, npes, nr_rows);

    a = (float **) shmem_malloc(sizeof(float *) * max_rows);
    b = (float **) shmem_malloc(sizeof(float *) * max_rows);
    c = (float *) shmem_malloc(sizeof(float) * M);
    d = (float *) shmem_malloc(sizeof(float) * M);
    c_a = (float *) shmemx_malloc_with_hint((size_t)(3 * sizeof(float) * M), SHMEM_HINT_DEVICE_GPU_MEM);
    c_b = (float *) shmemx_malloc_with_hint((size_t)(sizeof(float) * M), SHMEM_HINT_DEVICE_GPU_MEM);

    for (j = 0; j < max_rows; j++) {
        a[j] = (float *) shmem_malloc(sizeof(float) * M);
        b[j] = (float *) shmem_malloc(sizeof(float) * M);
        
        memset(a[j], 0, sizeof(float) * M);
        memset(b[j], 0, sizeof(float) * M);
    }

    for (j = 0; j < nr_rows; j++) {
        a[j][0] = 1;
    }
    for (j = 0; j < nr_rows; j++) {
        a[j][M-1] = 1;
    }

    up = (me == 0) ? -1 : me - 1;
    down = (me == (npes - 1)) ? -1 : me + 1;
    
    param.a = a;
    param.b = b;
    param.c = c;
    param.d = d;
    param.c_a = c_a;
    param.c_b = c_b;
    param.up = up;
    param.down = down;
    param.num_pes = npes;
    start = (first_row == 0) ? 1 : 0;
    stop = (first_row + nr_rows == M) ? nr_rows - 1 : nr_rows;
    param.mype = me;

    param.stop = stop;
    shmem_barrier_all();
    time1 = TIME();
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    for (i = 0; i < 1024; i++) {
        int l = 0;
        
        if (me != 0) {
            int up_rows = rows_for_pe(up, npes);
            shmem_float_get(c, a[up_rows - 1], M, up);
        } 
        if (me != (npes -1)) {
            shmem_float_get(d, a[0], M, down);
        }

        for (j = start; j < stop; j++) {
            param.j = j;
            
            foo(&param);
        }
        for (l = start; l < stop; l++) {
            for (k = 1; k < M - 1; k++) {
                a[l][k] = b[l][k];
            }
        }
        shmem_barrier_all();
        if (me == 0) {
            printf("iter %d complete\n", i);
        }
    }
    time2 = TIME();
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

#ifdef DEBUG
    if (me == 0) {
        printf("[debug output of b]\n");
        for (j = 0; j < nr_rows; j++) {
            for (k = 0; k < M; k++) {
                printf("%5.5g ", a[j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
    shmem_barrier_all();
    if (me == 1) {
        printf("[debug output of b]\n");
        for (j = 0; j < nr_rows; j++) {
            for (k = 0; k < M; k++) {
                printf("%5.5g ", a[j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
    shmem_barrier_all();
#endif
    result = time2 - time1;
    //result = mydifftime(time1, time2);
    if (me == 0) {
        printf("timing: %g sec\n", result / 1000000);
    }


    shmem_free(a);
    shmem_free(b);
    shmem_free(c);
    shmem_free(d);
    
//    shmem_free(c_a);
//    shmem_free(c_b);

    shmem_finalize();

    return 0;
}
