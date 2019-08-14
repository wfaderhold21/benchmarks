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
};
typedef struct params params_t;

void foo(params_t *);

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

    shmem_init();

    me = shmem_my_pe();
    npes = shmem_n_pes();
   
    printf("[%d] hello world %d of %d (allocating %d rows)\n", me, me, npes, M / npes); 

    a = (float **) shmem_malloc((sizeof(float *) * M) / npes);
    b = (float **) shmem_malloc((sizeof(float *) * M) / npes);
    c = (float *) shmem_malloc(sizeof(float) * M);
    d = (float *) shmem_malloc(sizeof(float) * M);
    c_a = (float *) shmemx_malloc_with_hint((size_t) (((sizeof(float) * M) / npes) * M), SHMEM_HINT_DEVICE_GPU_MEM);
    c_b = (float *) shmemx_malloc_with_hint((size_t) (((sizeof(float) * M) / npes) * M), SHMEM_HINT_DEVICE_GPU_MEM);

    for (j = 0; j < (M / npes); j++) {
        a[j] = (float *) shmem_malloc(sizeof(float) * M);
        b[j] = (float *) shmem_malloc(sizeof(float) * M);
        
        memset(a[j], 0, sizeof(float) * M);
        memset(b[j], 0, sizeof(float) * M);
    }

    for (j = 0; j < M / npes; j++) {
        a[j][0] = 1;
    }
    for (j = 0; j < M / npes; j++) {
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
    start = (me == 0) ? 1 : 0;
    stop = (me == 0 && npes > 1) ? M / npes : M / npes - 1;

    param.stop = stop;
    shmem_barrier_all();
    time1 = TIME();
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    for (i = 0; i < 1024; i++) {
        int l = 0;
        
        if (me != 0) {
            shmem_float_get(c, a[(M / npes) - 1], M, up);
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
    }
    time2 = TIME();
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

#ifdef DEBUG
    if (me == 0) {
        printf("[debug output of b]\n");
        for (j = 0; j < M/npes; j++) {
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
        for (j = 0; j < M/npes; j++) {
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
