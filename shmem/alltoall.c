#include <stdio.h>
#include <stdlib.h>

#include <shmem.h>

#include "cycles.c"

#define MAX_ITER    100
#define SKIP        10

/* arbitrary data larger than 64 bits */
struct data {
    char data;
};
typedef struct data data_t;

/*
 * Linear all to all implementation
 */
void myalltoall(void * dest, const void * source, size_t nelems, size_t selems, 
                int PE_start, int logPE_stride, int PE_size, long * pSync)
{
    int stride = 1 << logPE_stride;
    int i = PE_start;
    int mystarting_index = nelems * ((shmem_my_pe() - PE_start) / stride);

    for (i; i < PE_size; i += stride) {
        shmem_putmem((void *)((char *) dest + selems * mystarting_index), source, nelems * selems, i);
    }
    shmem_quiet();
    shmem_sync(PE_start, logPE_stride, PE_size, pSync);
}

int main(void) 
{
    int nr_elems;
    int my_pe, n_pes;
    long * pSync;
    data_t * data;
    data_t * alldata;
    int i = 0;
    cycles_t * tstamp;
    struct report_options report = {};

    shmem_init();
    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

    tstamp = (cycles_t *) malloc(sizeof(cycles_t) * 2 * MAX_ITER);

    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);
    for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    for (i = 0; i <= 20; i++) {
        nr_elems = (1 << i);
        int j = 0, k = 0, z = 0;

        data = (data_t *) shmem_malloc(sizeof(data_t) * nr_elems);
        alldata = (data_t *) shmem_malloc(sizeof(data_t) * nr_elems * n_pes);

        for (j = 0; j < nr_elems; j++) {
        //    data[j].nr = my_pe * nr_elems + j;
            data[j].data = (long) my_pe * i + j;
        }               

        for (k = 0; k < MAX_ITER; k++) {
            if (k >= SKIP) {
                tstamp[z++] = get_cycles();
            }
            myalltoall(alldata, data, nr_elems, sizeof(data_t), 0, 0, n_pes, pSync);
            if (k >= SKIP) {
                tstamp[z++] = get_cycles();
            }   
        }

        if (shmem_my_pe() == 0) {
            printf("completed iteration %d\n", i);
/*            for (j = 0; j < n_pes; j++) {
                int k = 0;
                printf("%d: ", j);
                for (k; k < nr_elems; k++) {
                    printf("%lu ", alldata[j*nr_elems + k].data);
                }
                printf("\n");
            }*/

            printf("Time with size: %lu:\n", sizeof(data_t) * nr_elems);
            print_report(&report, MAX_ITER - SKIP, tstamp);


        }

        shmem_free(data);
        shmem_free(alldata);
    }
    shmem_finalize();
    return 0;
}

