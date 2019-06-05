#include <stdio.h>
#include <stdlib.h>

#include <shmem.h>

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
    long * data;
    long * alldata;
    int i = 0;

    shmem_init();
    my_pe = shmem_my_pe();
    n_pes = shmem_n_pes();

    pSync = (long *) shmem_malloc(sizeof(long) * SHMEM_ALLTOALL_SYNC_SIZE);

    for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    for (i = 1; i < 4; i++) {
        nr_elems = i;
        int j = 0;

        data = (long *) shmem_malloc(sizeof(long) * nr_elems);
        alldata = (long *) shmem_malloc(sizeof(long) * nr_elems * n_pes);

        for (j = 0; j < nr_elems; j++) {
            data[j] = (long) my_pe * i + j;
        }               

        myalltoall(alldata, data, nr_elems, sizeof(long), 0, 0, n_pes, pSync);
        if (shmem_my_pe() == 0) {
            printf("completed iteration %d\n", i);
            for (j = 0; j < n_pes; j++) {
                int k = 0;
                printf("%d: ", j);
                for (k; k < nr_elems; k++) {
                    printf("%lu ", alldata[j*nr_elems + k]);
                }
                printf("\n");
            }
        }

        shmem_free(data);
        shmem_free(alldata);
    }
    shmem_finalize();
    return 0;
}

