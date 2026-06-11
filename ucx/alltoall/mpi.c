#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ucp/api/ucp.h>
#include <mpi.h>

#include "comm-mpi.h"
#include "errors.h"
#include "common.h"

MPI_Datatype mpi_worker_exchange_dt;
MPI_Datatype mpi_buffer_exchange_dt; 

int mpi_worker_exchange(void *** param_worker_addrs)
{
    void ** worker_addresses;
    size_t worker_len;
    void * worker_address;
    int error;
    int i;
    int ret = 0;
    size_t *worker_lens = NULL;
    int *counts = NULL, *displs = NULL;
    char *all_workers = NULL;

    /* allocate */
    worker_addresses = (void **) malloc(sizeof(void *) * size);
    if (NULL == worker_addresses) {
        return ERR_NO_MEMORY;
    }


    error = ucp_worker_get_address(ucp_worker,
                                   (ucp_address_t **) &worker_address,
                                   &worker_len);
    if(error < 0) {
        free(worker_addresses);
        return -1;
    }
    
    worker_lens = (size_t *)malloc(sizeof(size_t) * size);
    counts = (int *)malloc(sizeof(int) * size);
    displs = (int *)malloc(sizeof(int) * size);
    if (NULL == worker_lens || NULL == counts || NULL == displs) {
        ret = ERR_NO_MEMORY;
        goto fail_pack;
    }

    error = MPI_Allgather(&worker_len,
                          sizeof(worker_len),
                          MPI_BYTE,
                          worker_lens,
                          sizeof(worker_len),
                          MPI_BYTE,
                          MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        ret = -1;
        goto fail_pack;
    }

    int total_worker_len = 0;
    for (i = 0; i < size; i++) {
        counts[i] = (int)worker_lens[i];
        displs[i] = total_worker_len;
        total_worker_len += counts[i];
    }
    all_workers = (char *)malloc(total_worker_len);
    if (NULL == all_workers) {
        ret = ERR_NO_MEMORY;
        goto fail_pack;
    }

    error = MPI_Allgatherv(worker_address, (int)worker_len, MPI_BYTE,
                           all_workers, counts, displs, MPI_BYTE,
                           MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        ret = -1;
        goto fail_pack;
    }

    /* set up */
    for (i = 0; i < size; i++) {
        worker_addresses[i] = malloc(worker_lens[i]);
        if (NULL == worker_addresses[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_setup;
        }

        memcpy(worker_addresses[i], all_workers + displs[i], worker_lens[i]);
    }

    ucp_worker_release_address(ucp_worker, (ucp_address_t *)worker_address);
    free(all_workers);
    free(displs);
    free(counts);
    free(worker_lens);
    *param_worker_addrs = worker_addresses;
    
    return ret;

fail_setup:
    for (--i; i >= 0; i--) {
        free(worker_addresses[i]);
    }
fail_pack:
    if (worker_address) {
        ucp_worker_release_address(ucp_worker, (ucp_address_t *)worker_address);
    }
    free(all_workers);
    free(displs);
    free(counts);
    free(worker_lens);
    free(worker_addresses);
    free(endpoints);

    return ret;
}

int mpi_buffer_exchange(void * buffer,
                        void *** pack_param,
                        uint64_t * remotes,
                        void * register_buffer)
{
    int error = 0;
    void ** pack = NULL;
    ucp_mem_h * mem = (ucp_mem_h *)register_buffer;
    size_t pack_size; 
    int ret = 0, i;
    ucs_status_t status;
    void *local_pack = NULL;
    size_t *pack_sizes = NULL;
    uint64_t *all_remotes = NULL;
    int *counts = NULL, *displs = NULL;
    char *all_packs = NULL;

    pack = (void **) calloc(size, sizeof(void *));
    if (NULL == pack) {
        ret = ERR_NO_MEMORY;
        goto fail_mpi;
    }

    status = ucp_rkey_pack(ucp_context, *mem, &local_pack, &pack_size);
    if (status != UCS_OK) {
        ret = status;
        goto fail_mpi;
    }

    remotes[my_pe] = (uint64_t)buffer;

    pack_sizes = (size_t *)malloc(sizeof(size_t) * size);
    all_remotes = (uint64_t *)malloc(sizeof(uint64_t) * size);
    counts = (int *)malloc(sizeof(int) * size);
    displs = (int *)malloc(sizeof(int) * size);
    if (NULL == pack_sizes || NULL == all_remotes || NULL == counts || NULL == displs) {
        ret = ERR_NO_MEMORY;
        goto fail_mpi;
    }

    MPI_Allgather(&pack_size, sizeof(pack_size), MPI_BYTE,
                  pack_sizes, sizeof(pack_size), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&remotes[my_pe], 1, MPI_UINT64_T,
                  all_remotes, 1, MPI_UINT64_T, MPI_COMM_WORLD);

    int total_pack_size = 0;
    for (i = 0; i < size; i++) {
        counts[i] = (int)pack_sizes[i];
        displs[i] = total_pack_size;
        total_pack_size += counts[i];
    }

    all_packs = (char *)malloc(total_pack_size);
    if (NULL == all_packs) {
        ret = ERR_NO_MEMORY;
        goto fail_mpi;
    }

    MPI_Allgatherv(local_pack, (int)pack_size, MPI_BYTE,
                   all_packs, counts, displs, MPI_BYTE, MPI_COMM_WORLD);

    for (i = 0; i < size; i++) {
        remotes[i] = all_remotes[i];
        pack[i] = malloc(pack_sizes[i]);
        if (NULL == pack[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_purge_arrays;
        }
        memcpy(pack[i], all_packs + displs[i], pack_sizes[i]);
    }
    
    ucp_rkey_buffer_release(local_pack);
    free(all_packs);
    free(displs);
    free(counts);
    free(all_remotes);
    free(pack_sizes);
    *pack_param = pack; 

    return ret;

fail_purge_arrays:
    for (--i; i >= 0; i--) {
        free(pack[i]);
    }
fail_mpi:
    if (local_pack != NULL) {
        ucp_rkey_buffer_release(local_pack);
    }
    free(all_packs);
    free(displs);
    free(counts);
    free(all_remotes);
    free(pack_sizes);
    if (NULL != pack) {
        free(pack);
    }
    return ret;
}

void create_mpi_datatype(void)
{
}

int init_mpi(void)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
printf("size: %d\n", size);
    create_mpi_datatype(); 
    return 0;
}

int finalize_mpi(void)
{
    MPI_Finalize();
}
