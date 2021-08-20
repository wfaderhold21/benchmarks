#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <pmix.h>
#include <ucp/api/ucp.h>

#include "common.h"
#include "errors.h"

pmix_proc_t myproc;

int pmix_data_exchange(uint64_t remote, void *** pack_param) 
{
    struct data_exchange * ex;
    void ** pack = NULL;
    ucp_mem_h * mem = (ucp_mem_h *) register_buffer;
    size_t pack_size;
    pmix_value_t val;    
    int rc;
    int i = 0;
    ucs_status_t status;

    pack = (void **) malloc(sizeof(void *) * size);
    if (NULL == pack) {
        return ERR_NO_MEMORY;
    }

    status = ucp_rkey_pack(ucp_context, *mem, &pack[my_pe], 
                           &pack_size);
    if (status != UCS_OK) {
        free(pack);
        return status;
    }

    // allocate/exchange our own info
    ex = (struct data_exchange *) malloc(sizeof(struct data_exchange));
    ex->remote = remote;
    ex->pack_size = pack_size;
    memcpy(ex->pack, &pack[my_pe], pack_size);

    val.type = PMIX_BYTE_OBJECT;
    val.data.bo.size = sizeof(struct data_exchange);
    val.data.bo.bytes = (void *) ex;

    rc = PMIx_Put(PMIX_GLOBAL, "data_exchange", &val);
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Client ns %s rank %u: PMIx_Put exchange information failed: %d\n", myproc.nspace, myproc.rank, rc);
        return -1;
    }

    rc = PMIx_Commit();
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Failed to commit!\n");
        return -1;
    }

    // gather everyone's info
    for (; i < size; i++) {
        if (i != my_pe) {
            pmix_proc_t proc;
            pmix_value_t * value = &val;
            struct data_exchange * rx;

            proc = myproc;
            proc.rank = i;

            rc = PMIx_Get(&proc, "data_exchange", NULL, 0, &value);
            if (PMIX_SUCCESS != rc) {
                fprintf(stderr, "Client ns %s rank %d: PMIx_Get failed: %d\n", myproc.nspace, myproc.rank, rc);
                return -1;
            }

            rx = (struct data_exchange *) value->data.bo.bytes;
            printf("[%d] PE %d remote address: 0x%lx and buffer length %lu\n", my_pe, i, rx->remote, rx->pack_size);
            remote_addresses[i] = rx->remote;
            pack[i] = malloc(rx->pack_size);
            memcpy(pack[i], rx->pack, rx->pack_size);
        }
    }

    *pack_param = pack;

    return 0;
}

int pmix_worker_exchange(void *** param_worker_addrs) 
{
    struct worker_exchange * ex;
    void ** worker_addresses;
    size_t worker_len;
    void * worker_address; 
    pmix_value_t val;    
    int rc;
    int i = 0;

    // allocate/exchange our own info
    worker_addresses = (void **) malloc(sizeof(void *) * size);
    if (NULL == worker_addresses) {
        return ERR_NO_MEMORY;
    }

    rc = ucp_worker_get_address(ucp_worker,
                                   (ucp_address_t **) &worker_address,
                                   &worker_len);
    if (rc < 0) {
        free(worker_addresses);
        return -1;
    }

    ex = (struct worker_exchange *) malloc(sizeof(struct worker_exchange));
    ex->worker_len = worker_len;
    memcpy(&ex->worker, worker_address, worker_len);

    val.type = PMIX_BYTE_OBJECT;
    val.data.bo.size = sizeof(struct worker_exchange);
    val.data.bo.bytes = (void *) ex;

    rc = PMIx_Put(PMIX_GLOBAL, "worker_exchange", &val);
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Put exchange information failed: %d\n", myproc.nspace, myproc.rank, rc);
        return -1;
    }

    rc = PMIx_Commit();
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Failed to commit!\n");
        return -1;
    }

    // gather everyone's info
    for (; i < size; i++) {
        if (i != my_pe) {
            pmix_proc_t proc;
            pmix_value_t * value = &val;
            struct worker_exchange * rx;

            proc = myproc;
            proc.rank = i;

            rc = PMIx_Get(&proc, "worker_exchange", NULL, 0, &value);
            if (PMIX_SUCCESS != rc) {
                fprintf(stderr, "Client ns %s rank %d: PMIx_Get failed: %d\n", myproc.nspace, myproc.rank, rc);
                return -1;
            }

            rx = (struct worker_exchange *) value->data.bo.bytes;
//            printf("[%d] PE %d remote address: 0x%lx and buffer length %lu\n", my_pe, i, rx->remote, rx->length);
            worker_addresses[i] = malloc(rx->worker_len);
            memcpy(worker_addresses[i], rx->worker, rx->worker_len);
        }
    }

    free(ex);
    *param_worker_addrs = worker_addresses;

    return 0;
}

int barrier_all(void)
{
    int rc;
    pmix_proc_t proc;

    proc = myproc;
    proc.rank = PMIX_RANK_WILDCARD;

    rc = PMIx_Fence(&proc, 1, NULL, 0);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "Barrier Failed!");
        return -1;
    }
    return 0;
}

int init_pmix()
{
    int rc;
    pmix_value_t value;
    pmix_value_t * val = &value;
    pmix_proc_t proc;

    rc = PMIx_Init(&myproc, NULL, 0);
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Init failed: %d\n", myproc.nspace, myproc.rank, rc);
        return -1;
    }

    my_pe = myproc.rank;
    proc = myproc;
    proc.rank = PMIX_RANK_WILDCARD;

    rc = PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
    if (PMIX_SUCCESS != rc) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Get universe size failed: %d\n", myproc.nspace, myproc.rank, rc);
        return -1;
    }

    size = val->data.uint32;
    PMIX_VALUE_RELEASE(val);

    return 0;
}
