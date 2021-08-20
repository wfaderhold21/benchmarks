#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <mpi.h>
#include <pmix.h>
#include <ucp/api/ucp.h>

#include "errors.h"

// 4 kB page
#define PAGESIZE   (1<<12)
// 2 MB page
#define HUGEPAGE   (1<<21) 

struct data_exchange {
    size_t pack_size;
    uint64_t remote;
    char pack[300]; 
};

struct worker_exchange {
    size_t worker_len;
    char worker[300];
};

ucp_context_h ucp_context;
ucp_worker_h ucp_worker;
ucp_ep_h * endpoints;
ucp_rkey_h * rkeys;
ucp_mem_h register_buffer;
uint64_t * remote_addresses;

MPI_Datatype mpi_worker_exchange_dt;
MPI_Datatype mpi_buffer_exchange_dt; 

static pmix_proc_t myproc;

int my_pe;
int size;

struct exchange
{
    uint64_t remote;
    size_t pack_length;
    char pack[300];
};

double TIME()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }

    retval = ((double) tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

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
static inline int mpi_worker_exchange(void *** param_worker_addrs)
{
    struct worker_exchange * rx;
    struct worker_exchange * dx;
    void ** worker_addresses;
    size_t worker_len;
    void * worker_address;
    int error;
    int i;
    int ret = 0;

    /* allocate */
    worker_addresses = (void **) malloc(sizeof(void *)*size);
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
    
    /* pack */
    rx = (struct worker_exchange *) 
        malloc(sizeof(struct worker_exchange)*size);
    if (NULL == rx) {
        ret = ERR_NO_MEMORY;
        goto fail_pack;
    }

    dx = (struct worker_exchange *) malloc(sizeof(struct worker_exchange));
    if (NULL == dx) {
        ret = ERR_NO_MEMORY;
        free(rx);
        goto fail_pack;
    }

    dx->worker_len = worker_len;
    memcpy(&dx->worker, worker_address, worker_len);

    /* exchange */
    error = MPI_Allgather(dx,
                          1,
                          mpi_worker_exchange_dt,
                          rx,
                          1,
                          mpi_worker_exchange_dt,
                          MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        ret = -1;
        goto fail_exchange;
    }

    /* set up */
    for (i = 0; i < size; i++) {
        worker_addresses[i] = malloc(rx[i].worker_len);
        if (NULL == worker_addresses[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_setup;
        }

        memcpy(worker_addresses[i], rx[i].worker, rx[i].worker_len);        
    }

    free(dx);
    free(rx);
    *param_worker_addrs = worker_addresses;
    
    return ret;

fail_setup:
    for (--i; i >= 0; i--) {
        free(worker_addresses[i]);
    }
fail_exchange:
    free(rx);
    free(dx);
fail_pack:
    free(worker_addresses);
    free(endpoints);

    return ret;
}

static inline int mpi_buffer_exchange(void * buffer,
                                      void *** pack_param,
                                      uint64_t * remotes,
                                      void * register_buffer)
{
    int error = 0;
    void ** pack = NULL;
    struct data_exchange * dx;
    struct data_exchange * rx = NULL;
    ucp_mem_h * mem = (ucp_mem_h *)register_buffer;
    size_t pack_size; 
    int ret = 0, i;
    ucs_status_t status;

    pack = (void **) malloc(sizeof(void *)*size);
    if (NULL == pack) {
        ret = ERR_NO_MEMORY;
        goto fail_mpi;
    }

    status = ucp_rkey_pack(ucp_context, *mem, &pack[my_pe], 
                           &pack_size);
    if (status != UCS_OK) {
        ret = error;
        goto fail_mpi;
    }

    remotes[my_pe] = (uint64_t)buffer;
/*
    TODO:
1. I need to pack all of my data into a buffer, preferably a MPI_Datatype
2. I need to perform a mpi_allgather() on the data
3. I need to loop through the data and pull out the necessary parts
*/

    /* step 1: create a data type */
    rx = (struct data_exchange *)malloc(
                                    sizeof(struct data_exchange)*size);
    dx = (struct data_exchange *)malloc(sizeof(struct data_exchange));
    dx->pack_size = pack_size;
    memcpy(&dx->pack, pack[my_pe], pack_size);
    dx->remote = remotes[my_pe];

    /* step 2: perform the allgather on the data */
    MPI_Allgather(dx, 
                  1, 
                  mpi_buffer_exchange_dt, 
                  rx, 
                  1, 
                  mpi_buffer_exchange_dt, 
                  MPI_COMM_WORLD);


    /* step 3: loop over rx and pull out the necessary parts */ 
    /* obtain the network information here... */
    for (i=0;i<size;i++) {
        if (i == my_pe) {
            continue;
        }

        /*FIXME: i'm ignoring the worker length and pack size */
        remotes[i] = rx[i].remote;
        pack[i] = malloc(pack_size);
        if (NULL == pack[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_purge_arrays;
        }
        memcpy(pack[i], rx[i].pack, pack_size);
    }
    
    free(rx);
    free(dx);
    *pack_param = pack; 

    return ret;

fail_purge_arrays:
    for (--i; i >= 0; i--) {
        free(pack[i]);
    }
fail_mpi:
    if (rx != NULL) {
        free(rx);
    }
    if (NULL != pack) {
        free(pack);
    }
    return ret;
}

void create_mpi_datatype(void)
{
    int buffer_nr_items = 3;
    MPI_Aint buffer_displacements[3];
    int buffer_block_lengths[3] = {1,1,300};
    MPI_Datatype buffer_exchange_types[3] = {MPI_UINT64_T,
                                             MPI_UINT64_T,
                                             MPI_BYTE};

    /* MPI Datatype for ucp worker address exchange */
    int worker_nr_items = 2;
    MPI_Aint worker_displacements[2];
    int worker_block_lengths[2] = {1, 300};
    MPI_Datatype worker_exchange_types[2] = {MPI_UINT64_T, MPI_BYTE};

    buffer_displacements[0] = offsetof(struct data_exchange, pack_size);
    buffer_displacements[1] = offsetof(struct data_exchange, remote);
    buffer_displacements[2] = offsetof(struct data_exchange, pack);

    worker_displacements[0] = offsetof(struct worker_exchange, worker_len);
    worker_displacements[1] = offsetof(struct worker_exchange, worker);        

    /* create an exchange data type for group creation/buffer registration */
    MPI_Type_create_struct(buffer_nr_items, 
                           buffer_block_lengths, 
                           buffer_displacements, 
                           buffer_exchange_types, 
                           &mpi_buffer_exchange_dt);
    MPI_Type_commit(&mpi_buffer_exchange_dt);

    /* create an exchange data type for UCP worker information exchange */
    MPI_Type_create_struct(worker_nr_items,
                           worker_block_lengths,
                           worker_displacements,
                           worker_exchange_types,
                           &mpi_worker_exchange_dt);
    MPI_Type_commit(&mpi_worker_exchange_dt);
}

/* 
 * This will exchange networking information with all other PEs and 
 * register an allocated buffer with the local NIC. Will create endpoints 
 * if they are not already created. 
 */
static inline int reg_buffer(void * buffer, size_t length)
{
    int i = 0;
    int error = 0;
    void ** pack = NULL;
    ucs_status_t status;
    ucp_mem_map_params_t mem_map_params;

    rkeys = (ucp_rkey_h *) malloc(sizeof(ucp_rkey_h) * size);
    if (NULL == rkeys) {
        error = ERR_NO_MEMORY;
        goto fail;
    }
    
    remote_addresses = (uint64_t *) malloc(sizeof(uint64_t) * size);
    if (NULL == remote_addresses) {
        error = ERR_NO_MEMORY;
        goto fail_endpoints;
    }
    
    mem_map_params.address    = buffer;
    mem_map_params.length     = length;
    mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS
                              | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    status = ucp_mem_map(ucp_context, 
                        &mem_map_params, 
                        &register_buffer);
    if (UCS_OK != status) {
        error = -1;
        goto fail_full;
    }

#ifdef USE_MPI
    error = mpi_buffer_exchange(buffer,
                                &pack,
                                remote_addresses,
                                &register_buffer);
#else
    printf("calling pmix data\n");
    error = pmix_data_exchange((uint64_t) buffer,
                               &pack);
#endif
    if (OK != error) {
        goto fail_full;
    }

    /* register the buffers */
    for (i = 0; i < size; i++) {
        int rkey_error;

        rkey_error = ucp_ep_rkey_unpack(endpoints[i], 
                                        pack[i], 
                                        &rkeys[i]);
        if (UCS_OK != rkey_error) {
            error = -1;
            goto fail_full;
        }

        ucp_rkey_buffer_release(pack[i]); 
        pack[i] = NULL;
    }

    // NOTE: it's OK to keep pack if going to unpack on other endpoints later
    free(pack);

    return OK;

fail_full:
    free(remote_addresses);
fail_endpoints:
    free(endpoints);
fail:
    free(rkeys);

    register_buffer = NULL;
    rkeys = NULL;
    remote_addresses = NULL;

    return error;
}

/*
 * This function creates the ucp endpoints used for communication by SharP.
 * This leverages MPI to perform the data exchange
 */
static inline int create_ucp_endpoints(void)
{
    int error = 0;
    void ** worker_addresses = NULL;
    ucp_ep_params_t ep_params;
    int i;
    
    endpoints = (ucp_ep_h *) malloc(size * sizeof(ucp_ep_h));
    if (NULL == endpoints) {
        return ERR_NO_MEMORY;
    }
    
#ifdef USE_MPI
    error = mpi_worker_exchange(&worker_addresses);
#else
    error = pmix_worker_exchange(&worker_addresses);
#endif
    if (OK != error) {
        free(endpoints);
        return -1;
    }
    
    for (i = 0; i < size; i++) {
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address = (ucp_address_t *) worker_addresses[i];
        error = ucp_ep_create(ucp_worker,
                              &ep_params,
                              &endpoints[i]);
        if (UCS_OK != error) {
            free(endpoints);
            return -1;
        }
        free(worker_addresses[i]);
    }
    free(worker_addresses);
     
    return OK;
}

int comm_init()
{
    ucp_params_t ucp_params;
    ucp_config_t * config;
    ucs_status_t status;
    int error = 0;
    ucp_worker_params_t worker_params; 

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        return -1;
    }

    ucp_params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_AMO32;
    ucp_params.request_size = 0;
    ucp_params.request_init = NULL;
    ucp_params.request_cleanup = NULL;
    ucp_params.mt_workers_shared = 1;
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES
                          | UCP_PARAM_FIELD_MT_WORKERS_SHARED;

    status = ucp_init(&ucp_params, config, &ucp_context);
    if (status != UCS_OK) {
        return -1;
    }

    ucp_config_release(config);
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    status = ucp_worker_create(ucp_context, 
                               &worker_params, 
                               &ucp_worker);
    if (status != UCS_OK) {
        return -1;
    } 

    /* create our endpoints here */
    error = create_ucp_endpoints();
    if (error != OK) {
        return -1;
    } 

    return 0;
}
/******************************/

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

void barrier()
{
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#else
    barrier_all();
#endif
}

int comm_finalize()
{
    barrier();

    for (int i = 0; i < size; i++) {
        if (rkeys[i]) {
            ucp_rkey_destroy(rkeys[i]);
        }

        if (endpoints[i]) {
            ucp_ep_destroy(endpoints[i]);
        }
    }

    free(remote_addresses);
    free(endpoints);
    ucp_mem_unmap(ucp_context, register_buffer);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);

#ifdef USE_MPI
    barrier();
    MPI_Type_free(&mpi_worker_exchange_dt);
    MPI_Type_free(&mpi_buffer_exchange_dt);
    MPI_Finalize();
#else
    barrier();
    PMIx_Finalize(NULL, 0);
#endif
}   

int cmpfunc(const void * a, const void * b) 
{
    return ((*(double *)a) - (*(double *)b));
}

void put_lat(char * sdata, int iter, int skip, size_t data_size)
{
    double start, end;
    double median;
    double variance = 0.0;
    double total = 0.0;

    assert(iter > skip);

    double * times = (double *) malloc(sizeof(double) * iter);
    memset(times, 0, sizeof(double) * iter);

    for (int i = 0; i < data_size; i++) {
        sdata[i] = (char) i;
    }

    barrier();
    if (my_pe == 0) {
        int j = 0;
        for (int i = 0; i < iter; i++) {
            if (i >= skip) {
                start = TIME();
                ucp_put(endpoints[1], sdata, data_size, remote_addresses[1], rkeys[1]);
                ucp_worker_flush(ucp_worker);
                end = TIME();
                times[j++] = end - start;
                total += end - start;
            }
        }

        total = total / (iter - skip);
        qsort(times, iter - skip, sizeof(double), cmpfunc);
        median = (times[(iter - skip) / 2 - 1] + times[(iter - skip) / 2]) / 2;

        for (int i = 0; i < (iter - skip); i++) {
            times[i] = times[i] - total;
            variance += times[i] * times[i];
        }
        variance = variance / iter;
        printf("put avg lat: %0.2f us\n", total);
        printf("put median lat: %0.2f us\n", median);
        printf("put variance: %0.2f\n", variance);
    }
    barrier();
}


int main(void) 
{
    void * mybuff;
    char * shared_ptr;
    char * sdata;
    ucp_request_param_t req_param;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    create_mpi_datatype(); 

    /* initialize the runtime and communication components */
    comm_init();
    mybuff = malloc(HUGEPAGE);
    sdata = (char *)malloc(HUGEPAGE);

    /* register memory and wireup endpoints */
    reg_buffer(mybuff, HUGEPAGE);
    
    shared_ptr = (char *) mybuff;

    barrier();

    for (int i = 0; i < HUGEPAGE; i++) {
        shared_ptr[i] = (char) i;
    }

    put_lat(sdata, 100, 10, 8);

/*    
    if (my_pe == 0) {
        d = 0xdeadbeef;
        ucp_put(endpoints[1], &d, sizeof(int), remote_addresses[1], rkeys[1]);
    } else {
        d = 0xbeefdead;
        ucp_put(endpoints[0], &d, sizeof(int), remote_addresses[0], rkeys[0]);
    }
    ucp_worker_flush(ucp_worker);
*/
    barrier();

    comm_finalize();
    free(mybuff);
    return 0;
}
