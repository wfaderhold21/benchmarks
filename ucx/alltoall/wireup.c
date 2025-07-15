#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include <mpi.h>
#include <ucp/api/ucp.h>
#include <ucs/debug/debug.h>

#include "comm-mpi.h"
#include "errors.h"
#include "common.h"

ucp_context_h ucp_context;
ucp_worker_h ucp_worker;
ucp_ep_h * endpoints;
ucp_rkey_h * rkeys;
ucp_mem_h register_buffer;
uint64_t * remote_addresses;

int my_pe;
int size;

double TIME()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }

    retval = ((double) tv.tv_sec) * 1e6 + tv.tv_usec;
    return retval;
}


/* 
 * This will exchange networking information with all other PEs and 
 * register an allocated buffer with the local NIC. Will create endpoints 
 * if they are not already created. 
 */
int reg_buffer(void * buffer, size_t length)
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
    error = pmix_data_exchange((uint64_t) buffer,
                               &pack);
#endif
    if (OK != error) {
        goto fail_full;
    }

    /* unpack keys into rkey array */
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
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;

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

    /* initialize communication channel for exchanges */
#ifdef USE_MPI
    init_mpi();
#else
    init_pmix();
#endif

    /* create our endpoints here */
    error = create_ucp_endpoints();
    if (error != OK) {
        return -1;
    } 

    return 0;
}
/******************************/



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
    ucp_request_param_t req_param = {0};
    ucs_status_ptr_t req;

    req = ucp_worker_flush_nbx(ucp_worker, &req_param);
    if (UCS_OK != req) {
        if (UCS_PTR_IS_ERR(req)) {
            abort();
        } else {
            while (ucp_request_check_status(req) == UCS_INPROGRESS) {
                ucp_worker_progress(ucp_worker);
            }
            ucp_request_free(req);
        }
    }

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
   // ucp_cleanup(ucp_context);

#ifdef USE_MPI
    finalize_mpi();
#else
    barrier();
    PMIx_Finalize(NULL, 0);
#endif
}   

int cmpfunc(const void * a, const void * b) 
{
    return ((*(double *)a) - (*(double *)b));
}

void alltoall_benchmark(char * sdata, int iter, int skip, size_t data_size)
{
    double start, end;
    double median;
    double variance = 0.0;
    double total = 0.0;
    double total_bandwidth = 0.0;
    ucp_request_param_t req_param = {0};
    ucs_status_ptr_t ptrs[size];
    
    assert(iter > skip);

    double * times = (double *) malloc(sizeof(double) * iter);
    memset(times, 0, sizeof(double) * iter);
//    ptrs = (ucs_status_ptr_t *)malloc(sizeof(ucs_status_ptr_t) * size);

    /* Initialize send data */
    for (int i = 0; i < data_size; i++) {
        sdata[i] = (char) (my_pe + i);
    }

    barrier();
    
    int j = 0;
    for (int i = 0; i < iter; i++) {
        start = MPI_Wtime();
        /* Linear alltoall: each PE sends to all other PEs */
        for (int pe = 0; pe < size; pe++) {
            /* Calculate offset for this PE's data segment */
            uint64_t remote_offset = (uint64_t)my_pe * data_size;
            ptrs[pe] = ucp_put_nbx(endpoints[pe],
                        sdata,
                        data_size,
                        remote_addresses[pe] + remote_offset,
                        rkeys[pe],
                        &req_param);
        }
        barrier();
         
        /* Flush all put operations to ensure completion */
        ucs_status_ptr_t flush_req = ucp_worker_flush_nbx(ucp_worker, &req_param);
        if (UCS_PTR_IS_ERR(flush_req)) {
            fprintf(stderr, "Error in ucp_worker_flush_nbx\n");
            abort();
        } else if (flush_req != UCS_OK) {
            while (UCS_INPROGRESS == ucp_request_check_status(flush_req)) {
                ucp_worker_progress(ucp_worker);
                sched_yield();
            }
            ucp_request_free(flush_req);
        }
        
        end = MPI_Wtime();
        for (int pe = 0; pe < size; pe++) {
            if (ptrs[pe]) {
                ucp_request_free(ptrs[pe]);
            }
        }
        if (i >= skip) {
            times[j++] = 1e6 * (end - start);
            total += 1e6 * (end - start);
        }
        barrier();
    }

    barrier();
//    free(ptrs);
    /* Calculate statistics */
    total = (total) / (iter - skip);
    qsort(times, iter - skip, sizeof(double), cmpfunc);
    median = (times[(iter - skip) / 2 - 1] + times[(iter - skip) / 2]) / 2;

    for (int i = 0; i < (iter - skip); i++) {
        double deviation = times[i] - total;
        variance += deviation * deviation;
    }
    variance = variance / (iter - skip);
    
    /* Calculate bandwidth */
    /* Each PE sends data_size bytes to (size-1) other PEs */
    size_t total_data_per_pe = data_size * (size - 1);
    double bandwidth_mbps = (total_data_per_pe / (total / 1e6)) / (1024 * 1024); /* MB/s */
    
    /* Print results from PE 0 */
    if (my_pe == 0) {
        printf("=== AllToAll Benchmark Results ===\n");
        printf("Message size: %ld bytes\n", data_size);
        printf("Number of PEs: %d\n", size);
        printf("Data per PE: %ld bytes\n", total_data_per_pe);
        printf("Average latency: %.2f us\n", total);
        printf("Median latency: %.2f us\n", median);
        printf("Latency variance: %.2f\n", variance);
        printf("Bandwidth per PE: %.2f MB/s\n", bandwidth_mbps);
        printf("Aggregate bandwidth: %.2f MB/s\n", bandwidth_mbps * size);
        printf("===================================\n\n");
    }
    
    free(times);
    barrier();
}

int main(void) 
{
    void * mybuff;
    char * shared_ptr;
    char * sdata;
    ucp_request_param_t req_param;
    
    //ucs_debug_init();
    
    /* initialize the runtime and communication components */
    comm_init();
    
    /* Allocate larger buffer to accommodate alltoall data from all PEs */
    size_t buffer_size = HUGEPAGE * size;
    mybuff = malloc(buffer_size);
    sdata = (char *)malloc(HUGEPAGE);

    /* register memory  */
    reg_buffer(mybuff, buffer_size);
    
    shared_ptr = (char *) mybuff;

    barrier();

    /* Initialize the shared buffer */
    for (int i = 0; i < buffer_size; i++) {
        shared_ptr[i] = (char) i;
    }

    printf("Starting AllToAll benchmark on PE %d of %d\n", my_pe, size);
    
    /* Run alltoall benchmark for different message sizes */
    for (int i = 2; i <= 1024*1024; i *= 2) {
        printf("size %d\n", i);
        alltoall_benchmark(sdata, 1000, 10, i);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    comm_finalize();
    free(sdata);
    free(mybuff);

    //ucs_debug_cleanup();
    return 0;
}
