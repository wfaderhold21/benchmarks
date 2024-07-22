/*
 *  This benchmark measures bandwidth and latency for a2a calls in openshmem. 
 *
 *  Meant to be used with OSHMEM
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <shmem.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>

#include <ucc/api/ucc.h>

#define NR_ITER     110
#define SKIP        10

int verify(const void * src, const int64_t * dest, int64_t *src_count, ucc_aint_t *src_disp, int64_t *dst_count, ucc_aint_t *dst_disp, size_t count, int rank, int npes)
{
    int64_t * t_dest = (int64_t *)shmem_malloc(count * npes * sizeof(int64_t));
    int *mpi_src_count = (int *)malloc(npes * sizeof(int));
    int *mpi_src_disp = (int *)malloc(npes * sizeof(int));
    int *mpi_dst_count = (int *)malloc(npes * sizeof(int));
    int *mpi_dst_disp = (int *)malloc(npes * sizeof(int));
   
    shmem_barrier_all(); 
    for (int i = 0; i < npes; i++) {
        mpi_src_count[i] = src_count[i];
        mpi_dst_count[i] = dst_count[i];
        mpi_src_disp[i] = src_disp[i];
        mpi_dst_disp[i] = dst_disp[i];
    }
    MPI_Alltoallv(src, mpi_src_count, mpi_src_disp, MPI_LONG, t_dest, mpi_dst_count, mpi_dst_disp, MPI_LONG, MPI_COMM_WORLD);
    shmem_barrier_all();
    for (int i = 0; i < npes; i++) {
        if (dest[i] != t_dest[i]) {
            printf("[%d] error: does not validate on index i: %d (%ld != %ld)\n", rank, i, dest[i], t_dest[i]);
            return -1;
        }
    }
    return 0;
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)(uintptr_t)request;
#if 1
    /* FIXME: MPI_Test in oob_allgather_test results in no completion? leave as blocking for now */
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    *req = UCC_OK;
#endif
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
#if 0
    MPI_Request request = (MPI_Request)(uintptr_t)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);

    return completed ? UROM_OK : UROM_INPROGRESS;
#else
    return UCC_OK;
#endif
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

int main(int argc, char ** argv)
{
    int me;// = shmem_me();
    int npes;// = shmem_n_pes();
    int count = 32768 * 2;
    //int count = 262144;    
    long * pSync;
    long * pSync2;
    long * pSync3;
    double * pWrk;
    static long val = 9999;
    static double min_latency, max_latency;
    static double total_time = 0.0;
    static double start, end, total = 0.0;
    static double src_buff, dest_buff;
    int size = 1;
    int num = 1;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context;
    ucc_mem_map_t *maps = NULL;
    ucc_team_h ucc_team;
    ucc_team_params_t team_params;
    ucc_status_t status;
    ucc_lib_h ucc_lib;
    ucc_count_t *src_count, *dst_count;
    ucc_aint_t *src_disp, *dst_disp;

    if (argc > 1) {
        size = atoi(argv[1]) / 8;
        count = size;
        if (argc > 2) {
            num = atoi(argv[2]);
        }
    }
    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    src_count = malloc(sizeof(int64_t) * npes);
    dst_count = malloc(sizeof(int64_t) * npes);
    src_disp = malloc(sizeof(int64_t) * npes);
    dst_disp = malloc(sizeof(int64_t) * npes);

    int64_t* dest = (int64_t*) shmem_malloc(npes * count * npes * sizeof(int64_t));
    int64_t* source = (int64_t*) shmem_malloc(npes * count * npes * sizeof(int64_t));

    int64_t disp = 0;
    int64_t in_disp = 0;

    pSync = (long *) shmem_malloc(sizeof(long) * (5));
    pSync2 = (long *) shmem_malloc(sizeof(long) * (5));//SHMEM_ALLTOALL_SYNC_SIZE);
    pSync3 = (long *) shmem_malloc(sizeof(long) * SHMEM_REDUCE_SYNC_SIZE);
    pWrk = (double *) shmem_malloc(sizeof(double) * SHMEM_REDUCE_MIN_WRKDATA_SIZE);

    for (int i = 0; i < 5; i++) {
        pSync[i] = 0; //SHMEM_SYNC_VALUE;
        pSync2[i] = 0;// SHMEM_SYNC_VALUE;
        pSync3[i] = SHMEM_SYNC_VALUE;
    }

    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t) * 6);
    if (maps == NULL) {
        printf("OOM\n");
        return -1;
    }

    maps[0].address = dest;
    maps[0].len = npes * count * npes * sizeof(int64_t);
    maps[1].address = source;
    maps[1].len = npes * count * npes * sizeof(int64_t);
    maps[2].address = pSync;
    maps[2].len = 5 * sizeof(long);
    maps[3].address = pSync2;
    maps[3].len = 5 * sizeof(long);
    maps[4].address = pSync3;
    maps[4].len = SHMEM_REDUCE_SYNC_SIZE * sizeof(long);
    maps[5].address = pWrk;
    maps[5].len = SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep = me;
    ctx_params.mem_params.segments = maps;
    ctx_params.mem_params.n_segments = 6;

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    ucc_lib_config_h lib_config;

    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config)) {
        printf("lib config error\n");
        return -1;
    }

    if (UCC_OK != ucc_init(&lib_params, lib_config, &ucc_lib)) {
        printf("init error\n");
        return -1;
    }

    if (UCC_OK != ucc_context_config_read(ucc_lib, NULL, &ctx_config)) {
        printf("error on ctx\n");
        return -1;
    }

    if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context)) {
        printf("error on ctx create\n");
        return -1;
    }

    ucc_context_config_release(ctx_config);

    team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test = oob_allgather_test;
    team_params.oob.req_free = oob_allgather_free;
    team_params.oob.coll_info = MPI_COMM_WORLD;
    team_params.oob.n_oob_eps = npes;
    team_params.oob.oob_ep = me;
    team_params.ep = me;
    team_params.flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER;

    if (UCC_OK != ucc_team_create_post(&ucc_context, 1, &team_params, &ucc_team)) {
        printf("team create post failed\n");
        return -1; 
    }   

    while (UCC_INPROGRESS == (status = ucc_team_create_test(ucc_team))) {}
    if (UCC_OK != status) {
        printf("team create failed\n");
        return -1; 
    }
    shmem_barrier_all();
    if (me == 0) {
        printf("%-10s%-10s%15s%13s%13s%13s%13s%13s\n", "Size", 
                                              "PE size",
                                              "Bandwidth MB/s", 
                                              "Agg MB/s",
                                              "Max BW",
                                              "Avg Latency", 
                                              "Min Latency", 
                                              "Max Latency");
    }

    for (int k = size; k <= count - 1; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        double max_agg = 0;
        static double total_bw = 0, min = 0;
        min = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;
        disp = in_disp = 0;
        for (int i = 0; i < npes; i++) {
            src_count[i] = k;
            dst_count[i] = k;
            src_disp[i] = disp;
            dst_disp[i] = in_disp;
            disp += i * sizeof(int64_t) * k;
            in_disp += i * sizeof(int64_t) * k;
        }
        
        /* alltoall */
        for (int i = 0; i < NR_ITER + SKIP; i++) {
            long * a_psync = (i % 2) ? pSync : pSync2;
            double b_start, b_end;
            ucc_coll_args_t coll_args = {
                .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags     = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT 
                           | UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
                .coll_type = UCC_COLL_TYPE_ALLTOALLV,
                .src.info_v =
                    {
                        .buffer   = (void *)source,
                        .counts   = src_count,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_HOST,
                        .displacements = src_disp,
                    },
                .dst.info_v =
                    {
                        .buffer   = (void *)dest,
                        .counts    = dst_count,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_HOST,
                        .displacements = dst_disp,
                    },
                .global_work_buffer = a_psync,
            };
            ucc_coll_req_h req = NULL;
            status = ucc_collective_init(&coll_args, &req, ucc_team);
            if (status != UCC_OK) {
                printf("coll init failed\n");
                return -1;
            }
            shmem_barrier_all();
            start = MPI_Wtime();
            for (int z = 0; z < num; z++) {
                status = ucc_collective_post(req);
                if (status != UCC_OK) {
                    printf("FAILED TO POST\n");
                    abort();
                }
                while (UCC_OK != (status = ucc_collective_test(req))) {
                    if (0 > status) {
                        printf("collective failed\n");
                        abort();
                    }
                    ucc_context_progress(ucc_context);
                }
                ucc_collective_finalize(req);
            }
#ifdef WITH_BASIC
                shmem_barrier_all();
#endif
            end = MPI_Wtime();
            shmem_barrier_all();

            #ifdef WITH_VERIFY
            int ret = verify(source, dest, src_count, src_disp, dst_count, dst_disp, k, me, npes);
            if (ret != 0) {
                return ret;
            }
            #endif
            if (i > SKIP) {
                double time = (end - start);
                total += time;// - (b_end - b_start);
                if (time < min) {
                    min = time;
                } 
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            shmem_barrier_all();
        }

        shmem_double_min_to_all(&min_latency, &min, 1, 0, 0, npes, pWrk, pSync3);
        shmem_barrier_all();
        shmem_double_sum_to_all(&total_time, &total, 1, 0, 0, npes, pWrk, pSync3);
        total_time = total;
        total_bw = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (NR_ITER - SKIP)) / (total_time);
        src_buff = bandwidth;
        shmem_barrier_all();
        shmem_double_sum_to_all(&dest_buff, &src_buff, 1, 0, 0, npes, pWrk, pSync3); 
        agg_bandwidth = dest_buff;
        dest_buff = 0;
        shmem_barrier_all();
        if (me == 0) {
            printf("%-10ld", k * sizeof(uint64_t));
            printf("%-10ld", k * sizeof(uint64_t) * npes);
            printf("%15.2f", bandwidth / (1024 * 1024));
            printf("%13.2f", agg_bandwidth / (1024 * 1024));
            printf("%13.2f", total_bw);
            printf("%13.2f", (total_time * 1e6) / ((NR_ITER - SKIP)));
            printf("%13.2f", min_latency * 1e6);
            printf("%13.2f\n", max_latency * 1e6);
        }
    }

    shmem_barrier_all();
    shmem_quiet();
    shmem_finalize();
    return 0;
}
