#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

#include <mpi.h>
#include <ucc/api/ucc.h>

#define NR_ITER     110
#define SKIP        10

ucc_context_h ucc_context;
ucc_lib_h ucc_lib;
ucc_team_h ucc_team;
long * pSync;

static ucc_status_t oob_allgather(void * sbuf, void * rbuf, size_t msglen,
                                  void * coll_info, void ** req)
{
    MPI_Comm comm = (MPI_Comm) coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                    &request);
    *req = (void *) request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}
int setup_ucc(void * buffer, size_t size, int me, int npes)
{
    ucc_mem_map_t * maps = NULL;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_lib_config_h lib_config;
    ucc_lib_params_t lib_params;
    ucc_thread_mode_t tm_requested; 
    ucc_ep_map_t map;
    ucc_context_attr_t attr;

    // ucc_init
    tm_requested = UCC_THREAD_SINGLE;
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = tm_requested;

    ucc_lib_config_read("UCC", NULL, &lib_config);
    ucc_init(&lib_params, lib_config, &ucc_lib);
    ucc_lib_config_release(lib_config);

    // context 
    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t));
    if (NULL == maps) {
        return -1;
    }

    maps->address = buffer;
    maps->len = size;

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep = me;
    ctx_params.mem_params.segments = maps;
    ctx_params.mem_params.n_segments = 1;

    ucc_context_config_read(ucc_lib, NULL, &ctx_config);
    ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context);

    // team 
    memset(buffer, 0, size);

    map.type = UCC_EP_MAP_ARRAY;
    map.ep_num = npes;
    map.array.elem_size = 4;

    ucc_team_params_t team_params = {
        .mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE | 
                UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS,
        .oob = {
            .allgather = oob_allgather,
            .req_test = oob_allgather_test,
            .req_free = oob_allgather_free,
            .coll_info = MPI_COMM_WORLD,
            .n_oob_eps = npes,
            .oob_ep = me,
        },
        .ep = me,
        .ep_map = map,
        .flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER,
    };

    ucc_team_create_post(&ucc_context, 1, &team_params, &ucc_team);
    while (UCC_INPROGRESS == ucc_team_create_test(ucc_team)) {
        ucc_context_progress(ucc_context);
    }
    return 0;
}

void ucc_alltoall(void * src, void * dst, size_t count, int npes, void *psync)
{
    ucc_coll_req_h req;
    ucc_coll_args_t coll = {
        .mask = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
        .coll_type = UCC_COLL_TYPE_ALLTOALL,
        .src.info = {
            .buffer = (void *)src,
            .count = count * npes,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .dst.info = {
            .buffer = dst,
            .count = count * npes,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .flags = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
        .global_work_buffer = psync,
    };

    ucc_collective_init(&coll, &req, ucc_team);
    ucc_collective_post(req);
    while (ucc_collective_test(req) == UCC_INPROGRESS) {
        ucc_context_progress(ucc_context);
    }
    ucc_collective_finalize(req);
}

void ucc_alltoallv(void * src, void * dst, size_t* counts, size_t *rcounts, int64_t *sdispl, int64_t *rdispl, int npes, void *psync)
{
    ucc_coll_req_h req;
    ucc_coll_args_t coll = {
        .mask = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
        .coll_type = UCC_COLL_TYPE_ALLTOALLV,
        .src.info_v = {
            .buffer = (void *)src,
            .counts = counts,
            .displacements = (ucc_aint_t *)sdispl,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .dst.info_v = {
            .buffer = dst,
            .counts = rcounts,
            .displacements =(ucc_aint_t *) rdispl,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .flags = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS | UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT | UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                 UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER,
        .global_work_buffer = psync,
    };

    ucc_collective_init(&coll, &req, ucc_team);
    ucc_collective_post(req);
    while (ucc_collective_test(req) == UCC_INPROGRESS) {
        ucc_context_progress(ucc_context);
    }
    ucc_collective_finalize(req);
}

int main(int argc, char ** argv)
{
    int me;
    int npes;
    int count = 32768 * 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    void * buffer = malloc(8 * ((npes / 2) * count * npes * sizeof(int64_t)) + 8 * sizeof(int64_t));
    int64_t* dest = buffer; //(int64_t*) shmem_malloc(count * npes * sizeof(int64_t));
    int64_t* validation = malloc(count * npes * sizeof(int64_t));
    int64_t* source = buffer + (4 * (npes / 2) * count * npes * sizeof(int64_t));
    int64_t* displ = malloc(sizeof(int64_t) * npes);
    int64_t* rdispl = malloc(sizeof(int64_t) * npes);
    size_t * counts = malloc(sizeof(size_t) * npes);
    size_t * rcounts = malloc(sizeof(size_t) * npes);
    int *mpi_displ = malloc(sizeof(int) * npes);
    int *mpi_rdispl = malloc(sizeof(int) * npes);
    int *mpi_scount = malloc(sizeof(int) * npes);
    int *mpi_rcount = malloc(sizeof(int) * npes);
    static double min_latency, max_latency;
    static double src_buff, dest_buff;
    static double total_time = 0.0;
    static double start, end, total = 0.0;
    int size = 1;
    size_t sncounts = 0, rncounts = 0;
    int displacements = 0;

    if (argc > 1) {
        size = atoi(argv[1]) / 8;
        count = size;
    }

    setup_ucc(buffer, (8 * (npes / 2) * (count * npes * sizeof(int64_t)) + sizeof(int64_t) * 8), me, npes);

    for (int i = 0; i < npes; i++) {
        counts[i] = ((i % 2) == 0) ? 1 : 2;
    }

    MPI_Alltoall(counts, 1, MPI_LONG, rcounts, 1, MPI_LONG, MPI_COMM_WORLD);

    /* assign source values */
    for (int pe = 0; pe < npes; pe++) {
        for (int i = 0; i < count; i++) {
            source[(pe * count) + i] = ((me + 1) * 10) + i;
            dest[(pe * count) + i] = 9999;
        }
        displ[pe] = sncounts;
        rdispl[pe] = rncounts;
        mpi_displ[pe] = sncounts;
        mpi_rdispl[pe] = rncounts;
        mpi_scount[pe] = counts[pe];
        mpi_rcount[pe] = rcounts[pe];
        sncounts += counts[pe];
        rncounts += rcounts[pe];
    }

    MPI_Alltoallv(source, mpi_scount, mpi_displ, MPI_LONG, validation, mpi_rcount, mpi_rdispl, MPI_LONG, MPI_COMM_WORLD);
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

    for (int k = size; k <= count; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        double max_agg = 0;
        static double total_bw = 0, min = 0;
        min = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;
        
        /* alltoall */
        for (int i = 0; i < NR_ITER; i++) {
            double b_start, b_end;
            size_t v_dis = 0;
            start = MPI_Wtime();
#ifdef WITH_UCC 
            ucc_alltoallv(source, dest, counts, rcounts, displ, rdispl, npes, buffer + (7 * (count * npes * sizeof(int64_t))));
#else
            MPI_Alltoallv(source, mpi_scount, mpi_displ, MPI_LONG, dest, mpi_rcount, mpi_rdispl, MPI_LONG, MPI_COMM_WORLD);
#endif

#ifdef VALIDATE
            for (int i = 0; i < npes; i++) {
                for (int j = 0; j < rcounts[i]; j++) {
                    if (validation[v_dis + j] != dest[v_dis + j]) {
                        printf("invalid on pe %d\n", i);
                    }
                }
            }
#endif
            end = MPI_Wtime();

            if (i > SKIP) {
                double time = end - start;
                total += time;// - (b_end - b_start);
                if (time < min) {
                    min = time;
                } 
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Allreduce(&min, &min_latency, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&total, &total_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        total_time = total;//total_time / (npes);
        total_bw = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (NR_ITER - SKIP)) / (total_time);
        src_buff = bandwidth;
        MPI_Allreduce(&src_buff, &dest_buff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        agg_bandwidth = dest_buff;
        MPI_Barrier(MPI_COMM_WORLD);

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

    ucc_team_destroy(ucc_team);
    ucc_context_destroy(ucc_context);
    ucc_finalize(ucc_lib);
    free(buffer);
    MPI_Finalize();
    return 0;
}
