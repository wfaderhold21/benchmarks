/*
 *  Alltoall benchmark with a concurrent background allreduce.
 *
 *  Splits MPI_COMM_WORLD by parity:
 *    - Even ranks form an "alltoall" UCC team and run the normal a2a
 *      bandwidth/latency sweep across message sizes.
 *    - Odd  ranks form an "allreduce" UCC team and continuously run a
 *      128 MB allreduce until the even ranks signal completion via a
 *      MPI_Ibarrier on MPI_COMM_WORLD.
 *
 *  Both groups share the wire so the alltoall sweep experiences network
 *  contention from the background allreduce, which is the MoE backward-pass
 *  motivating case for the congestion-avoidance work.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <malloc.h>

#include <ucc/api/ucc.h>

#include "../../common/bench_output.h"
#include "../../common/transport_detect.h"

#define NR_ITER     100
#define SKIP        10

/* 128 MB worth of int64 elements */
#define AR_BYTES    (128ULL * 1024ULL * 1024ULL)
#define AR_COUNT    (AR_BYTES / sizeof(int64_t))

#define NUM_HW_COUNTERS 4

const char* hw_counter_files[NUM_HW_COUNTERS] = {
    "np_cnp_sent",
    "rp_cnp_handled",
    "rp_cnp_ignored",
    "np_ecn_marked_roce_packets"
};

const char* hw_counter_names[NUM_HW_COUNTERS] = {
    "NP CNP Sent",
    "RP CNP Handled",
    "RP CNP Ignored",
    "NP ECN Marked RoCE Packets"
};

typedef struct {
    uint64_t counters[NUM_HW_COUNTERS];
    int hw_counters_available;
} hw_counter_data_t;

int check_hw_counters_available(const char *base_path) {
    char filepath[256];
    FILE* fp;
    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        snprintf(filepath, sizeof(filepath), "%s%s", base_path, hw_counter_files[i]);
        fp = fopen(filepath, "r");
        if (fp == NULL) return 0;
        fclose(fp);
    }
    return 1;
}

int read_hw_counters(hw_counter_data_t* data, const char *base_path) {
    char filepath[256];
    FILE* fp;
    if (!data->hw_counters_available) return 0;
    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        snprintf(filepath, sizeof(filepath), "%s%s", base_path, hw_counter_files[i]);
        fp = fopen(filepath, "r");
        if (fp == NULL) { data->hw_counters_available = 0; return 0; }
        if (fscanf(fp, "%lu", &data->counters[i]) != 1) {
            fclose(fp); data->hw_counters_available = 0; return 0;
        }
        fclose(fp);
    }
    return 1;
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm, &request);
    *req = (void *)(uintptr_t)request;
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    *req = UCC_OK;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)  { return UCC_OK; }
static ucc_status_t oob_allgather_free(void *req)  { return UCC_OK; }

static size_t get_ucc_work_buffer_size(ucc_context_h ctx)
{
    ucc_context_attr_t attr;

    attr.mask = UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE;
    if (UCC_OK != ucc_context_get_attr(ctx, &attr) ||
        attr.global_work_buffer_size == 0) {
        return 5 * sizeof(long);
    }
    return attr.global_work_buffer_size;
}

static void *alloc_zeroed_aligned(size_t size)
{
    void *ptr = NULL;

    if (posix_memalign(&ptr, 4096, size) != 0) {
        return NULL;
    }
    memset(ptr, 0, size);
    return ptr;
}

/* Create a UCC lib + context + team scoped to `comm`.
 * If `mem_segments` is non-NULL, register them with the context (used by the
 * one-sided alltoall path). */
static int create_ucc_stack(MPI_Comm comm,
                            ucc_mem_map_t *mem_segments, int n_segments,
                            ucc_lib_h *out_lib,
                            ucc_context_h *out_ctx,
                            ucc_team_h *out_team)
{
    int my_rank, nranks;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nranks);

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    ucc_lib_config_h lib_config;
    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config)) return -1;
    if (UCC_OK != ucc_init(&lib_params, lib_config, out_lib)) return -1;
    ucc_lib_config_release(lib_config);

    ucc_context_params_t ctx_params = {0};
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test  = oob_allgather_test;
    ctx_params.oob.req_free  = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)comm;
    ctx_params.oob.n_oob_eps = nranks;
    ctx_params.oob.oob_ep    = my_rank;
    if (mem_segments && n_segments > 0) {
        ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
        ctx_params.mem_params.n_segments = n_segments;
        ctx_params.mem_params.segments   = mem_segments;
    }

    ucc_context_config_h ctx_config;
    if (UCC_OK != ucc_context_config_read(*out_lib, NULL, &ctx_config)) return -1;
    if (UCC_OK != ucc_context_create(*out_lib, &ctx_params, ctx_config, out_ctx)) return -1;
    ucc_context_config_release(ctx_config);

    ucc_team_params_t team_params = {0};
    team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void *)comm;
    team_params.oob.n_oob_eps = nranks;
    team_params.oob.oob_ep    = my_rank;
    team_params.ep    = my_rank;
    team_params.flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER;

    if (UCC_OK != ucc_team_create_post(out_ctx, 1, &team_params, out_team)) return -1;
    ucc_status_t s;
    while (UCC_INPROGRESS == (s = ucc_team_create_test(*out_team))) {
        ucc_context_progress(*out_ctx);
    }
    if (UCC_OK != s) return -1;
    return 0;
}

/* Background allreduce loop. Runs until the MPI_Ibarrier `done_req`
 * (posted on MPI_COMM_WORLD) completes — which happens when the even ranks
 * reach the matching MPI_Barrier at the end of the alltoall sweep. */
static void run_background_allreduce(ucc_context_h ctx, ucc_team_h team,
                                     int64_t *sbuf, int64_t *rbuf,
                                     MPI_Request done_req)
{
    int done = 0;
    uint64_t iters = 0;
    while (!done) {
        ucc_coll_args_t args = {
            .mask      = 0,
            .coll_type = UCC_COLL_TYPE_ALLREDUCE,
            .op        = UCC_OP_SUM,
            .src.info  = { .buffer = sbuf, .count = AR_COUNT,
                           .datatype = UCC_DT_INT64,
                           .mem_type = UCC_MEMORY_TYPE_HOST },
            .dst.info  = { .buffer = rbuf, .count = AR_COUNT,
                           .datatype = UCC_DT_INT64,
                           .mem_type = UCC_MEMORY_TYPE_HOST },
        };
        ucc_coll_req_h req = NULL;
        if (UCC_OK != ucc_collective_init(&args, &req, team)) {
            fprintf(stderr, "bg allreduce init failed\n"); abort();
        }
        if (UCC_OK != ucc_collective_post(req)) {
            fprintf(stderr, "bg allreduce post failed\n"); abort();
        }
        ucc_status_t s;
        while (UCC_OK != (s = ucc_collective_test(req))) {
            if (s < 0) { fprintf(stderr, "bg allreduce failed\n"); abort(); }
            ucc_context_progress(ctx);
        }
        ucc_collective_finalize(req);
        iters++;
        MPI_Test(&done_req, &done, MPI_STATUS_IGNORE);
    }
    fprintf(stderr, "[bg] completed %lu allreduce iterations\n", iters);
}

int main(int argc, char ** argv)
{
    int world_me, world_npes;
    int count = 1048576;
    long *pSync = NULL;
    double min_latency, max_latency;
    double total_time = 0.0;
    double start, end, total = 0.0;
    double src_buff;
    int size = 1;
    int num = 1;
    size_t iter = NR_ITER;
    int ppn = 1;
    const char *hw_iface = NULL;
    const char *csv_path   = NULL;
    FILE       *csv_fp     = NULL;
    char hw_counter_base_path[256];
    hw_counter_data_t hw_counters_available_check = {0};
    char c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
    MPI_Comm_size(MPI_COMM_WORLD, &world_npes);

    while ((c = getopt(argc, argv, "i:s:d:p:c:o:")) != -1) {
        switch (c) {
            case 's': size     = atoi(optarg); break;
            case 'i': iter     = atoi(optarg); break;
            case 'd': num      = atoi(optarg); break;
            case 'p': ppn      = atoi(optarg); break;
            case 'c': hw_iface = optarg;      break;
            case 'o': csv_path = optarg;       break;
            default: MPI_Finalize(); return -1;
        }
    }

    /* CSV metadata */
    bench_meta_t meta = { "ucc_a2a", "mpi_ar", world_npes, ppn, NULL, NULL };

    if (hw_iface) {
        hw_counters_available_check.hw_counters_available = check_hw_counters_available(hw_counter_base_path);
        if (world_me == 0) {
            if (hw_counters_available_check.hw_counters_available) {
                printf("Hardware counter monitoring enabled for %s\n", hw_iface);
            } else {
                printf("Hardware counter monitoring requested for %s but counters not available, disabling\n", hw_iface);
                hw_iface = NULL;
            }
        }
    }

    /* CSV output file - only rank 0 opens */
    if (!csv_path) csv_path = getenv("BENCH_CSV");
    if (csv_path && world_me == 0) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) fprintf(stderr, "cannot open CSV: %s\n", csv_path);
    }

    if (world_npes < 4) {
        if (world_me == 0) {
            fprintf(stderr, "Need at least 4 ranks (2 even + 2 odd)\n");
        }
        MPI_Finalize();
        return -1;
    }

    /* Split: even -> alltoall (color 0), odd -> allreduce (color 1) */
    int color = world_me % 2;
    MPI_Comm sub_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_me, &sub_comm);
    int sub_me, sub_npes;
    MPI_Comm_rank(sub_comm, &sub_me);
    MPI_Comm_size(sub_comm, &sub_npes);

    if (color == 1) {
        /* ---------- odd ranks: background allreduce ---------- */
        ucc_lib_h     lib;
        ucc_context_h ctx;
        ucc_team_h    team;

        int64_t *sbuf, *rbuf;
        if (posix_memalign((void **)&sbuf, 4096, AR_BYTES) != 0 ||
            posix_memalign((void **)&rbuf, 4096, AR_BYTES) != 0) {
            fprintf(stderr, "OOM (bg buffers)\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(sbuf, 0, AR_BYTES);
        memset(rbuf, 0, AR_BYTES);

        if (create_ucc_stack(sub_comm, NULL, 0, &lib, &ctx, &team) != 0) {
            fprintf(stderr, "bg UCC stack creation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Post the done-signal: completes when even ranks reach the matching
         * MPI_Barrier on MPI_COMM_WORLD at the end of the sweep. */
        MPI_Request done_req;
        MPI_Ibarrier(MPI_COMM_WORLD, &done_req);

        run_background_allreduce(ctx, team, sbuf, rbuf, done_req);

        ucc_status_t destroy_status;
        while (UCC_INPROGRESS == (destroy_status = ucc_team_destroy(team))) {
            ucc_context_progress(ctx);
        }
        if (UCC_OK != destroy_status) {
            fprintf(stderr, "bg team destroy failed: %d\n", destroy_status);
        }
        ucc_context_destroy(ctx);
        ucc_finalize(lib);
        free(sbuf); free(rbuf);
        MPI_Comm_free(&sub_comm);
        MPI_Finalize();
        return 0;
    }

    /* ---------- even ranks: alltoall benchmark ---------- */
    int npes = sub_npes;
    int me   = sub_me;

    int64_t *source;
    if (posix_memalign((void **)&source, 4096, (size_t)npes * count * sizeof(int64_t)) != 0) {
        fprintf(stderr, "Failed to allocate source\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int64_t *dest = source;

    ucc_mem_map_t maps[1];
    maps[0].address = source;
    maps[0].len     = (size_t)npes * count * sizeof(int64_t);

    ucc_lib_h     lib;
    ucc_context_h ucc_context;
    ucc_team_h    ucc_team;
    if (create_ucc_stack(sub_comm, maps, 1, &lib, &ucc_context, &ucc_team) != 0) {
        fprintf(stderr, "a2a UCC stack creation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t work_buffer_size = get_ucc_work_buffer_size(ucc_context);
    pSync = (long *)alloc_zeroed_aligned(work_buffer_size);
    if (NULL == pSync) {
        fprintf(stderr, "Failed to allocate UCC work buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(sub_comm);

    /* Transport detection after UCC init for reliable results */
    transport_info_t ti_ar = transport_detect();

    /* CSV metadata - set transport fields after UCC init */
    if (csv_fp && me == 0) {
        meta.ucc_tls = ti_ar.ucc_tls;
        meta.ucx_tls = ti_ar.ucx_tls;
        bench_csv_header(csv_fp);
    }

    if (me == 0) {
        printf("# alltoall over %d even ranks, background 128 MB allreduce over %d odd ranks\n",
               npes, world_npes - npes);
        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            printf("%-10s%-12s%15s%15s%15s%14s%14s%14s%14s%14s%15s%15s%15s%20s\n",
                   "Size (B)", "Total (B)",
                   "BW (MB/s)", "Agg BW (MB/s)", "Max BW (MB/s)",
                   "Avg Lat (us)", "Min Lat (us)", "Max Lat (us)", "Var (us^2)",
                   "CNP Sent", "CNP Handled", "CNP Ignored", "ECN Marked");
        } else {
            printf("%-10s%-12s%15s%15s%15s%14s%14s%14s%14s\n",
                   "Size (B)", "Total (B)",
                   "BW (MB/s)", "Agg BW (MB/s)", "Max BW (MB/s)",
                   "Avg Lat (us)", "Min Lat (us)", "Max Lat (us)", "Var (us^2)");
        }
    }

    for (int k = 1; k <= count; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        double total_bw = 0, min = 0;
        double welford_mean = 0.0, welford_M2 = 0.0;
        int welford_count = 0;
        min = (double)INT_MAX;
        max_latency = (double)INT_MIN;
        total = 0;

        hw_counter_data_t total_size_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            for (int i = 0; i < NUM_HW_COUNTERS; i++) total_size_counters.counters[i] = 0;
        }

        for (int i = 0; i < (int)iter + SKIP; i++) {
            long *a_psync = pSync;
            ucc_coll_args_t coll_args = {
                .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags     = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS |
                              UCC_COLL_ARGS_FLAG_IN_PLACE |
                              UCC_COLL_ARGS_FLAG_PERSISTENT,
                .coll_type = UCC_COLL_TYPE_ALLTOALL,
                .src.info  = { .buffer = (void *)source, .count = k * npes,
                               .datatype = UCC_DT_INT64,
                               .mem_type = UCC_MEMORY_TYPE_HOST },
                .dst.info  = { .buffer = (void *)dest,  .count = k * npes,
                               .datatype = UCC_DT_INT64,
                               .mem_type = UCC_MEMORY_TYPE_HOST },
                .global_work_buffer = a_psync,
            };

            hw_counter_data_t iter_start_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
            if (hw_iface && hw_counters_available_check.hw_counters_available) {
                read_hw_counters(&iter_start_counters, hw_counter_base_path);
            }

            ucc_coll_req_h req = NULL;
            ucc_status_t status = ucc_collective_init(&coll_args, &req, ucc_team);
            if (status != UCC_OK) { fprintf(stderr, "coll init failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

            MPI_Barrier(sub_comm);
            start = MPI_Wtime();
            for (int z = 0; z < num; z++) {
                status = ucc_collective_post(req);
                if (status != UCC_OK) { fprintf(stderr, "FAILED TO POST\n"); abort(); }
                while (UCC_OK != (status = ucc_collective_test(req))) {
                    if (status < 0) { fprintf(stderr, "collective failed\n"); abort(); }
                    ucc_context_progress(ucc_context);
                }
            }
            ucc_collective_finalize(req);

            MPI_Barrier(sub_comm);
            end = MPI_Wtime();

            hw_counter_data_t iter_end_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
            if (hw_iface && hw_counters_available_check.hw_counters_available && i >= SKIP) {
                read_hw_counters(&iter_end_counters, hw_counter_base_path);
                for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                    total_size_counters.counters[j] += (iter_end_counters.counters[j] - iter_start_counters.counters[j]);
                }
            }

            if (i >= SKIP) {
                double time = (end - start);
                total += time;
                welford_count++;
                double w_delta = time - welford_mean;
                welford_mean += w_delta / welford_count;
                welford_M2 += w_delta * (time - welford_mean);
                if (time < min) min = time;
                if (time > max_latency) max_latency = time;
            }
            MPI_Barrier(sub_comm);
        }

        hw_counter_data_t global_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            for (int j = 0; j < NUM_HW_COUNTERS; j++) global_counters.counters[j] = 0;
            for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                uint64_t local_count = total_size_counters.counters[j];
                uint64_t global_count = 0;
                MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, sub_comm);
                global_counters.counters[j] = global_count;
            }
        }

        double global_min, global_max, global_total;
        MPI_Allreduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, sub_comm);
        MPI_Allreduce(&max_latency, &global_max, 1, MPI_DOUBLE, MPI_MAX, sub_comm);
        MPI_Allreduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, sub_comm);

        min_latency = global_min;
        total_time = global_total;
        max_latency = global_max;
        double n = (double)npes * (double)iter;
        double avg_time = global_total / npes;
        double global_mean = global_total / n;
        double local_dev_sq = welford_M2 + (welford_mean - global_mean) *
                              (welford_mean - global_mean) * welford_count;
        double global_dev_sq;
        MPI_Allreduce(&local_dev_sq, &global_dev_sq, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
        double variance_us2 = (global_dev_sq / n) * 1e12;

        total_bw  = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (double)iter) / avg_time;
        src_buff  = bandwidth;

        MPI_Allreduce(&src_buff, &agg_bandwidth, 1, MPI_DOUBLE, MPI_SUM, sub_comm);
        MPI_Barrier(sub_comm);
        if (me == 0) {
            printf("%-10ld%-12ld%15.2f%15.2f%15.2f%14.2f%14.2f%14.2f%14.2f",
                   k * sizeof(uint64_t),
                   k * sizeof(uint64_t) * npes,
                   (bandwidth / (1024 * 1024)) * ppn,
                   agg_bandwidth / (1024 * 1024),
                   total_bw * ppn,
                   (avg_time * 1e6) / (double)iter,
                   min_latency * 1e6,
                   max_latency * 1e6,
                   variance_us2);
            if (hw_iface && hw_counters_available_check.hw_counters_available) {
                printf("%15lu%15lu%15lu%20lu",
                       global_counters.counters[0],
                       global_counters.counters[1],
                       global_counters.counters[2],
                       global_counters.counters[3]);
            }
            printf("\n");

            /* CSV row */
            if (csv_fp) {
                bench_csv_row(csv_fp, &meta,
                              k * sizeof(uint64_t), (int)iter,
                              avg_time * 1e6 / iter, min_latency * 1e6,
                              max_latency * 1e6, sqrt(variance_us2),
                              bandwidth / (1024.0 * 1024.0));
            }
        }
    }

    MPI_Barrier(sub_comm);

    /* Signal odd ranks to stop the background allreduce. */
    MPI_Barrier(MPI_COMM_WORLD);

    ucc_status_t destroy_status;
    while (UCC_INPROGRESS == (destroy_status = ucc_team_destroy(ucc_team))) {
        ucc_context_progress(ucc_context);
    }
    if (UCC_OK != destroy_status) {
        fprintf(stderr, "a2a team destroy failed: %d\n", destroy_status);
    }
    ucc_context_destroy(ucc_context);
    ucc_finalize(lib);

    if (csv_fp && me == 0) { fflush(csv_fp); fclose(csv_fp); }

    free(source);
    free(pSync);
    MPI_Comm_free(&sub_comm);
    MPI_Finalize();
    return 0;
}
