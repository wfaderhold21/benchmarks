/*
 * ucc_bench_a2a_shrink.c
 *
 * Alltoall bandwidth benchmark with simulated rank failure and UCC team
 * shrink recovery. Runs alltoall across all ranks up to a configurable
 * message size, then simulates a failure by shrinking the team and
 * continues benchmarking with the reduced rank set.
 *
 * Usage: mpirun -np N ./ucc_bench_a2a_shrink [-i iters] [-p ppn] [-k kill] [-f fail_bytes]
 *   -i N   : iterations per message size (default 100)
 *   -p N   : processes per node for bandwidth scaling (default 1)
 *   -k N   : number of ranks to simulate failing (default 1, always last N ranks)
 *   -f N   : per-rank message size in bytes at which to trigger failure (default 8192)
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <mpi.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>

#include <ucc/api/ucc.h>

#define NR_ITER  100
#define SKIP     10

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm, &request);
    *req = (void *)(uintptr_t)request;
    /* FIXME: MPI_Test in oob_allgather_test results in no completion; leave as blocking */
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    *req = (void *)(uintptr_t)UCC_OK;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) { return UCC_OK; }
static ucc_status_t oob_allgather_free(void *req) { return UCC_OK; }

static void print_header(void)
{
    printf("%-10s%-12s%15s%15s%15s%14s%14s%14s%14s\n",
           "Size (B)", "Total (B)",
           "BW (MB/s)", "Agg BW (MB/s)", "Max BW (MB/s)",
           "Avg Lat (us)", "Min Lat (us)", "Max Lat (us)", "Var (us^2)");
}

/*
 * Run alltoall for k values [start_k, end_k] (doubling each step).
 * Reports per-message-size bandwidth/latency stats to stdout from rank 0.
 */
static int run_a2a_phase(int me, int npes, int start_k, int end_k,
                          size_t iter, int ppn,
                          int64_t *source, int64_t *dest, long *pSync,
                          ucc_team_h team, ucc_context_h ctx, MPI_Comm comm)
{
    ucc_status_t status;

    for (int k = start_k; k <= end_k; k *= 2) {
        double wf_mean = 0.0, wf_M2 = 0.0;
        int    wf_n    = 0;
        double min_lat = (double)INT_MAX;
        double max_lat = (double)INT_MIN;
        double total   = 0.0;

        for (int i = 0; i < (int)iter + SKIP; i++) {
            ucc_coll_args_t args = {
                .mask               = UCC_COLL_ARGS_FIELD_FLAGS |
                                      UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags              = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
                .coll_type          = UCC_COLL_TYPE_ALLTOALL,
                .src.info           = {
                    .buffer   = (void *)source,
                    .count    = (uint64_t)k * npes,
                    .datatype = UCC_DT_INT64,
                    .mem_type = UCC_MEMORY_TYPE_HOST,
                },
                .dst.info           = {
                    .buffer   = (void *)dest,
                    .count    = (uint64_t)k * npes,
                    .datatype = UCC_DT_INT64,
                    .mem_type = UCC_MEMORY_TYPE_HOST,
                },
                .global_work_buffer = pSync,
            };

            MPI_Barrier(comm);
            double t0 = MPI_Wtime();

            ucc_coll_req_h req = NULL;
            if (UCC_OK != ucc_collective_init(&args, &req, team)) {
                fprintf(stderr, "[rank %d] coll init failed\n", me);
                return -1;
            }
            if (UCC_OK != ucc_collective_post(req)) {
                fprintf(stderr, "[rank %d] coll post failed\n", me);
                return -1;
            }
            while (UCC_OK != (status = ucc_collective_test(req))) {
                if (status < 0) {
                    fprintf(stderr, "[rank %d] collective failed: %d\n", me, status);
                    return -1;
                }
                ucc_context_progress(ctx);
            }
            ucc_collective_finalize(req);

            double t1 = MPI_Wtime();
            MPI_Barrier(comm);

            if (i >= SKIP) {
                double t = t1 - t0;
                total   += t;
                wf_n++;
                double d = t - wf_mean;
                wf_mean += d / wf_n;
                wf_M2   += d * (t - wf_mean);
                if (t < min_lat) min_lat = t;
                if (t > max_lat) max_lat = t;
            }
            MPI_Barrier(comm);
        }

        double gmin, gmax, gtotal;
        MPI_Allreduce(&min_lat, &gmin,   1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&max_lat, &gmax,   1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&total,   &gtotal, 1, MPI_DOUBLE, MPI_SUM, comm);

        double n       = (double)npes * (double)iter;
        double gmean   = gtotal / n;
        double ldev_sq = wf_M2 + (wf_mean - gmean) * (wf_mean - gmean) * wf_n;
        double gdev_sq;
        MPI_Allreduce(&ldev_sq, &gdev_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
        double var_us2 = (gdev_sq / n) * 1e12;

        double bw = (double)(npes * k * (int)sizeof(int64_t)) * (double)iter / total;
        double agg_bw;
        MPI_Allreduce(&bw, &agg_bw, 1, MPI_DOUBLE, MPI_SUM, comm);
        double max_bw = (double)(npes * k * (int)sizeof(int64_t)) /
                        (1024.0 * 1024.0 * gmin);

        MPI_Barrier(comm);
        if (me == 0) {
            printf("%-10ld%-12ld%15.2f%15.2f%15.2f%14.2f%14.2f%14.2f%14.2f\n",
                   (long)(k * sizeof(int64_t)),
                   (long)((long)k * npes * sizeof(int64_t)),
                   bw / (1024.0 * 1024.0) * ppn,
                   agg_bw / (1024.0 * 1024.0),
                   max_bw * ppn,
                   gtotal / npes / (double)iter * 1e6,
                   gmin * 1e6,
                   gmax * 1e6,
                   var_us2);
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    int me, npes;
    int count            = 1048576; /* max k (int64_t units per rank) */
    size_t iter          = NR_ITER;
    int ppn              = 1;
    int kill_count       = 1;
    int fail_after_bytes = 8192;    /* fail after per-rank msg size of this many bytes */
    char c;
    ucc_status_t status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    while ((c = getopt(argc, argv, "i:p:k:f:")) != -1) {
        switch (c) {
            case 'i': iter             = (size_t)atoi(optarg); break;
            case 'p': ppn              = atoi(optarg);         break;
            case 'k': kill_count       = atoi(optarg);         break;
            case 'f': fail_after_bytes = atoi(optarg);         break;
            default:
                if (me == 0)
                    fprintf(stderr,
                            "Usage: %s [-i iters] [-p ppn] [-k kill_count] [-f fail_bytes]\n",
                            argv[0]);
                MPI_Finalize();
                return 1;
        }
    }

    if (kill_count < 1 || kill_count >= npes) {
        if (me == 0)
            fprintf(stderr, "kill_count must be in [1, npes-1], got %d\n", kill_count);
        MPI_Finalize();
        return 1;
    }

    /* k is int64_t-unit count per rank; msg bytes = k * sizeof(int64_t) */
    int fail_after_k = fail_after_bytes / (int)sizeof(int64_t);
    if (fail_after_k < 1) fail_after_k = 1;

    int surviving_npes = npes - kill_count;
    int is_failed      = (me >= surviving_npes); /* last kill_count ranks "fail" */

    /* Allocate data and sync buffers sized for the full team */
    int64_t *source;
    if (posix_memalign((void **)&source, 4096,
                       (size_t)npes * count * sizeof(int64_t)) != 0) {
        fprintf(stderr, "[rank %d] OOM: source buffer\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int64_t *dest = source; /* in-place alltoall */

    long *pSync;
    if (posix_memalign((void **)&pSync, 4096, 5 * sizeof(long)) != 0) {
        fprintf(stderr, "[rank %d] OOM: pSync\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(pSync, 0, 5 * sizeof(long));

    ucc_mem_map_t maps[2] = {
        { .address = source, .len = (size_t)npes * count * sizeof(int64_t) },
        { .address = pSync,  .len = 5 * sizeof(long) },
    };

    /* UCC library init */
    ucc_lib_params_t lib_params = {
        .mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    ucc_lib_config_h lib_config;
    ucc_lib_h ucc_lib;
    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config)) {
        fprintf(stderr, "[rank %d] lib config error\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (UCC_OK != ucc_init(&lib_params, lib_config, &ucc_lib)) {
        fprintf(stderr, "[rank %d] ucc_init error\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ucc_lib_config_release(lib_config);

    /* UCC context over MPI_COMM_WORLD */
    ucc_context_params_t ctx_params = {
        .mask       = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS,
        .oob        = {
            .allgather = oob_allgather,
            .req_test  = oob_allgather_test,
            .req_free  = oob_allgather_free,
            .coll_info = (void *)(uintptr_t)MPI_COMM_WORLD,
            .n_oob_eps = npes,
            .oob_ep    = me,
        },
        .mem_params = { .n_segments = 2, .segments = maps },
    };
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context;
    if (UCC_OK != ucc_context_config_read(ucc_lib, NULL, &ctx_config)) {
        fprintf(stderr, "[rank %d] ctx config error\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context)) {
        fprintf(stderr, "[rank %d] ctx create error\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ucc_context_config_release(ctx_config);

    /* UCC team over MPI_COMM_WORLD */
    ucc_team_params_t team_params = {
        .mask  = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB |
                 UCC_TEAM_PARAM_FIELD_FLAGS,
        .oob   = {
            .allgather = oob_allgather,
            .req_test  = oob_allgather_test,
            .req_free  = oob_allgather_free,
            .coll_info = (void *)(uintptr_t)MPI_COMM_WORLD,
            .n_oob_eps = npes,
            .oob_ep    = me,
        },
        .ep    = me,
        .flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER,
    };
    ucc_team_h ucc_team;
    if (UCC_OK != ucc_team_create_post(&ucc_context, 1, &team_params, &ucc_team)) {
        fprintf(stderr, "[rank %d] team create post error\n", me);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    while (UCC_INPROGRESS == (status = ucc_team_create_test(ucc_team))) {}
    if (UCC_OK != status) {
        fprintf(stderr, "[rank %d] team create failed: %d\n", me, status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ================================================================
     * PHASE 1: Pre-failure alltoall benchmark (all ranks)
     * ================================================================ */
    if (me == 0) {
        printf("=== PHASE 1: PRE-FAILURE (%d ranks) ===\n", npes);
        print_header();
    }

    if (run_a2a_phase(me, npes, 1, fail_after_k, iter, ppn,
                      source, dest, pSync, ucc_team, ucc_context,
                      MPI_COMM_WORLD) != 0)
        return -1;

    /* ================================================================
     * FAILURE + RECOVERY
     * ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) {
        printf("\n[FAILURE EVENT] Simulating failure of rank(s) %d-%d (%d rank(s))\n",
               surviving_npes, npes - 1, kill_count);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_recover_start = MPI_Wtime();

    /* Destroy old team and context while all ranks are still alive */
    double t0_td = MPI_Wtime();
    status = ucc_team_destroy(ucc_team);
    double t1_td = MPI_Wtime();
    if (UCC_OK != status)
        fprintf(stderr, "[rank %d] team destroy failed: %d\n", me, status);

    double t0_cd = MPI_Wtime();
    status = ucc_context_destroy(ucc_context);
    double t1_cd = MPI_Wtime();
    if (UCC_OK != status)
        fprintf(stderr, "[rank %d] context destroy failed: %d\n", me, status);

    /* Split communicator — failed ranks receive MPI_COMM_NULL */
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_failed ? MPI_UNDEFINED : 0, me, &new_comm);

    if (is_failed) {
        ucc_finalize(ucc_lib);
        free(source);
        free(pSync);
        MPI_Finalize();
        return 0;
    }

    int new_me, new_npes;
    MPI_Comm_rank(new_comm, &new_me);
    MPI_Comm_size(new_comm, &new_npes);

    /* Create new UCC context over the shrunken communicator */
    double t0_cc = MPI_Wtime();
    ucc_context_params_t new_ctx_params = {
        .mask       = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS,
        .oob        = {
            .allgather = oob_allgather,
            .req_test  = oob_allgather_test,
            .req_free  = oob_allgather_free,
            .coll_info = (void *)(uintptr_t)new_comm,
            .n_oob_eps = new_npes,
            .oob_ep    = new_me,
        },
        .mem_params = { .n_segments = 2, .segments = maps },
    };
    ucc_context_config_h new_ctx_config;
    ucc_context_h new_context;
    if (UCC_OK != ucc_context_config_read(ucc_lib, NULL, &new_ctx_config)) {
        fprintf(stderr, "[rank %d] new ctx config error\n", new_me);
        MPI_Abort(new_comm, 1);
    }
    if (UCC_OK != ucc_context_create(ucc_lib, &new_ctx_params, new_ctx_config,
                                     &new_context)) {
        fprintf(stderr, "[rank %d] new ctx create error\n", new_me);
        MPI_Abort(new_comm, 1);
    }
    ucc_context_config_release(new_ctx_config);
    double t1_cc = MPI_Wtime();

    /* Create new UCC team over the shrunken communicator */
    double t0_tc = MPI_Wtime();
    ucc_team_params_t new_team_params = {
        .mask  = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB |
                 UCC_TEAM_PARAM_FIELD_FLAGS,
        .oob   = {
            .allgather = oob_allgather,
            .req_test  = oob_allgather_test,
            .req_free  = oob_allgather_free,
            .coll_info = (void *)(uintptr_t)new_comm,
            .n_oob_eps = new_npes,
            .oob_ep    = new_me,
        },
        .ep    = new_me,
        .flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER,
    };
    ucc_team_h new_team;
    if (UCC_OK != ucc_team_create_post(&new_context, 1, &new_team_params, &new_team)) {
        fprintf(stderr, "[rank %d] new team create post error\n", new_me);
        MPI_Abort(new_comm, 1);
    }
    while (UCC_INPROGRESS == (status = ucc_team_create_test(new_team))) {}
    if (UCC_OK != status) {
        fprintf(stderr, "[rank %d] new team create failed: %d\n", new_me, status);
        MPI_Abort(new_comm, 1);
    }
    double t1_tc = MPI_Wtime();

    double t_recover_end = MPI_Wtime();

    /* Report recovery timing (max across surviving ranks) */
    double times[5] = {
        t_recover_end - t_recover_start,
        t1_td - t0_td,
        t1_cd - t0_cd,
        t1_cc - t0_cc,
        t1_tc - t0_tc,
    };
    double max_times[5];
    MPI_Allreduce(times, max_times, 5, MPI_DOUBLE, MPI_MAX, new_comm);

    MPI_Barrier(new_comm);
    if (new_me == 0) {
        printf("[RECOVERY COMPLETE] %d ranks -> %d ranks (killed %d)\n",
               npes, new_npes, kill_count);
        printf("  Total recovery:  %8.2f ms\n",   max_times[0] * 1e3);
        printf("  Team destroy:    %8.2f ms\n",   max_times[1] * 1e3);
        printf("  Context destroy: %8.2f ms\n",   max_times[2] * 1e3);
        printf("  Context create:  %8.2f ms\n",   max_times[3] * 1e3);
        printf("  Team create:     %8.2f ms\n\n", max_times[4] * 1e3);
    }

    /* ================================================================
     * PHASE 2: Post-recovery alltoall benchmark (reduced team)
     * ================================================================ */
    int start_k = fail_after_k * 2;

    if (new_me == 0) {
        printf("=== PHASE 2: POST-RECOVERY (%d ranks) ===\n", new_npes);
        if (start_k > count)
            printf("  (no sizes remaining; increase -f to push failure point earlier)\n");
        else
            print_header();
    }

    if (start_k <= count) {
        if (run_a2a_phase(new_me, new_npes, start_k, count, iter, ppn,
                          source, dest, pSync, new_team, new_context,
                          new_comm) != 0)
            return -1;
    }

    /* Cleanup */
    MPI_Barrier(new_comm);
    ucc_team_destroy(new_team);
    ucc_context_destroy(new_context);
    ucc_finalize(ucc_lib);
    free(source);
    free(pSync);
    MPI_Comm_free(&new_comm);
    MPI_Finalize();
    return 0;
}
