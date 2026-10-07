/*
 *  This benchmark measures bandwidth and latency for a2a calls in MPI.
 *
 *  Meant to be used with MPI
 *
 *  NIC / ICMC measurement (optional):
 *    -c <dev>      enable node-wide NIC counters for device <dev>
 *                  (e.g. mlx5_0).  Device-wide sysfs counters are read
 *                  ONCE PER NODE by the node leader (lowest rank of the
 *                  shared-memory group), then reduced across node
 *                  leaders - so values are not overcounted by ppn
 *                  (fixes the old per-rank Allreduce(SUM) bug).
 *    -C <list>     counter backends, comma separated: sysfs,iio
 *                  (default when -c is given: "sysfs,iio").
 *                  sysfs: CNP/ECN hw_counters (node sum over leaders).
 *                  iio:   uncore IIO PMU NIC read counters - all
 *                          non-payload NIC reads on the node, an
 *                          ESTIMATE of ICM-cache (ICMC) misses.
 *                          Raw rd txn/bytes are node sums; derived
 *                          per-message metrics are the per-node MAX.
 *    -Q            sample RDMA QP counts: node-wide QPs on <dev> (by
 *                  type) per message size, sampled after SKIP warm-up
 *                  (UCX creates eps/DCIs lazily; lanes differ for
 *                  eager vs rndv), plus this process's own QPs
 *                  (measured, not computed: OMPI pml ucx, OSHMEM
 *                  spml ucx and UCC tl/ucp each own UCP workers/QPs).
 *                  Also prints a per-rank-0 baseline: pre-team /
 *                  post-team / post-first-collective.
 *    -F <bytes>    NIC put fragment size estimate for the
 *                  messages-per-node computation (default 65536).
 *    -w <usec>     per-iteration CPU compute (busy-work) injected
 *                  BETWEEN posting the d collectives and draining them:
 *                    for i in iters:
 *                        for z in depth: post(coll[z])
 *                        compute (-w usec)
 *                        wait all completions
 *                  0 (default) = pure-collective mode, timing identical
 *                  to the original harness. Port of the compute mode in
 *                  ucc/pipeline-a2av/bw-pipeline.c, kept byte-compatible
 *                  (same inner loop) so the two are comparable.
 *
 *  All counter/QP reads happen OUTSIDE the timed region, so latency/BW
 *  are unaffected (validate: counters on vs off agree within variance).
 *
 *  If a backend is unavailable (e.g. no uncore IIO on the arch, no
 *  `rdma` tool), the corresponding columns print 0 / stay empty and the
 *  benchmark still runs.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>

#include <ucc/api/ucc.h>

#include "../../common/bench_output.h"
#include "../../common/transport_detect.h"
#include "../../common/nic_counters.h"
#include "../../common/iio_pmu.h"
#include "../../common/qp_count.h"

#define NR_ITER     100
#define SKIP        10
#define NOISE_WIN_S 1.0
#define DEFAULT_PUT_FRAG 65536
#define MAX_PIPELINE 64   /* upper bound on -d (true concurrent depth) */

#define NUM_HW_COUNTERS 4

const char* hw_counter_names[NUM_HW_COUNTERS] = {
    "NP CNP Sent",
    "RP CNP Handled",
    "RP CNP Ignored",
    "NP ECN Marked RoCE Packets"
};

/* Per-size, per-node stats (leader fills; non-leaders leave 0). */
typedef struct {
    uint64_t cnp[NUM_HW_COUNTERS];
    uint64_t iio_rd_txn;      /* 64 B read txns, this node */
    uint64_t iio_rd_bytes;    /* this node read bytes */
    double   excess_rd_B_per_msg;   /* this node */
    double   icm_fetch_per_msg;     /* this node (estimated) */
    uint64_t qp_node;         /* node-wide QPs on <dev> (or all) */
    char     qp_names[QP_TYPE_MAX][8];
    uint64_t qp_types[QP_TYPE_MAX];
    uint64_t qp_ntypes;
} size_stats_t;

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

    return completed ? UCC_OK : UCC_INPROGRESS;
#else
    (void)req;
    return UCC_OK;
#endif
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

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

/* Query this process's own QPs; -1 if the backend is unavailable. */
static int qp_self_count(void)
{
    qp_count_t q;
    if (qp_query(NULL, (int)getpid(), &q, NULL) == 0)
        return (int)q.total;
    return -1;
}

/* ---- compute injection (port of ucc/pipeline-a2av/bw-pipeline.c) ----
 * Same inner busy-work loop as bw-pipeline's compute_on_host_/
 * do_compute_cpu_, with a volatile sink: without it gcc -O3 deletes the
 * arithmetic entirely (bw-pipeline relies on the cross-TU loop shape to
 * survive; here we make the work real explicitly). */
static void
compute_on_host_(void)
{
    volatile int x = 0;
    int i, j;
    for (i = 0; i < 25; i++)
        for (j = 0; j < 25; j++)
            x = x + i * j;
}

static void
do_compute_cpu_(double target_usec)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_usec) {
        t1 = MPI_Wtime();
        compute_on_host_();
        t2 = MPI_Wtime();
        time_elapsed += (t2 - t1) * 1e6;   /* MPI_Wtime is seconds */
    }
}

int main(int argc, char ** argv)
{
    int me;
    int npes;
    int count = 1048576;
    long * pSync;
    double min_latency, max_latency;
    double start, end, total = 0.0;
    double src_buff;
    int size = 1;
    int num = 1;
    size_t iter = NR_ITER;
    int ppn = 1;
    const char *hw_iface = NULL;
    const char *backends = NULL;
    int qp_sampling = 0;
    int put_frag = DEFAULT_PUT_FRAG;
    double compute_usec = 0.0;      /* -w: per-iteration CPU work between post and drain */
    double compute_total = 0.0;     /* accumulated over timed iters */
    const char *csv_path   = NULL;
    FILE       *csv_fp     = NULL;
    char c;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context;
    ucc_mem_map_t *maps = NULL;
    ucc_team_h ucc_team;
    ucc_team_params_t team_params;
    ucc_status_t status;
    ucc_lib_h ucc_lib;

    /* per-node / leader state */
    MPI_Comm  node_comm = MPI_COMM_NULL;
    int       me_node = 0, nnode = 1;
    int       is_leader = 0;
    int       leader0 = 0;          /* node-0 leader (broadcast root) */
    int       n_nodes = 1;
    nic_counters_t *nc = NULL;      /* leader only */
    int       iio_active = 0;       /* any leader has the IIO backend */
    uint64_t  noise_txn = 0, noise_units = 0;   /* idle window, per leader */

    /* QP baseline breakdown (this process, for rank 0) */
    int qp_pre_team = -1, qp_post_team = -1, qp_post_coll = -1;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    while ((c = getopt(argc, argv, "i:s:d:p:c:C:QF:w:o:")) != -1) {
        switch (c) {
            case 's': size   = atoi(optarg); break;
            case 'i': iter   = atoi(optarg); break;
            case 'd': num    = atoi(optarg); break;

            case 'p': ppn    = atoi(optarg); break;
            case 'c': hw_iface = optarg;     break;
            case 'C': backends = optarg;     break;
            case 'Q': qp_sampling = 1;       break;
            case 'F': put_frag = atoi(optarg); break;
            case 'w': compute_usec = atof(optarg); break;
            case 'o': csv_path = optarg;      break;
            default: return -1;
        }
    }
    if (hw_iface && !backends) backends = "sysfs,iio";
    if (num < 1) num = 1;
    if (num > MAX_PIPELINE) num = MAX_PIPELINE;

    /* ---- node topology: shared-memory groups (one leader per node) ---- */
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, me,
                            MPI_INFO_NULL, &node_comm) != MPI_SUCCESS) {
        /* fallback: infer nodes from ppn (MPI < 4.0 runtime) */
        int color = (ppn > 0) ? me / ppn : 0;
        MPI_Comm_split(MPI_COMM_WORLD, color, me, &node_comm);
    }
    MPI_Comm_size(node_comm, &nnode);
    MPI_Comm_rank(node_comm, &me_node);
    int *node_ranks = (int *)malloc((size_t)nnode * sizeof(int));
    MPI_Gather(&me, 1, MPI_INT, node_ranks, 1, MPI_INT, 0, node_comm);
    if (me_node == 0) {
        int min_r = INT_MAX;
        for (int i = 0; i < nnode; i++)
            if (node_ranks[i] < min_r) min_r = node_ranks[i];
        is_leader = (me == min_r);
    }
    free(node_ranks);
    int *leader_flag = (int *)malloc((size_t)npes * sizeof(int));
    MPI_Allgather(&is_leader, 1, MPI_INT, leader_flag, 1, MPI_INT,
                  MPI_COMM_WORLD);
    n_nodes = 0;
    for (int i = 0; i < npes; i++)
        if (leader_flag[i]) {
            if (n_nodes == 0) leader0 = i;
            n_nodes++;
        }
    free(leader_flag);

    /* CSV output file - only rank 0 opens */
    if (!csv_path) csv_path = getenv("BENCH_CSV");
    if (csv_path && me == 0) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) fprintf(stderr, "cannot open CSV: %s\n", csv_path);
    }

    /* CSV metadata */
    bench_meta_t meta = { "ucc_a2a", "mpi", npes, ppn, NULL, NULL };

    int64_t* source;
    if (posix_memalign((void**)&source, 4096, npes * count * sizeof(int64_t)) != 0) {
        printf("Failed to allocate aligned memory for source\n");
        return -1;
    }
    int64_t* dest = source;

    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t));
    if (maps == NULL) {
        printf("OOM\n");
        free(source);
        return -1;
    }

    maps[0].address = source;
    maps[0].len = npes * count * sizeof(int64_t);

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep = me;
    ctx_params.mem_params.n_segments = 1;
    ctx_params.mem_params.segments = maps;

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
    ucc_lib_config_release(lib_config);

    if (UCC_OK != ucc_context_config_read(ucc_lib, NULL, &ctx_config)) {
        printf("error on ctx\n");
        return -1;
    }

    if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context)) {
        printf("error on ctx create\n");
        return -1;
    }

    ucc_context_config_release(ctx_config);

    size_t work_buffer_size = get_ucc_work_buffer_size(ucc_context);
    /* depth fix: one work buffer per in-flight collective instance —
     * persistent UCC collectives may not share a global_work_buffer
     * concurrently.  pSync[0] doubles as the num==1 buffer. */
    pSync = (long *)alloc_zeroed_aligned(work_buffer_size * MAX_PIPELINE);
    if (NULL == pSync) {
        printf("Failed to allocate UCC work buffer\n");
        return -1;
    }
    static long * pSync_arr[MAX_PIPELINE];
    for (int z = 0; z < MAX_PIPELINE; z++)
        pSync_arr[z] = (long *)((char *)pSync + (size_t)z * work_buffer_size);

    /* QP baseline: before the UCC team is created (UCC runtime UCX
     * eps/DCIs don't exist yet; MPI PML / OSHMEM SMPL QPs do). */
    if (qp_sampling)
        qp_pre_team = qp_self_count();

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

    while (UCC_INPROGRESS == (status = ucc_team_create_test(ucc_team))) {
        ucc_context_progress(ucc_context);
    }
    if (UCC_OK != status) {
        printf("team create failed\n");
        return -1; 
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (qp_sampling && qp_pre_team >= 0)
        qp_post_team = qp_self_count();

    /* Transport detection after UCC init for reliable results */
    transport_info_t ti = transport_detect();
    meta.ucc_tls = ti.ucc_tls;
    meta.ucx_tls = ti.ucx_tls;

    /* ---- node-wide NIC counters (leader of each node only) ---- */
    if (hw_iface) {
        if (is_leader)
            nc = nic_counters_create(hw_iface, backends);
        int local_iio = (nc && (nic_counters_backends(nc) & NC_MASK_IIO)) ? 1 : 0;
        MPI_Allreduce(&local_iio, &iio_active, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (me == 0) {
            if (!nc && is_leader) {
                printf("WARNING: NIC counters requested for %s but no backend "
                       "available on this node, columns stay 0\n", hw_iface);
            } else if (nc) {
                printf("NIC counter monitoring enabled for %s (backends: %s%s%s)\n",
                       hw_iface,
                       (nic_counters_backends(nc) & NC_MASK_SYSCNP) ? "sysfs " : "",
                       (nic_counters_backends(nc) & NC_MASK_IIO) ? "iio " : "", "");
                const char *pmu = nic_counters_iio_pmu(nc);
                if (pmu)
                    printf("IIO backend: PMU %s on socket CPU %d; read "
                           "counters are an ESTIMATE of ICMC misses\n",
                           pmu, iio_pmu_cpu());
            }
        }
    }

    /* ---- noise floor: IIO deltas over an idle (no-collective) window ----
     * The idle window must NOT run MPI_Barrier in a loop: with UCC as the
     * MPI collective layer the barrier can return before every rank
     * arrives, so ranks desync (one slips into the first alltoall while
     * the others are still in the barrier) and the run deadlocks.  Instead:
     * align once, idle-spin locally for NOISE_WIN_S, align once. */
    if (hw_iface && iio_active) {
        nc_snapshot_t a, b, d;
        MPI_Barrier(MPI_COMM_WORLD);          /* align all ranks on t0 */
        double t0 = MPI_Wtime();
        if (is_leader && nc)
            nic_counters_snapshot(nc, &a);
        while (MPI_Wtime() - t0 < NOISE_WIN_S) { /* local idle, no collective */ }
        MPI_Barrier(MPI_COMM_WORLD);          /* re-align after idle window */
        if (is_leader && nc) {
            nic_counters_snapshot(nc, &b);
            nic_counters_diff(&a, &b, &d);
            if (d.valid & NC_MASK_IIO) {
                noise_txn   = d.iio_rd_txn_64b;
                noise_units = d.iio_rd_units_4b;
            }
        }
    }

    int want_iio_cols = hw_iface && backends && strstr(backends, "iio");

    if (me == 0) {
        printf("%-10s%-12s%15s%15s%15s%14s%14s%14s%14s",
               "Size (B)", "Total (B)",
               "BW (MB/s)", "Agg BW (MB/s)", "Max BW (MB/s)",
               "Avg Lat (us)", "Min Lat (us)", "Max Lat (us)", "Var (us^2)");
        if (hw_iface) {
            printf("%15s%15s%15s%14s",
                   "CNP Sent", "CNP Handled", "CNP Ignored", "ECN Marked");
            if (want_iio_cols)
                printf("%15s%15s%18s%15s",
                       "IIO RD Txn", "IIO RD B",
                       "Excess RD B/msg", "ICM Fetch/msg");
        }
        if (qp_sampling)
            printf("%12s%12s%24s", "QP Node", "QP Rank", "QP Types");
        if (compute_usec > 0.0)
            printf("%10s%14s%14s", "Depth", "Compute us", "Window us");
        printf("\n");
    }

    if (me == 0 && csv_fp) {
        char ext_cols[512];
        snprintf(ext_cols, sizeof(ext_cols), "%s,pipeline_depth,compute_us,window_us",
                 BENCH_EXT_COLS_ICMC);
        bench_csv_header_ext(csv_fp, ext_cols);
    }

    size_stats_t st = {0};
    size_stats_t gr = {0};
    uint64_t u64r;

    for (int k = 1; k <= count; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        double total_bw = 0, min = 0;
        double welford_mean = 0.0, welford_M2 = 0.0;
        int welford_count = 0;
        min = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;
        compute_total = 0.0;   /* per-size-point, like total/welford */
        memset(&st, 0, sizeof(st));

        nc_snapshot_t win_a, win_b;
        double win_t0 = 0, win_t1 = 0;
        int have_window = 0;
        uint64_t local_qp_self = 0;

        /* alltoall */
        for (int i = 0; i < (int)(iter + SKIP); i++) {
            /* one counter window per message size, around all TIMED
             * iterations; QP sampled after SKIP warm-up. */
            if (i == SKIP) {
                if (qp_sampling) {
                    if (is_leader) {
                        qp_count_t qnode;
                        if (qp_query(hw_iface, -1, &qnode, NULL) == 0) {
                            st.qp_node   = qnode.total;
                            st.qp_ntypes = qnode.ntypes;
                            for (int t = 0; t < qnode.ntypes; t++) {
                                snprintf(st.qp_names[t],
                                         sizeof(st.qp_names[t]),
                                         "%s", qnode.types[t].name);
                                st.qp_types[t] = qnode.types[t].count;
                            }
                        }
                    }
                    int self = qp_self_count();
                    local_qp_self = (self >= 0) ? (uint64_t)self : 0;
                }
                if (hw_iface && is_leader && nc) {
                    nic_counters_snapshot(nc, &win_a);
                    win_t0 = MPI_Wtime();
                    have_window = 1;
                }
            }

            long * a_psync = pSync;
            ucc_coll_args_t coll_args = {
                .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags     = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS |
                              UCC_COLL_ARGS_FLAG_PERSISTENT,
                .coll_type = UCC_COLL_TYPE_ALLTOALL,
                .src.info =
                    {
                        .buffer   = (void *)source,
                        .count    = k * npes,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_HOST,
                    },
                .dst.info =
                    {
                        .buffer   = (void *)dest,
                        .count    = k * npes,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_HOST,
                    },
                .global_work_buffer = a_psync,
            };

            /* depth fix: init BEFORE the start barrier (out of the timed
             * window), post all `num` after the barrier (true concurrent
             * depth), drain, finalize AFTER the end barrier. */
            ucc_coll_req_h reqs[MAX_PIPELINE];
            for (int z = 0; z < num; z++) {
                reqs[z] = NULL;
                ucc_coll_args_t z_args = coll_args;   /* private copy per in-flight instance */
                z_args.global_work_buffer = pSync_arr[z];
                status = ucc_collective_init(&z_args, &reqs[z], ucc_team);
                if (status != UCC_OK) {
                    printf("coll init failed\n");
                    return -1;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
            for (int z = 0; z < num; z++) {
                status = ucc_collective_post(reqs[z]);
                if (status != UCC_OK) {
                    printf("FAILED TO POST\n");
                    abort();
                }
            }
            /* compute phase: injected CPU work while all d collectives
             * are in flight (bw-pipeline's run_ucc shape).  Inside the
             * timed window on purpose: window = max(comm, comm-after-
             * compute) + bookends; -w 0 leaves this a no-op. */
            if (compute_usec > 0.0) {
                double ct0 = MPI_Wtime();
                do_compute_cpu_(compute_usec);
                if (i >= SKIP) compute_total += MPI_Wtime() - ct0;
            }
            for (int z = 0; z < num; z++) {
                while (UCC_OK != (status = ucc_collective_test(reqs[z]))) {
                    if (0 > status) {
                        printf("collective failed\n");
                        abort();
                    }
                    ucc_context_progress(ucc_context);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            end = MPI_Wtime();
            for (int z = 0; z < num; z++)
                ucc_collective_finalize(reqs[z]);

            /* post-first-collective QP sample (lazy eps/DCI creation has
             * happened) - once, right after the very first collective. */
            if (i == 0 && qp_sampling && qp_pre_team >= 0)
                qp_post_coll = qp_self_count();

            if (i == (int)(iter + SKIP) - 1 && have_window) {
                if (hw_iface && is_leader && nc) {
                    nic_counters_snapshot(nc, &win_b);
                    win_t1 = MPI_Wtime();
                }
            }

            if (i >= SKIP) {
                double time = (end - start);
                total += time;
                welford_count++;
                double w_delta = time - welford_mean;
                welford_mean += w_delta / welford_count;
                welford_M2 += w_delta * (time - welford_mean);
                if (time < min) {
                    min = time;
                }
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        /* ---- per-node derived metrics (leader only) ---- */
        if (have_window && hw_iface && is_leader && nc) {
            nc_snapshot_t d;
            nic_counters_diff(&win_a, &win_b, &d);
            double win_s = win_t1 - win_t0;
            if (d.valid & NC_MASK_SYSCNP) {
                st.cnp[0] = d.np_cnp_sent;
                st.cnp[1] = d.rp_cnp_handled;
                st.cnp[2] = d.rp_cnp_ignored;
                st.cnp[3] = d.np_ecn_marked;
            }
            if (d.valid & NC_MASK_IIO) {
                double noise_txn_s   = (double)noise_txn / NOISE_WIN_S;
                double noise_units_s = (double)noise_units / NOISE_WIN_S;
                uint64_t rd_bytes    = d.iio_rd_units_4b * 4u;
                /* NIC messages this node drives: each rank puts
                 * ceil(msg_bytes/frag) messages to each of npes-1 peers. */
                uint64_t msg_bytes   = (uint64_t)k * sizeof(uint64_t);
                uint64_t frags       = (msg_bytes + (uint64_t)put_frag - 1) /
                                       (uint64_t)put_frag;
                uint64_t nic_msgs    = (uint64_t)(npes - 1) * frags *
                                       (uint64_t)iter * (uint64_t)ppn;
                uint64_t payload_read_B = (uint64_t)(npes - 1) * msg_bytes *
                                          (uint64_t)iter * (uint64_t)ppn;

                st.iio_rd_txn   = d.iio_rd_txn_64b;
                st.iio_rd_bytes = rd_bytes;

                double excess_B = (double)rd_bytes - (double)payload_read_B -
                                  noise_units_s * 4.0 * win_s;
                if (excess_B < 0) excess_B = 0;
                double excess_txn = (double)d.iio_rd_txn_64b -
                                    noise_txn_s * win_s;
                if (excess_txn < 0) excess_txn = 0;

                if (nic_msgs > 0) {
                    st.excess_rd_B_per_msg = excess_B / (double)nic_msgs;
                    st.icm_fetch_per_msg   = excess_txn / (double)nic_msgs;
                }
            }
        }

        /* ---- reduce across the world (non-leaders contribute 0) ----
         * CNP/ECN: node SUM (each leader owns a distinct device);
         * IIO raw: node SUM; derived per-msg metrics: per-node MAX
         * (worst node); QP counts: per-node MAX. */
        for (int j = 0; j < NUM_HW_COUNTERS; j++)
            MPI_Allreduce(&st.cnp[j], &gr.cnp[j], 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
        MPI_Allreduce(&st.iio_rd_txn, &gr.iio_rd_txn, 1, MPI_UINT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&st.iio_rd_bytes, &gr.iio_rd_bytes, 1, MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&st.excess_rd_B_per_msg, &gr.excess_rd_B_per_msg, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&st.icm_fetch_per_msg, &gr.icm_fetch_per_msg, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&st.qp_node, &gr.qp_node, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        for (int t = 0; t < QP_TYPE_MAX; t++)
            MPI_Allreduce(&st.qp_types[t], &gr.qp_types[t], 1, MPI_UINT64_T,
                          MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&st.qp_ntypes, &u64r, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        gr.qp_ntypes = (int)u64r;
        /* type names: the node-0 leader holds the node-wide breakdown
         * (identical topology on all nodes); stage it into gr and bcast. */
        if (is_leader)
            memcpy(gr.qp_names, st.qp_names, sizeof(gr.qp_names));
        MPI_Bcast(gr.qp_names, sizeof(gr.qp_names), MPI_CHAR, leader0,
                  MPI_COMM_WORLD);
        /* per-rank own-pid QPs: every rank participates (measured, not
         * computed) */
        MPI_Allreduce(&local_qp_self, &u64r, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        uint64_t qp_rank = u64r;

        double global_min, global_max, global_total;
        MPI_Allreduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_latency, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* compute bookkeeping: worst-rank average compute per timed iter,
         * and the average timed window per iter (pre depth-normalization;
         * the window CONTAINS compute when -w > 0). */
        double compute_local = (iter > 0) ? compute_total / (double)iter : 0.0;
        double compute_avg;
        MPI_Allreduce(&compute_local, &compute_avg, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        double window_avg_us = (npes > 0 && iter > 0)
                                   ? (global_total / (double)npes / (double)iter) * 1e6
                                   : 0.0;

        min_latency = global_min;
        max_latency = global_max;
        double n = (double)npes * (double)iter;
        double avg_time = global_total / npes;
        double global_mean = global_total / n;
        /* Combine per-rank Welford M2 with correction for difference between
         * local and global mean, then allreduce for numerically stable variance. */
        double local_dev_sq = welford_M2 + (welford_mean - global_mean) *
                              (welford_mean - global_mean) * welford_count;
        double global_dev_sq;
        MPI_Allreduce(&local_dev_sq, &global_dev_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double variance_us2 = (global_dev_sq / n) * 1e12;

        /* Per-collective normalization for true pipeline depth: the timed
         * window covers `num` concurrent collectives, so report per-
         * collective latency (window/num) and count each collective's
         * payload (BW formulas below see latency-per-collective).
         * num==1: exact no-op -> prior results unchanged. */
        if (num > 1) {
            avg_time    /= num;
            min_latency /= num;
            max_latency /= num;
        }

        total_bw = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (double)iter) / avg_time;
        src_buff = bandwidth;

        MPI_Allreduce(&src_buff, &agg_bandwidth, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (me == 0) {
            char types[256];
            if (qp_sampling) {
                size_t tn = 0;
                for (int t = 0; t < (int)gr.qp_ntypes; t++) {
                    tn += (size_t)snprintf(types + tn, sizeof(types) - tn,
                                           "%s%s=%llu", t ? " " : "",
                                           gr.qp_names[t],
                                           (unsigned long long)gr.qp_types[t]);
                }
                if (types[0] == '\0') snprintf(types, sizeof(types), "-");
            } else {
                snprintf(types, sizeof(types), "-");
            }

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

            if (hw_iface) {
                printf("%15lu%15lu%15lu%14lu",
                       gr.cnp[0], gr.cnp[1], gr.cnp[2], gr.cnp[3]);
                if (want_iio_cols)
                    printf("%15lu%15lu%18.3f%15.4f",
                           gr.iio_rd_txn, gr.iio_rd_bytes,
                           gr.excess_rd_B_per_msg, gr.icm_fetch_per_msg);
            }
            if (qp_sampling)
                printf("%12lu%12llu%24.24s",
                       gr.qp_node, (unsigned long long)qp_rank, types);
            if (compute_usec > 0.0)
                printf("%10d%14.2f%14.2f", num, compute_avg * 1e6, window_avg_us);
            printf("\n");

            /* CSV row */
            if (csv_fp) {
                char extra[640];
                snprintf(extra, sizeof(extra),
                         "%llu,%llu,%s,%llu,%llu,%.4f,%.4f,%d,%.2f,%.2f",
                         (unsigned long long)gr.qp_node,
                         (unsigned long long)qp_rank,
                         (hw_iface || qp_sampling) ? types : "",
                         (unsigned long long)gr.iio_rd_txn,
                         (unsigned long long)gr.iio_rd_bytes,
                         (hw_iface && want_iio_cols) ?
                             gr.excess_rd_B_per_msg : 0.0,
                         (hw_iface && want_iio_cols) ?
                             gr.icm_fetch_per_msg : 0.0,
                         num, compute_avg * 1e6, window_avg_us);
                bench_csv_row_ext(csv_fp, &meta,
                                  k * sizeof(uint64_t), (int)iter,
                                  avg_time * 1e6 / iter, min_latency * 1e6,
                                  max_latency * 1e6, sqrt(variance_us2),
                                  bandwidth / (1024.0 * 1024.0),
                                  extra);
            }
        }
    }

    /* ---- QP baseline summary (rank 0) ---- */
    if (qp_sampling && me == 0) {
        printf("\nQP baseline (rank 0, this process): pre-team=%d "
               "post-team=%d post-first-coll=%d\n",
               qp_pre_team, qp_post_team, qp_post_coll);
        if (qp_pre_team < 0)
            printf("WARNING: QP sampling requested but no QP backend worked "
                   "(`rdma -j res show qp`); QP columns stay 0.\n");
        /* sanity: with RC, N ranks on one node must own >= (ppn-1) QPs
         * each; with DC, the node must show the configured DCI count. */
        const char *tls = getenv("UCX_TLS");
        const char *ndci = getenv("UCX_DC_MLX5_NUM_DCI");
        if (tls && qp_pre_team >= 0) {
            if (strstr(tls, "rc") && strstr(tls, "dc") == NULL) {
                int expected = (ppn > 1) ? (ppn - 1) : (npes > 1 ? npes - 1 : 0);
                if (qp_pre_team > 0 && qp_pre_team < expected)
                    printf("WARNING: UCX_TLS=rc expected >= %d RC QPs per rank "
                           "after team create, saw %d\n",
                           expected, qp_post_team);
            } else if (ndci) {
                int k = atoi(ndci);
                if (k > 0 && gr.qp_node > 0 &&
                    (int)gr.qp_node < k * n_nodes)
                    printf("WARNING: UCX_DC_MLX5_NUM_DCI=%d expected >= %d DCI "
                           "QPs node-wide, saw %llu total\n",
                           k, k * n_nodes,
                           (unsigned long long)gr.qp_node);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (csv_fp && me == 0) { fflush(csv_fp); fclose(csv_fp); }
    transport_free(ti.ucc_tls);
    transport_free(ti.ucx_tls);

    if (is_leader) nic_counters_destroy(nc);
    free(source);
    free(pSync);
    free(maps);
    if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);

    MPI_Finalize();
    return 0;
}
