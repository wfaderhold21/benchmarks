#include <shmem.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <shmemx.h>
#include <assert.h>
#include <mpi.h>
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>
#include <sched.h>

enum {
    FUNC_FAILURE = -1,
    FUNC_SUCCESS,
    FUNC_NOXGVMI
};

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *addr;
    size_t        len;
    void         *packed_memh;
    size_t        packed_memh_len;
    void         *packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct bench_desc {
   double compute_time;
   int    loop;
   int    msg_size;
   int    numprocs;
   int    myid;
   char*  name;
   char   *s_buf;
   char   *r_buf;
   int    *counters;
   int    num_counters;
   int    p_d;
   char*  pSync_buf;
   int    use_xgvmi; /* use xgvmi in offload */
   int    ppw; /* processes per worker */
   double (*run)(struct bench_desc *d, int do_compute, double comp_time,
                 double *init_t, double *comp_t, double *wait_t);
}bench_desc_t;

long pSyncRed1[_SHMEM_REDUCE_SYNC_SIZE];
long pSyncRed2[_SHMEM_REDUCE_SYNC_SIZE];
long pSync_a2a[_SHMEM_ALLTOALL_SYNC_SIZE];

double pWrk1[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double pWrk2[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];

#define FIELD_WIDTH 20

#define FLOAT_PRECISION 2

#define max(a, b) \
({ \
    typeof(a) _a = (a);  \
    typeof(b) _b = (b);  \
    _a > _b ? _a : _b;   \
})

double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

#define TIME getMicrosecondTimeStamp

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = calloc(1, sizeof(ucc_status_t));
    /* FIXME: MPI_Test in oob_allgather_test results in no completion? leave as blocking for now */
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    *req = UCC_OK;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    return UCC_OK;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}


int ucp_init_ex(ucp_context_h *ucp_ctx, ucp_worker_h *ucp_worker, ucp_address_t **ucp_addr, size_t *len)
{
    ucs_status_t  ucs_status;
    ucp_config_t *ucp_config;
    ucp_params_t  ucp_params;
    ucp_context_h ucp_context;
    ucp_worker_params_t worker_params;
    ucp_address_t *worker_addr;
    ucp_worker_h worker;
    size_t length;

    ucs_status = ucp_config_read(NULL, NULL, &ucp_config);
    assert(ucs_status == UCS_OK);

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG | UCP_FEATURE_RMA |
                          UCP_FEATURE_AMO64 | UCP_FEATURE_EXPORTED_MEMH;

    ucs_status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    if (ucs_status != UCS_OK) {
        return FUNC_FAILURE;
    }

    *ucp_ctx = ucp_context;

    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status = ucp_worker_create(ucp_context, &worker_params, &worker);
    if (ucs_status != UCS_OK) {
        printf("error on worker create\n");
        return FUNC_FAILURE;
    }

    *ucp_worker = worker;
    ucs_status = ucp_worker_get_address(worker, &worker_addr, &length);
    if (ucs_status != UCS_OK) {
        printf("failed to get address\n");
        return FUNC_FAILURE;
    }
    *ucp_addr = worker_addr;
    *len = length;

    return 0;
}

int buffer_export_ucc(ucp_context_h ucp_context, void *buf, size_t len,
                      struct export_buf *ebuf)
{
    ucs_status_t           ucs_status;
    ucp_mem_map_params_t   params;
    ucp_memh_pack_params_t pack_params;
    int ret = FUNC_SUCCESS;

    ebuf->ucp_context = ucp_context;
    ebuf->addr = buf;
    ebuf->len = len;

    params.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = buf;
    params.length  = len;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    assert(ucs_status == UCS_OK);

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        printf("ucp_rkey_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        return FUNC_FAILURE;
    }

    return ret;
}


void
compute_on_host_()
{
    int x=0;
    int i = 0, j = 0;
    for (i = 0; i < 25; i++) {
        for (j = 0; j < 25; j++) {
            x = x + i*j;
        }
    }
}

static inline void
do_compute_cpu_(double target_usec)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_usec) {
        t1 = TIME();
        compute_on_host_();
        t2 = TIME();
        time_elapsed += (t2-t1);
    }
}
                     
static double run_ucc(bench_desc_t *d, int do_compute, double comp_time, double *init_t, double *comp_t, double *wait_t, int rank, int size, ucc_team_h team, ucc_context_h context)
{
    int i, j;
    double t_total, comp, init, wait, total;
    static int round = 0;
    static int f = 1;
    int num_ucc_posts = 0;
    ucc_coll_req_h req[d->p_d];
    ucc_status_t status;
    int64_t *src_count = calloc(1,sizeof(int64_t) * size);
    int64_t *dst_count = calloc(1,sizeof(int64_t) * size);
    int64_t *src_disp = calloc(1,sizeof(int64_t) * size);
    int64_t *dst_disp = calloc(1,sizeof(int64_t) * size);
    int64_t disp = 0;
    int64_t in_disp = 0;

    for (int i = 0; i < size; i++) {
        src_count[i] = d->msg_size / sizeof(int64_t);
        dst_count[i] = d->msg_size / sizeof(int64_t);
        src_disp[i] = disp;
        dst_disp[i] = in_disp;
        disp += d->msg_size / sizeof(int64_t);// * /*sizeof(int64_t) * k;
        in_disp += d->msg_size / sizeof(int64_t);// * /*sizeof(int64_t) * k;
    }

    ucc_coll_args_t coll = {
        .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
        .flags     = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT 
                   | UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
        .coll_type = UCC_COLL_TYPE_ALLTOALLV,
        .src.info_v = {
            .buffer = d->s_buf,
            .counts = src_count,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_HOST,
            .displacements = src_disp,
        },
        .dst.info_v = {
            .buffer = d->r_buf,
            .counts = dst_count,
            .datatype = UCC_DT_INT64,
            .mem_type = UCC_MEMORY_TYPE_HOST,
            .displacements = dst_disp,
        },
        .global_work_buffer = d->pSync_buf,
    };

    *init_t = *comp_t = *wait_t = t_total = 0.0, total = 0.0;

    total = TIME();
    for (i = 0; i < d->loop; i++) {
        int index = i % 16;
        coll.global_work_buffer = d->pSync_buf + index * sizeof(long);
        init = TIME();
        status = ucc_collective_init(&coll, &req[num_ucc_posts], team);
        if (status != UCC_OK) {
            abort();
        }
        status = ucc_collective_post(req[num_ucc_posts]);
        if (status != UCC_OK) {
            abort();
        }
        *init_t += TIME() - init;
        num_ucc_posts++;
        if (do_compute) {
            comp = TIME();
            do_compute_cpu_(comp_time);
            *comp_t += TIME() - comp;
        }
        if (!(round++ % d->p_d)) {
            wait = TIME();
            for (int k = 0; k < num_ucc_posts; ++k) {
                while (UCC_OK != (status = ucc_collective_test(req[k]))) {
                    if (0 > status) {
                        printf("UCC ERROR %s\n", ucc_status_string(status));
                        abort();
                        return 0.0;
                    }
                    ucc_context_progress(context);
                } 
                ucc_collective_finalize(req[k]);
            }
            *wait_t += TIME() - wait;
            num_ucc_posts = 0;

            if (!f) {
                f = 1;
            }
        }
        shmem_barrier_all();
    }
    t_total = TIME() - total;
    shmem_barrier_all();
   
    free(src_count);
    free(dst_count);
    free(src_disp);
    free(dst_disp);
    return t_total / d->loop;
}

int setup_ucc(int rank, int size, bench_desc_t *desc, 
              struct export_buf *ebuf, ucc_lib_h *lib, 
              ucc_context_h *context, ucc_team_h *team)
{
    ucc_context_h ucc_context;
    ucc_team_h  ucc_team;
    ucc_lib_h ucc_lib;
    ucc_team_params_t team_params = {0};
    ucc_lib_config_h lib_config;
    ucc_lib_params_t lib_params;
    ucc_context_config_h ctx_config;
    ucc_context_params_t ctx_params = {0};
    ucc_mem_map_t maps[3] = {0};
    ucc_status_t status;

    /* make ucc here */
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;

    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config)) {
        printf("lib config error\n");
        return -1;
    }

    if (UCC_OK != ucc_init(&lib_params, lib_config, &ucc_lib)) {
        printf("lib init error\n");
        return -1;
    }
    *lib = ucc_lib;

    maps[0].address = desc->s_buf;
    maps[0].len = ebuf->len;
    maps[1].address = desc->pSync_buf;
    maps[1].len = 16 * sizeof(long);

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test  = oob_allgather_test;
    ctx_params.oob.req_free  = oob_allgather_free;
    ctx_params.oob.coll_info = MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = size;
    ctx_params.oob.oob_ep    = rank;
    ctx_params.mem_params.segments = maps;
    ctx_params.mem_params.n_segments = 2;

    if (UCC_OK != ucc_context_config_read(ucc_lib, NULL, &ctx_config)) {
        printf("error ucc ctx config read\n");
        return -1;
    }

    if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context)) {
        printf("ERROR ucc ctx create\n");
        return -1;
    }
    *context = ucc_context;

    ucc_context_config_release(ctx_config);

    team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test = oob_allgather_test;
    team_params.oob.req_free = oob_allgather_free;
    team_params.oob.coll_info = MPI_COMM_WORLD;
    team_params.oob.n_oob_eps = size;
    team_params.oob.oob_ep = rank;
    team_params.ep = rank;
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
    *team = ucc_team;

    return 0;
}

int main(int argc, char *argv[])
{
    int numprocs, i, j, ppn;
    int num_counters;
    bench_desc_t desc;
    char *s_buf_heap, *r_buf_heap;
    int align_size, is_many_counters;
    static double comp_total, wait_total, timer, init_total;
    static double latency_avg, latency,  avg_time, max_time, min_time;
    double avg_wt, avg_it, avg_ct, avg_tt;
    char c;
    int mppw;
    int use_host;

#ifdef OSHM_1_3
    shmem_init();
    desc.myid       = shmem_my_pe();
    desc.numprocs   = shmem_n_pes();
#else
    start_pes(0);
    desc.myid       = _my_pe();
    desc.numprocs   = _num_pes();
#endif
    numprocs        = desc.numprocs;

    /* Defaults */
    desc.loop         = 10000;
    desc.msg_size     = 8192;
    desc.p_d          = num_counters = 1;
    desc.name         = "barrier";
    desc.compute_time = -1;
    desc.use_xgvmi    = 0;
    mppw              = 0;
    ppn               = 1;
    is_many_counters  = 0;
    use_host = 1;

    while ((c = getopt(argc, argv, "i:s:d:p:w:u:bx")) != -1) {
        switch (c) {
            case 's':
                desc.msg_size = atoi(optarg);
                break;
            case 'i':
                desc.loop = atoi(optarg);
                break;
            case 'd':
                desc.p_d = atoi(optarg);
                break;
            case 'p':
                ppn = atoi(optarg);
                break;
            case 'w':
                desc.compute_time = atof(optarg);
                break;
            case 'x':
                desc.use_xgvmi = 1;
                break;
            case 'u':
                mppw = atoi(optarg);
                break;
            case 'b':
                use_host = 0;
                break;
            default:
                return -1;
        }
    }

    desc.num_counters = is_many_counters ?  max(desc.p_d, desc.loop) : desc.p_d;
    align_size = 64;
    shmem_barrier_all();

    for (int i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        pSync_a2a[i] = SHMEM_SYNC_VALUE;
    }

#ifdef OSHM_1_3
    desc.counters   = (int *)shmem_malloc(desc.num_counters*sizeof(int));
#else
    desc.counters   = (int *)shmalloc(desc.num_counters*sizeof(int));
#endif
    /* init urom */
    ucp_context_h         ucp_ctx;
    ucp_worker_h          ucp_worker;
    ucp_address_t        *ucp_worker_addr;
    size_t                ucp_worker_addr_len;
    int                   rank, size;
    char                 *device       = NULL;
    char *sr_buffer;
    char *send_buf;
    char *recv_buf;
    int ret = 0;
    struct export_buf     ebuf;
    ucc_context_h urom_ucc_context;
    ucc_team_h urom_ucc_team;
    size_t length = desc.msg_size;
    size_t alloc_size;
    ucc_lib_h ucc_lib;

    rank = shmem_my_pe();
    size = shmem_n_pes();
    uint64_t              worker_id;

    alloc_size = 2 * size * length;

    sr_buffer = shmem_malloc(alloc_size);
    send_buf = sr_buffer;
    recv_buf = sr_buffer + (size * length);

    desc.s_buf = (char *)send_buf;
    desc.r_buf = (char *)recv_buf;
    desc.pSync_buf = shmem_calloc(sizeof(long), 16);

    for (i = 0; i < length; i++) {
        desc.s_buf[i] = 'a';
        desc.r_buf[i] = 'b';
    }

    /* initialize ucx here */
    ret = ucp_init_ex(&ucp_ctx, &ucp_worker, &ucp_worker_addr, &ucp_worker_addr_len);
    if (ret != FUNC_SUCCESS) {
        printf("error on ucp_init\n");
        return ret;
    }

    ret = buffer_export_ucc(ucp_ctx, sr_buffer, alloc_size, &ebuf);
    if (ret == FUNC_FAILURE) {
        printf("failed to export buffer\n");
        return ret;
    } else if (ret == FUNC_NOXGVMI) {
        desc.use_xgvmi = 0;
    }

    ret = setup_ucc(rank, size, &desc, &ebuf, &ucc_lib, &urom_ucc_context, &urom_ucc_team);
    if (ret != 0) {
        printf("failure in ucc setup\n");
        return ret;
    }

    if(desc.myid == 0) {
        fprintf(stdout, "# SHMEM overlap BW benchmark\n");
        fprintf(stdout, "# loop %d; size %d; sync: %s, p_d %d, ctrs %d\n\n",
                desc.loop, desc.msg_size, desc.name, desc.p_d, desc.num_counters);
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Comm Pure(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "BW Pure (MB/s)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH + 10, "Wait(us) avrg|min|max");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Tot ovrlped(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "BW (MB/s)");
        fflush(stdout);
    }

    memset(desc.counters, 0, desc.num_counters * sizeof(int));
    shmem_barrier_all();

    latency = run_ucc(&desc, 0, 0.0, &init_total, &comp_total, &wait_total, rank, size, urom_ucc_team, urom_ucc_context);
    shmem_barrier_all();
    latency = 0.0;
    latency = run_ucc(&desc, 0, 0.0, &init_total, &comp_total, &wait_total, rank, size, urom_ucc_team, urom_ucc_context);
    shmem_double_sum_to_all(&latency_avg, &latency, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    latency_avg /= numprocs;
    if (desc.compute_time < 0) {
        desc.compute_time = latency_avg*0.9;
    }
    memset(desc.counters, 0, desc.num_counters*sizeof(int));
    shmem_barrier_all();

    timer = 0.0;
    timer = run_ucc(&desc, 1, desc.compute_time, &init_total, &comp_total, &wait_total, rank, size, urom_ucc_team, urom_ucc_context);

    shmem_double_min_to_all(&min_time, &wait_total, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    shmem_double_max_to_all(&max_time, &wait_total, 1, 0, 0, numprocs, pWrk2, pSyncRed2);
    shmem_double_sum_to_all(&avg_time, &wait_total, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    avg_wt = avg_time/numprocs;
    shmem_double_sum_to_all(&avg_time, &init_total, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    avg_it = avg_time/numprocs;
    shmem_double_sum_to_all(&avg_time, &comp_total, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    avg_ct = avg_time/numprocs;
    shmem_double_sum_to_all(&avg_time, &timer, 1, 0, 0, numprocs, pWrk1, pSyncRed1);
    avg_tt = avg_time/numprocs;

    if(desc.myid == 0) {
        int LOCAL_WIDTH = 19;
        fprintf(stdout, "%-*d", 10, desc.msg_size);
        fprintf(stdout, "%*.*f%*.*f%*.*f%*.*f%*.*f %.*f %.*f%*.*f%*.*f\n",
                LOCAL_WIDTH, FLOAT_PRECISION, latency_avg,
                LOCAL_WIDTH, FLOAT_PRECISION, desc.msg_size*numprocs*ppn/latency_avg,
                LOCAL_WIDTH, FLOAT_PRECISION, avg_ct/desc.loop,
                LOCAL_WIDTH, FLOAT_PRECISION, avg_it/desc.loop,
                LOCAL_WIDTH, FLOAT_PRECISION, avg_wt/desc.loop,
                FLOAT_PRECISION, min_time/desc.loop,
                FLOAT_PRECISION, max_time/desc.loop,
                LOCAL_WIDTH, FLOAT_PRECISION, avg_tt,
                LOCAL_WIDTH, FLOAT_PRECISION, desc.msg_size*numprocs*ppn/avg_tt);

        fflush(stdout);
    }

#ifdef OSHM_1_3
    shmem_finalize();
#endif
    return EXIT_SUCCESS;
}

