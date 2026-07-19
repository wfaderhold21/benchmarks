/*
 *  This benchmark measures bandwidth and latency for a2a calls using
 *  UCC with CUDA device buffers.
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

#include <cuda_runtime.h>

#include <ucc/api/ucc.h>

#include "../../common/bench_output.h"
#include "../../common/transport_detect.h"

#define NR_ITER     100
#define SKIP        10

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
        if (fp == NULL) {
            return 0;
        }
        fclose(fp);
    }
    return 1;
}

int read_hw_counters(hw_counter_data_t* data, const char *base_path) {
    char filepath[256];
    FILE* fp;

    if (!data->hw_counters_available) {
        return 0;
    }

    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        snprintf(filepath, sizeof(filepath), "%s%s", base_path, hw_counter_files[i]);
        fp = fopen(filepath, "r");
        if (fp == NULL) {
            data->hw_counters_available = 0;
            return 0;
        }

        if (fscanf(fp, "%lu", &data->counters[i]) != 1) {
            fclose(fp);
            data->hw_counters_available = 0;
            return 0;
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
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)(uintptr_t)request;
#if 1
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    *req = UCC_OK;
#endif
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

static cudaError_t cuda_or_die(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(e));
    }
    return e;
}

int main(int argc, char ** argv)
{
    int me;
    int npes;
    int count = 1048576;
    long * pSync;
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
    int use_reg = 0;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context;
    ucc_mem_map_t *maps = NULL;
    ucc_team_h ucc_team;
    ucc_team_params_t team_params;
    ucc_status_t status;
    ucc_lib_h ucc_lib;
    uint64_t local_count = 0;
    uint64_t global_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    while ((c = getopt(argc, argv, "i:s:d:p:c:o:r")) != -1) {
        switch (c) {
            case 's': size   = atoi(optarg); break;
            case 'i': iter   = atoi(optarg); break;
            case 'd': num    = atoi(optarg); break;
            case 'p': ppn    = atoi(optarg); break;
            case 'c': hw_iface = optarg;     break;
            case 'o': csv_path = optarg;      break;
            case 'r': use_reg = 1;           break;
            default: return -1;
        }
    }

    /* CSV output file - only rank 0 opens */
    if (!csv_path) csv_path = getenv("BENCH_CSV");
    if (csv_path && me == 0) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) fprintf(stderr, "cannot open CSV: %s\n", csv_path);
    }

    bench_meta_t meta = { "ucc_a2a", use_reg ? "cuda_reg" : "cuda", npes, ppn, NULL, NULL };

    if (hw_iface) {
        snprintf(hw_counter_base_path, sizeof(hw_counter_base_path),
                 "/sys/class/infiniband/%s/ports/1/hw_counters/", hw_iface);
    }

    size_t buf_len = npes * count * sizeof(int64_t);
    int64_t* d_source;
    cuda_or_die(cudaMalloc((void **)&d_source, buf_len), "cudaMalloc source");
    int64_t* d_dest;
    cuda_or_die(cudaMalloc((void **)&d_dest, buf_len), "cudaMalloc dest");

    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t) * 2);
    if (maps == NULL) {
        printf("OOM\n");
        return -1;
    }

    maps[0].address = d_source;
    maps[0].len     = buf_len;
    maps[1].address = d_dest;
    maps[1].len     = buf_len;

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test  = oob_allgather_test;
    ctx_params.oob.req_free  = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep    = me;
    ctx_params.mem_params.n_segments = 2;
    ctx_params.mem_params.segments   = maps;

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
    pSync = (long *)alloc_zeroed_aligned(work_buffer_size);
    if (NULL == pSync) {
        printf("Failed to allocate UCC work buffer\n");
        return -1;
    }

    team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = MPI_COMM_WORLD;
    team_params.oob.n_oob_eps = npes;
    team_params.oob.oob_ep    = me;
    team_params.ep            = me;
    team_params.flags         = UCC_TEAM_FLAG_COLL_WORK_BUFFER;

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

    /* Transport detection after UCC init for reliable results */
    transport_info_t ti = transport_detect();
    meta.ucc_tls = ti.ucc_tls;
    meta.ucx_tls = ti.ucx_tls;
    if (csv_fp && me == 0) bench_csv_header(csv_fp);

    /* Optional: register the CUDA buffer with UCC mem registry */
    ucc_mem_h   cuda_mem = NULL;
    ucc_memory_info_t cuda_mem_info;
    if (use_reg) {
        cuda_mem_info.address = d_source;
        cuda_mem_info.length  = buf_len;
        status = ucc_mem_register(ucc_context, &cuda_mem_info, 1, &cuda_mem);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_mem_register failed: %d\n", (int)status);
        } else {
            printf("Rank %d: CUDA buffer registered with UCC mem registry\n", me);
        }
    }

    if (hw_iface) {
        hw_counters_available_check.hw_counters_available = check_hw_counters_available(hw_counter_base_path);
        if (me == 0) {
            if (hw_counters_available_check.hw_counters_available) {
                printf("Hardware counter monitoring enabled for %s\n", hw_iface);
            } else {
                printf("Hardware counter monitoring requested for %s but counters not available, disabling\n", hw_iface);
                hw_iface = NULL;
            }
        }
    }

    if (me == 0) {
        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            printf("%-10s%-12s%15s%15s%15s%14s%14s%14s%14s%15s%15s%15s%20s\n",
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
        min = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;

        hw_counter_data_t total_size_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};

        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            for (int i = 0; i < NUM_HW_COUNTERS; i++) {
                total_size_counters.counters[i] = 0;
            }
        }

        /* alltoall */
        for (int i = 0; i < iter + SKIP; i++) {
            long * a_psync = pSync;
            ucc_coll_args_t coll_args = {
                .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags     = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS |
                               UCC_COLL_ARGS_FLAG_PERSISTENT,
                .coll_type = UCC_COLL_TYPE_ALLTOALL,
                .src.info =
                    {
                        .buffer   = (void *)d_source,
                        .count    = k * npes,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_CUDA,
                    },
                .dst.info =
                    {
                        .buffer   = (void *)d_dest,
                        .count    = k * npes,
                        .datatype = UCC_DT_INT64,
                        .mem_type = UCC_MEMORY_TYPE_CUDA,
                    },
                .global_work_buffer = a_psync,
            };

            if (cuda_mem) {
                coll_args.src.info.mem_handle = cuda_mem;
                coll_args.dst.info.mem_handle = cuda_mem;
            }

            hw_counter_data_t iter_start_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
            if (hw_iface && hw_counters_available_check.hw_counters_available) {
                read_hw_counters(&iter_start_counters, hw_counter_base_path);
            }

            ucc_coll_req_h req = NULL;
            status = ucc_collective_init(&coll_args, &req, ucc_team);
            if (status != UCC_OK) {
                printf("coll init failed\n");
                return -1;
            }

            MPI_Barrier(MPI_COMM_WORLD);
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
            }
            ucc_collective_finalize(req);

            MPI_Barrier(MPI_COMM_WORLD);
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
                if (time < min) {
                    min = time;
                }
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        hw_counter_data_t global_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        if (hw_iface && hw_counters_available_check.hw_counters_available) {
            for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                global_counters.counters[j] = 0;
            }
            for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                local_count = total_size_counters.counters[j];
                MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                global_counters.counters[j] = global_count;
            }
        }

        double global_min, global_max, global_total;
        MPI_Allreduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_latency, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        min_latency = global_min;
        total_time = global_total;
        max_latency = global_max;
        double n = (double)npes * (double)iter;
        double avg_time = global_total / npes;
        double global_mean = global_total / n;
        double local_dev_sq = welford_M2 + (welford_mean - global_mean) *
                              (welford_mean - global_mean) * welford_count;
        double global_dev_sq;
        MPI_Allreduce(&local_dev_sq, &global_dev_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double variance_us2 = (global_dev_sq / n) * 1e12;

        total_bw = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (double)iter) / avg_time;
        src_buff = bandwidth;

        MPI_Allreduce(&src_buff, &agg_bandwidth, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
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

            if (csv_fp) {
                bench_csv_row(csv_fp, &meta,
                              k * sizeof(uint64_t), (int)iter,
                              avg_time * 1e6 / iter, min_latency * 1e6,
                              max_latency * 1e6, sqrt(variance_us2),
                              bandwidth / (1024.0 * 1024.0));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (csv_fp && me == 0) { fflush(csv_fp); fclose(csv_fp); }

    if (cuda_mem) {
        ucc_mem_deregister(ucc_context, &cuda_mem, 1);
    }

    cudaFree(d_source);
    cudaFree(d_dest);
    free(pSync);
    free(maps);

    transport_free(ti.ucc_tls);
    transport_free(ti.ucx_tls);

    MPI_Finalize();
    return 0;
}
