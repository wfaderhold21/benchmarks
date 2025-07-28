/*
 *  This benchmark measures bandwidth and latency for a2a calls in MPI. 
 *
 *  Meant to be used with MPI
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>

#include <ucc/api/ucc.h>

#define NR_ITER     100
#define SKIP        10

// Hardware counter file paths
#define HW_COUNTER_BASE_PATH "/sys/class/infiniband/mlx5_0/ports/1/hw_counters/"
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

int check_hw_counters_available() {
    char filepath[256];
    FILE* fp;

    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        snprintf(filepath, sizeof(filepath), "%s%s", HW_COUNTER_BASE_PATH, hw_counter_files[i]);
        fp = fopen(filepath, "r");
        if (fp == NULL) {
            return 0; // At least one file doesn't exist
        }
        fclose(fp);
    }
    return 1; // All files exist
}

int read_hw_counters(hw_counter_data_t* data) {
    char filepath[256];
    FILE* fp;

    if (!data->hw_counters_available) {
        return 0;
    }

    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        snprintf(filepath, sizeof(filepath), "%s%s", HW_COUNTER_BASE_PATH, hw_counter_files[i]);
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

void print_hw_counter_diff(const char* prefix, hw_counter_data_t* start, hw_counter_data_t* end) {
    if (!start->hw_counters_available || !end->hw_counters_available) {
        return;
    }

    printf("%s Hardware Counters:\n", prefix);
    for (int i = 0; i < NUM_HW_COUNTERS; i++) {
        uint64_t diff = end->counters[i] - start->counters[i];
        printf("  %s: %lu\n", hw_counter_names[i], diff);
    }
}

int verify(const void * src, const int64_t * dest, int64_t *src_count, ucc_aint_t *src_disp, int64_t *dst_count, ucc_aint_t *dst_disp, size_t count, int rank, int npes)
{
    int64_t * t_dest = (int64_t *)malloc(count * npes * sizeof(int64_t));
    int *mpi_src_count = (int *)malloc(npes * sizeof(int));
    int *mpi_src_disp = (int *)malloc(npes * sizeof(int));
    int *mpi_dst_count = (int *)malloc(npes * sizeof(int));
    int *mpi_dst_disp = (int *)malloc(npes * sizeof(int));
   
    MPI_Barrier(MPI_COMM_WORLD); 
    for (int i = 0; i < npes; i++) {
        mpi_src_count[i] = src_count[i];
        mpi_dst_count[i] = dst_count[i];
        mpi_src_disp[i] = src_disp[i];
        mpi_dst_disp[i] = dst_disp[i];
    }
    MPI_Alltoallv(src, mpi_src_count, mpi_src_disp, MPI_LONG, t_dest, mpi_dst_count, mpi_dst_disp, MPI_LONG, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < npes; i++) {
        if (dest[i] != t_dest[i]) {
            printf("[%d] error: does not validate on index i: %d (%ld != %ld)\n", rank, i, dest[i], t_dest[i]);
            return -1;
        }
    }
    free(t_dest);
    free(mpi_src_count);
    free(mpi_src_disp);
    free(mpi_dst_count);
    free(mpi_dst_disp);
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

    return completed ? UCC_OK : UCC_INPROGRESS;
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
    int me;
    int npes;
    int count = 1048576;  // Reduced from 1048576 to avoid memory registration issues
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
    size_t iter = NR_ITER;
    int ppn = 1;
    int monitor_hw_counters = 0;  // Flag for hardware counter monitoring
    hw_counter_data_t hw_counters_available_check = {0};
    char c;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context;
    ucc_mem_map_t *maps = NULL;
    ucc_team_h ucc_team;
    ucc_team_params_t team_params;
    ucc_status_t status;
    ucc_lib_h ucc_lib;
    static uint64_t local_count = 0;
    static uint64_t global_count = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    while ((c = getopt(argc, argv, "i:s:d:p:c")) != -1) {
        switch (c) {
            case 's':
                size = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 'd':
                num = atoi(optarg);
                break;
            case 'p':
                ppn = atoi(optarg);
                break;
            case 'c':
                monitor_hw_counters = 1;
                break;
            default:
                return -1;
        }
    }

    // Allocate memory with proper alignment for InfiniBand registration
    int64_t* source;// = (int64_t *) malloc(npes * count * sizeof(int64_t));
    
    if (posix_memalign((void**)&source, 4096, npes * count * sizeof(int64_t)) != 0) {
        printf("Failed to allocate aligned memory for source\n");
        return -1;
    }
    int64_t* dest = source;

    // Allocate synchronization arrays with proper alignment
    if (posix_memalign((void**)&pSync, 4096, sizeof(long) * 5) != 0) {
        printf("Failed to allocate aligned memory for pSync\n");
        return -1;
    }

    for (int i = 0; i < 5; i++) {
        pSync[i] = 0;
    }

    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t) * 2);
    if (maps == NULL) {
        printf("OOM\n");
        return -1;
    }

    maps[0].address = source;
    maps[0].len = npes * count * sizeof(int64_t);
    maps[1].address = pSync;
    maps[1].len = 5 * sizeof(long);

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep = me;
    ctx_params.mem_params.n_segments = 2;
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
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize hardware counter monitoring if enabled
    if (monitor_hw_counters) {
        hw_counters_available_check.hw_counters_available = check_hw_counters_available();
        if (me == 0) {
            if (hw_counters_available_check.hw_counters_available) {
                printf("Hardware counter monitoring enabled - congestion control counters available\n");
            } else {
                printf("Hardware counter monitoring enabled - WARNING: congestion control counters not available, monitoring disabled\n");
                monitor_hw_counters = 0; // Disable monitoring
            }
        }
    }

    if (me == 0) {
        if (monitor_hw_counters && hw_counters_available_check.hw_counters_available) {
            printf("%-10s%-10s%15s%13s%13s%13s%13s%13s%15s%15s%15s%20s\n", "Size",
                                                  "PE size",
                                                  "Bandwidth MB/s",
                                                  "Agg MB/s",
                                                  "Max BW",
                                                  "Avg Latency",
                                                  "Min Latency",
                                                  "Max Latency",
                                                  "CNP Sent",
                                                  "CNP Handled",
                                                  "CNP Ignored",
                                                  "ECN Marked");
        } else {
            printf("%-10s%-10s%15s%13s%13s%13s%13s%13s\n", "Size",
                                                  "PE size",
                                                  "Bandwidth MB/s",
                                                  "Agg MB/s",
                                                  "Max BW",
                                                  "Avg Latency",
                                                  "Min Latency",
                                                  "Max Latency");
        }
    }

    for (int k = 1; k <= count; k *= 2) {
        double bandwidth = 0, agg_bandwidth = 0;
        double max_agg = 0;
        static double total_bw = 0, min = 0;
        min = (double) INT_MAX;
        max_latency = (double) INT_MIN;
        total = 0;

        // Hardware counter data for this message size
        hw_counter_data_t size_start_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        hw_counter_data_t size_end_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        hw_counter_data_t total_size_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};

        // Initialize total counters to zero
        if (monitor_hw_counters && hw_counters_available_check.hw_counters_available) {
            for (int i = 0; i < NUM_HW_COUNTERS; i++) {
                total_size_counters.counters[i] = 0;
            }
        }
        /* alltoall */
        for (int i = 0; i < iter + SKIP; i++) {
            long * a_psync = pSync;
            double b_start, b_end;
            ucc_coll_args_t coll_args = {
                .mask      = UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER,
                .flags     = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
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

            ucc_coll_req_h req = NULL;
            status = ucc_collective_init(&coll_args, &req, ucc_team);
            if (status != UCC_OK) {
                printf("coll init failed\n");
                return -1;
            }
            // Start hardware counter measurement for this iteration
            hw_counter_data_t iter_start_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
            if (monitor_hw_counters && hw_counters_available_check.hw_counters_available) {
                read_hw_counters(&iter_start_counters);
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
                ucc_collective_finalize(req);
            }
            end = MPI_Wtime();

            // Stop hardware counter measurement for this iteration
            hw_counter_data_t iter_end_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
            if (monitor_hw_counters && hw_counters_available_check.hw_counters_available && i >= SKIP) {
                read_hw_counters(&iter_end_counters);
                // Accumulate counter differences for valid iterations
                for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                    total_size_counters.counters[j] += (iter_end_counters.counters[j] - iter_start_counters.counters[j]);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (i > SKIP) {
                double time = (end - start);
                total += time;
                if (time < min) {
                    min = time;
                } 
                if (time > max_latency) {
                    max_latency = time;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Aggregate hardware counter results across all processes using MPI
        hw_counter_data_t global_counters = {.hw_counters_available = hw_counters_available_check.hw_counters_available};
        if (monitor_hw_counters && hw_counters_available_check.hw_counters_available) {
            // Initialize global counters to zero
            for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                global_counters.counters[j] = 0;
            }
            // Sum hardware counters across all processes using MPI_Allreduce
            for (int j = 0; j < NUM_HW_COUNTERS; j++) {
                local_count = total_size_counters.counters[j];
                MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                global_counters.counters[j] = global_count;
            }
        }

        // Use MPI collective operations for statistics
        double global_min, global_max, global_total;
        MPI_Allreduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_latency, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        min_latency = global_min;
        total_time = total;  // Use local total for bandwidth (like OpenSHMEM)
        max_latency = global_max;
        double avg_time = global_total / npes;  // Global average time for latency calculation
        
        total_bw = (npes * (k * sizeof(uint64_t))) / (1024 * 1024 * min_latency);
        bandwidth = (npes * (k * sizeof(uint64_t)) * (iter - SKIP)) / (total_time);
        src_buff = bandwidth;
        
        // Aggregate bandwidth across all processes
        MPI_Allreduce(&src_buff, &agg_bandwidth, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (me == 0) {
            printf("%-10ld", k * sizeof(uint64_t));
            printf("%-10ld", k * sizeof(uint64_t) * npes);
            printf("%15.2f", (bandwidth / (1024 * 1024)) * ppn);
            printf("%13.2f", agg_bandwidth / (1024 * 1024));
            printf("%13.2f", total_bw);
            printf("%13.2f", (avg_time * 1e6) / ((iter - SKIP)));
            printf("%13.2f", min_latency * 1e6);
            printf("%13.2f", max_latency * 1e6);

            // Print hardware counter results as additional columns if monitoring is enabled
            if (monitor_hw_counters && hw_counters_available_check.hw_counters_available) {
                printf("%15lu", global_counters.counters[0]); // CNP Sent
                printf("%15lu", global_counters.counters[1]); // CNP Handled
                printf("%15lu", global_counters.counters[2]); // CNP Ignored
                printf("%20lu", global_counters.counters[3]); // ECN Marked
            }
            printf("\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Cleanup
    free(source);
    free(pSync);
    free(pSync2);
    free(pSync3);
    free(pWrk);
    free(maps);
    
    MPI_Finalize();
    return 0;
} 