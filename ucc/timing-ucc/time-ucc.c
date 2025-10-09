/*
 *  This benchmark measures context and team creation time for UCC. 
 *
 *  Meant to be used with MPI
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
#include <link.h>

#include <ucc/api/ucc.h>

#define NR_ITER     10
#define SKIP        1
#define MAX_TLS     32

// Structure to collect loaded UCC TLs
typedef struct {
    char tl_names[MAX_TLS][64];
    int count;
} ucc_tl_list_t;

// Callback for dl_iterate_phdr to find loaded UCC TLs
static int find_ucc_tls_callback(struct dl_phdr_info *info, size_t size, void *data) {
    ucc_tl_list_t *tl_list = (ucc_tl_list_t *)data;
    const char *name = info->dlpi_name;
    
    if (name && strlen(name) > 0) {
        // Look for libucc_tl_*.so pattern
        const char *basename = strrchr(name, '/');
        basename = basename ? basename + 1 : name;
        
        if (strncmp(basename, "libucc_tl_", 10) == 0) {
            // Extract TL name between "libucc_tl_" and ".so"
            const char *tl_start = basename + 10;
            const char *tl_end = strstr(tl_start, ".so");
            
            if (tl_end && tl_list->count < MAX_TLS) {
                size_t len = tl_end - tl_start;
                if (len > 0 && len < 63) {
                    strncpy(tl_list->tl_names[tl_list->count], tl_start, len);
                    tl_list->tl_names[tl_list->count][len] = '\0';
                    tl_list->count++;
                }
            }
        }
    }
    return 0;
}

// Get list of loaded UCC TLs
static void get_loaded_ucc_tls(ucc_tl_list_t *tl_list) {
    tl_list->count = 0;
    dl_iterate_phdr(find_ucc_tls_callback, tl_list);
}

// Callback for dl_iterate_phdr to find loaded UCX transports
static int find_ucx_tls_callback(struct dl_phdr_info *info, size_t size, void *data) {
    ucc_tl_list_t *tl_list = (ucc_tl_list_t *)data;
    const char *name = info->dlpi_name;
    
    if (name && strlen(name) > 0) {
        // Look for libuct_*.so pattern (UCX transport libraries)
        const char *basename = strrchr(name, '/');
        basename = basename ? basename + 1 : name;
        
        if (strncmp(basename, "libuct_", 7) == 0) {
            // Extract transport name between "libuct_" and ".so"
            const char *tl_start = basename + 7;
            const char *tl_end = strstr(tl_start, ".so");
            
            if (tl_end && tl_list->count < MAX_TLS) {
                size_t len = tl_end - tl_start;
                if (len > 0 && len < 63) {
                    strncpy(tl_list->tl_names[tl_list->count], tl_start, len);
                    tl_list->tl_names[tl_list->count][len] = '\0';
                    tl_list->count++;
                }
            }
        }
    }
    return 0;
}

// Get list of loaded UCX transports
static void get_loaded_ucx_transports(ucc_tl_list_t *tl_list) {
    tl_list->count = 0;
    dl_iterate_phdr(find_ucx_tls_callback, tl_list);
}

void print_statistics(const char *label, double *times, int count, int rank) {
    if (rank != 0) return;
    
    double sum = 0.0, min = times[0], max = times[0];
    for (int i = 0; i < count; i++) {
        sum += times[i];
        if (times[i] < min) min = times[i];
        if (times[i] > max) max = times[i];
    }
    double avg = sum / count;
    
    // Calculate variance and standard deviation
    double variance = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - avg;
        variance += diff * diff;
    }
    variance /= count;
    double stddev = sqrt(variance);
    
    printf("\n=== %s Statistics ===\n", label);
    printf("  Iterations:  %d\n", count);
    printf("  Total:       %.2f ms\n", sum * 1000);
    printf("  Average:     %.2f ms\n", avg * 1000);
    printf("  Min:         %.2f ms\n", min * 1000);
    printf("  Max:         %.2f ms\n", max * 1000);
    printf("  Std Dev:     %.2f ms\n", stddev * 1000);
    printf("  Variance:    %.2f ms²\n", variance * 1000000);
}

void print_ucc_info(ucc_lib_h lib, ucc_context_h ctx, int rank) {
    if (rank != 0) return;
    
    printf("\n=== UCC Configuration ===\n");
    
    // Print UCC library attributes
    ucc_lib_attr_t lib_attr;
    lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
    if (UCC_OK == ucc_lib_get_attr(lib, &lib_attr)) {
        printf("  Thread mode: %d\n", lib_attr.thread_mode);
    }
    
    // Query UCC context attributes to get TL information
    ucc_context_attr_t ctx_attr;
    ctx_attr.mask = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR | 
                    UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
                    UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE;
    
    if (UCC_OK == ucc_context_get_attr(ctx, &ctx_attr)) {
        printf("  Context Attributes:\n");
        if (ctx_attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
            printf("    Context address length: %zu bytes\n", ctx_attr.ctx_addr_len);
        }
        if (ctx_attr.mask & UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE) {
            printf("    Work buffer size: %zu bytes\n", ctx_attr.global_work_buffer_size);
        }
    }
    
    // Get actually loaded TLs by examining loaded shared libraries
    printf("\n  Loaded Transport Layers (via dl_iterate_phdr):\n");
    
    ucc_tl_list_t loaded_tls;
    get_loaded_ucc_tls(&loaded_tls);
    
    if (loaded_tls.count > 0) {
        printf("    Found %d loaded UCC TL%s:\n", loaded_tls.count, 
               loaded_tls.count > 1 ? "s" : "");
        for (int i = 0; i < loaded_tls.count; i++) {
            printf("      - %s", loaded_tls.tl_names[i]);
            
            // Add descriptions for known TLs
            if (strcmp(loaded_tls.tl_names[i], "ucp") == 0) {
                printf(" (UCX Protocol - general purpose)");
            } else if (strcmp(loaded_tls.tl_names[i], "nccl") == 0) {
                printf(" (NVIDIA NCCL - GPU collectives)");
            } else if (strcmp(loaded_tls.tl_names[i], "sharp") == 0) {
                printf(" (Mellanox SHARP - in-network aggregation)");
            } else if (strcmp(loaded_tls.tl_names[i], "self") == 0) {
                printf(" (Self/loopback)");
            } else if (strcmp(loaded_tls.tl_names[i], "shm") == 0) {
                printf(" (Shared memory)");
            } else if (strcmp(loaded_tls.tl_names[i], "cuda") == 0) {
                printf(" (CUDA-aware collectives)");
            } else if (strcmp(loaded_tls.tl_names[i], "rocm") == 0) {
                printf(" (ROCm/HIP-aware collectives)");
            }
            printf("\n");
        }
    } else {
        printf("    No UCC TL libraries detected (they may load later)\n");
    }
    
    // Show what was requested vs what's loaded
    char *requested_tls = getenv("UCC_TLS");
    if (requested_tls) {
        printf("    Requested via UCC_TLS: %s\n", requested_tls);
    } else {
        printf("    UCC_TLS not set (using all available TLs)\n");
    }
    
    // Print environment variables related to transports
    printf("\n  Transport Environment Variables:\n");
    char *env_vars[] = {
        "UCC_TLS",
        "UCC_CLS",
        "UCC_TL_UCP_TUNE",
        "UCC_CL_BASIC_TUNE",
        "UCX_TLS",
        "UCX_NET_DEVICES",
        "NCCL_IB_DISABLE",
        "NCCL_NET_GDR_LEVEL",
        NULL
    };
    
    int found_count = 0;
    for (int i = 0; env_vars[i] != NULL; i++) {
        char *val = getenv(env_vars[i]);
        if (val) {
            printf("    %s = %s\n", env_vars[i], val);
            found_count++;
        }
    }
    
    if (found_count == 0) {
        printf("    (No transport environment variables set - using UCC/UCX defaults)\n");
    }
    
    // Print UCX transport information (loaded libuct_*.so libraries)
    printf("\n  Loaded UCX Transports (via dl_iterate_phdr):\n");
    
    ucc_tl_list_t ucx_transports;
    get_loaded_ucx_transports(&ucx_transports);
    
    if (ucx_transports.count > 0) {
        printf("    Found %d loaded UCX transport%s:\n", ucx_transports.count,
               ucx_transports.count > 1 ? "s" : "");
        for (int i = 0; i < ucx_transports.count; i++) {
            printf("      - %s", ucx_transports.tl_names[i]);
            
            // Add descriptions for known UCX transports
            if (strstr(ucx_transports.tl_names[i], "ib") || 
                strstr(ucx_transports.tl_names[i], "mlx")) {
                printf(" (InfiniBand/Mellanox)");
            } else if (strstr(ucx_transports.tl_names[i], "tcp")) {
                printf(" (TCP/IP sockets)");
            } else if (strstr(ucx_transports.tl_names[i], "sm") || 
                       strstr(ucx_transports.tl_names[i], "shm")) {
                printf(" (Shared Memory)");
            } else if (strstr(ucx_transports.tl_names[i], "self")) {
                printf(" (Loopback)");
            } else if (strstr(ucx_transports.tl_names[i], "cuda")) {
                printf(" (NVIDIA CUDA)");
            } else if (strstr(ucx_transports.tl_names[i], "rocm") ||
                       strstr(ucx_transports.tl_names[i], "gdr")) {
                printf(" (GPU Direct)");
            }
            printf("\n");
        }
    } else {
        printf("    No UCX transport libraries detected\n");
    }
    
    // Show UCX_TLS configuration if set
    char *ucx_tls = getenv("UCX_TLS");
    if (ucx_tls) {
        printf("    Requested via UCX_TLS: %s\n", ucx_tls);
    } else {
        printf("    UCX_TLS not set (auto-detection enabled)\n");
        printf("    Tip: Set UCX_LOG_LEVEL=INFO to see transport selection details\n");
    }
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)(uintptr_t)request;
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

int main(int argc, char ** argv)
{
    int me;
    int npes;
//    int count = 4;
    int count = 32768*2;
    //int count = 262144 * 4;    
    long * pSync;
    long * pSync2;
    long * pSync3;
    double * pWrk;
    double start, end;
    double *ctx_times = NULL;
    double *team_times = NULL;
    int ctx_time_count = 0;
    int team_time_count = 0;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_context_h ucc_context[NR_ITER + 1];
    ucc_mem_map_t *maps = NULL;
    ucc_team_h ucc_team[NR_ITER];
    ucc_team_params_t team_params;
    ucc_status_t status;
    ucc_lib_h ucc_lib;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    // Allocate timing arrays
    ctx_times = (double *) malloc(sizeof(double) * NR_ITER);
    team_times = (double *) malloc(sizeof(double) * NR_ITER);
    if (!ctx_times || !team_times) {
        printf("Failed to allocate timing arrays\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (me == 0) {
        printf("========================================\n");
        printf("UCC Context/Team Creation Benchmark\n");
        printf("========================================\n");
        printf("Processes:   %d\n", npes);
        printf("Iterations:  %d\n", NR_ITER);
        printf("Skip:        %d\n", SKIP);
        printf("========================================\n");
    }

    int64_t* source = (int64_t *) malloc(1024 * 1024 * 1024);

    pSync = (long *) malloc(sizeof(long) * (5));
    pSync2 = (long *) malloc(sizeof(long) * (5));
    pSync3 = (long *) malloc(sizeof(long) * 256);
    pWrk = (double *) malloc(sizeof(double) * 128);

    for (int i = 0; i < 5; i++) {
        pSync[i] = 0;
        pSync2[i] = 0;
    }
    for (int i = 0; i < 256; i++) {
        pSync3[i] = 0;
    }

    maps = (ucc_mem_map_t *)malloc(sizeof(ucc_mem_map_t) * 6);
    if (maps == NULL) {
        printf("OOM\n");
        return -1;
    }

    maps[0].address = source;
    maps[0].len = 2 * npes * count * sizeof(int64_t);
    maps[1].address = pSync;
    maps[1].len = 5 * sizeof(long);
    maps[2].address = pSync2;
    maps[2].len = 5 * sizeof(long);
    maps[3].address = pSync3;
    maps[3].len = 256 * sizeof(long);
    maps[4].address = pWrk;
    maps[4].len = 128 * sizeof(double);

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = npes;
    ctx_params.oob.oob_ep = me;
    ctx_params.mem_params.segments = maps;
    ctx_params.mem_params.n_segments = 5;

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

    if (me == 0) {
        printf("\n>>> Starting Context Creation Benchmark <<<\n");
    }
    
    for (int i = 0; i < NR_ITER; i++) {
        if (i >= SKIP) { 
            start = MPI_Wtime();
            if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context[i])) {
                printf("error on ctx create\n");
                return -1;
            }
            end = MPI_Wtime();
            ctx_times[ctx_time_count++] = end - start;
            MPI_Barrier(MPI_COMM_WORLD);
            if (me == 0) {
                printf("  Iteration %2d: %6.2f ms\n", i, (end - start) * 1000);
            }
//            ucc_context_destroy(ucc_context[i]);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    if (UCC_OK != ucc_context_create(ucc_lib, &ctx_params, ctx_config, &ucc_context[NR_ITER])) {
        printf("error on ctx create\n");
        return -1;
    }

    print_statistics("Context Creation", ctx_times, ctx_time_count, me);

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

    if (me == 0) {
        printf("\n>>> Starting Team Creation Benchmark <<<\n");
    }
    
    for (int i = 0; i < NR_ITER; i++) {
        if (i >= SKIP) {
            start = MPI_Wtime();
            if (UCC_OK != ucc_team_create_post(&ucc_context[NR_ITER], 1, &team_params, &ucc_team[i])) {
                printf("team create post failed\n");
                return -1; 
            }   
            
            status = ucc_team_create_test(ucc_team[i]);
            while (UCC_INPROGRESS == status) {
                status = ucc_team_create_test(ucc_team[i]);
            }
            if (UCC_OK != status) {
                printf("team create failed\n");
                return -1; 
            }
            end = MPI_Wtime();
            team_times[team_time_count++] = end - start;
            MPI_Barrier(MPI_COMM_WORLD);
            if (me == 0) {
                printf("  Iteration %2d: %6.2f ms\n", i, (end - start) * 1000);
            }

        }
        //ucc_team_destroy(ucc_team[i]);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    print_statistics("Team Creation", team_times, team_time_count, me);
    
    // Print UCC configuration info
    print_ucc_info(ucc_lib, ucc_context[NR_ITER], me);
    
    if (me == 0) {
        printf("\n========================================\n");
        printf("Benchmark completed successfully\n");
        printf("========================================\n");
    }
    
    // Cleanup
    free(ctx_times);
    free(team_times);
    free(source);
    free(pSync);
    free(pSync2);
    free(pSync3);
    free(pWrk);
    free(maps);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
