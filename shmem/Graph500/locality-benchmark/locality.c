#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

MPI_Datatype sharp_ucx_mpi_worker_exchange;
MPI_Datatype sharp_ucx_mpi_buffer_exchange;
#include <sharp_mdomains/api/sharp_mdomains_md.h>
#include <sharp_maps/api/sharp_maps.h>
#include <sharp_dtiers/api/sharp_dtiers.h>
#include <sharp_groups/api/sharp_groups.h>
#include <sobjects/api/sharp_objects.h>
#include <comms/comms-include.h>

#define NR_ITERATIONS   50000

int sharp_init(int argc, char ** argv) {
    int error;
    int size = 0;

    error = sharp_create_node_info();
    if(error != SHARP_OK) {
        return error;
    }

    error = sharp_comms_init(argc, argv);
    if(error != SHARP_OK) {
        printf("error on comm init\n");
        return error;
    } else {
        printf("FERROL: Comms success\n");
    }

    return 0;
}

void sharp_finalize(void) {
    sharp_comms_finalize();
    sharp_destroy_node_info();
}

int cmpfunc(const void * a, const void * b) {
    return ((*(double *)a) - (*(double *)b));
}

static inline int get_times(int nr_iterations, int dest_pe, double * agg, double * times, sharp_group_allocated_t * comm_group) {
    int i = 0;
    double start, end;
    double median;
    double variance = 0;
    int error;

    *agg = 0;
    memset(times, 0, sizeof(double) * nr_iterations);
    printf("Performing get operation between %lu and %d\n", comms_my_pe, dest_pe);
    for (i = 0; i < nr_iterations; i++) {
        uint64_t a = (uint64_t) i;
        uint64_t offset = (i % 1024) * sizeof(uint64_t);
        
        start = MPI_Wtime();
        error = sharp_comms_get(&a, sizeof(uint64_t), offset, dest_pe, comm_group->id);
        end = MPI_Wtime();
        if (error != SHARP_OK) {
            return -1;
        }
        *agg += end - start;
        times[i] = end-start;
    }

    *agg = *agg / nr_iterations; //the avg
    qsort(times, nr_iterations, sizeof(double), cmpfunc);
    median = (times[nr_iterations/2-1] + times[nr_iterations/2]) / 2;
    
    for (i = 0; i < nr_iterations; i++) {
        times[i] = times[i] - *agg;
        variance += times[i] * times[i];
    }
    variance = variance / nr_iterations;

    printf("\tget average latency: %0.10g\n", *agg);
    printf("\tget median latency: %0.10g\n", median);
    printf("\tget variance: %0.10g\n", variance);
    return 0;
}

static inline int cswap_times(int nr_iterations, int dest_pe, double * agg, double * times, sharp_group_allocated_t * comm_group) {
    int i = 0;
    double start, end;
    double median;
    double variance = 0;
    int error;

    *agg = 0;
    memset(times, 0, sizeof(double) * nr_iterations);
    printf("Performing cswap operation between %lu and %d\n", comms_my_pe, dest_pe);
    for (i = 0; i < nr_iterations; i++) {
        uint64_t a = (uint64_t) (i % 1024);
        uint64_t offset = (i % 1024) * sizeof(uint64_t);
        uint64_t exp = 0;

        sharp_comms_get(&exp, sizeof(uint64_t), offset, dest_pe, comm_group->id);

        start = MPI_Wtime();
        error = sharp_comms_atomic_cswap64(&a, exp, offset, dest_pe, comm_group->id);
        end = MPI_Wtime();
        if (error < SHARP_OK) {
            printf("\t\tcswap error: %d\n", error);
            return -1;
        }
        *agg += end - start;
        times[i] = end-start;
    }

    *agg = *agg / nr_iterations; //the avg
    printf("times[%d] + times[%d]\n", nr_iterations/2 - 1, nr_iterations/2);
    qsort(times, nr_iterations, sizeof(double), cmpfunc);
    median = (times[nr_iterations/2-1] + times[nr_iterations/2]) / 2;
    
    for (i = 0; i < nr_iterations; i++) {
        times[i] = times[i] - *agg;
        variance += times[i] * times[i];
    }
    variance = variance / nr_iterations;

    printf("\tcswap average latency: %0.10g\n", *agg);
    printf("\tcswap median latency: %0.10g\n", median);
    printf("\tcswap variance: %0.10g\n", variance);
    return 0;
}

static inline int put_times(int nr_iterations, int dest_pe, double * agg, double * times, sharp_group_allocated_t * comm_group) {
    int i = 0;
    double start, end;
    double median;
    double variance = 0;
    int error;

    *agg = 0;
    memset(times, 0, sizeof(double) * nr_iterations);
    printf("Performing put operation between %lu and %d\n", comms_my_pe, dest_pe);
    for (i = 0; i < nr_iterations; i++) {
        uint64_t a = (uint64_t) i;
        uint64_t offset = (i % 1024) * sizeof(uint64_t);
        
        start = MPI_Wtime();
        error = sharp_comms_put(&a, sizeof(uint64_t), offset, dest_pe, comm_group->id);
        sharp_comms_flush_pe(dest_pe, comm_group->id);
        end = MPI_Wtime();
        if (error != SHARP_OK) {
            return -1;
        }
        *agg += end - start;
        times[i] = end-start;
    }

    *agg = *agg / nr_iterations; //the avg
    printf("times[%d] + times[%d]\n", nr_iterations/2 - 1, nr_iterations/2);
    qsort(times, nr_iterations, sizeof(double), cmpfunc);
    median = (times[nr_iterations/2-1] + times[nr_iterations/2]) / 2;
    
    for (i = 0; i < nr_iterations; i++) {
        times[i] = times[i] - *agg;
        variance += times[i] * times[i];
    }
    variance = variance / nr_iterations;

    printf("\tput average latency: %0.10g\n", *agg);
    printf("\tput median latency: %0.10g\n", median);
    printf("\tput variance: %0.10g\n", variance);
    return 0;
} 

int main(int argc, char ** argv) {
    int error;
    size_t alloc_size = 1024 * sizeof(uint64_t);
    int am_near_nic = 0;
    double agg = 0;
    double * times;
    int * dist_pes; // describes the pes that are near (1) or far (0) from nic
    sharp_data_tier data_tiers;
    sharp_group_allocated_t comm_group;
    int node_nr[4] = {0, 0, 1, 1};

    MPI_Init(&argc, &argv);
    error = sharp_init(argc, argv);
    if (error != SHARP_OK) {
        return error;
    }

    alloc_size *= comms_size;
    dist_pes = (int *)malloc(sizeof(int) * comms_size);
    times = (double *)malloc(sizeof(double) * NR_ITERATIONS); 
    memset(dist_pes, 0, sizeof(int) * comms_size);

    sharp_create_data_tier(&data_tiers, 1,
                           SHARP_HINT_CPU, SHARP_ACCESS_INTERP | 1<<7, 1);
#ifdef RHEA
    printf("data_tiers.nr_mds: %d\n", data_tiers.nr_mds);
    if (comms_my_pe == 1 || comms_my_pe == 3) {
        data_tiers.md_ids[0] = data_tiers.md_ids[1];
    }

    printf("%lu: %d\n", comms_my_pe, data_tiers.md_ids[0]);
    if (data_tiers.md_ids[0] != 0) {
        printf("%lu is close to the nic\n", comms_my_pe);
        dist_pes[comms_my_pe] = 1;
    }
#elif LAPTOP
    // just going to simulate this since there is only one socket
    if ((comms_my_pe & 1) == 0) {
        dist_pes[comms_my_pe] = 1;
    }
#else
    printf("%lu: %d\n", comms_my_pe, data_tiers.md_ids[0]);
    if (data_tiers.md_ids[0] == 0) { /* this is always the first numa */
        printf("%lu is close to the nic\n", comms_my_pe);
        dist_pes[comms_my_pe] = 1;
    }
#endif
    
    MPI_Allreduce(MPI_IN_PLACE, dist_pes, comms_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    error = sharp_create_group_allocate(alloc_size, 
                                        SHARP_MPI, MPI_COMM_WORLD, comms_size,
                                        &data_tiers, &comm_group);
    if (error != SHARP_OK) {
        fprintf(stderr, "ERROR: Failed to allocate memory (error code = %d)\n", error);
        return error;
    }
    
    /**
        There are three different things to test here:
            1. near nic to near nic communication times (best case)
            2. far nic to near nic communication times 
            3. far nic to far nic communication times (worst case)
    */
    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
    MPI_Barrier(MPI_COMM_WORLD);
    if (dist_pes[comms_my_pe] == 1) {
        // need to find my partner
        int partner;
        for (partner = 0; partner < comms_size; partner++) {
            if (node_nr[partner] != node_nr[comms_my_pe] && dist_pes[partner] == 1 && partner != comms_my_pe) {
                if (comms_my_pe < partner) {
                    error = put_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on put (%d)\n", error);
                        return error;
                    }
                    error = get_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on get (%d)\n", error);
                        return error;
                    }
                    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
                    error = cswap_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on cswap (%d)\n", error);
                        return error;
                    }
                    fflush(stdout);
                }
                break;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
    MPI_Barrier(MPI_COMM_WORLD);
    if (dist_pes[comms_my_pe] == 0) {
        // need to find my partner
        int partner;
        for (partner = 0; partner < comms_size; partner++) {
            if (node_nr[partner] != node_nr[comms_my_pe] && dist_pes[partner] == 1 && partner != comms_my_pe) {
                if (comms_my_pe < partner) {
                    error = put_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on put (%d)\n", error);
                        return error;
                    }
                    error = get_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on get (%d)\n", error);
                        return error;
                    }
                    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
                    error = cswap_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on cswap (%d)\n", error);
                        return error;
                    }
                    fflush(stdout);
                }
                break;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
    MPI_Barrier(MPI_COMM_WORLD);
    if (dist_pes[comms_my_pe] == 0) {
        // need to find my partner
        int partner;
        for (partner = 0; partner < comms_size; partner++) {
            if (node_nr[partner] != node_nr[comms_my_pe] && dist_pes[partner] == 0 && partner != comms_my_pe) {
                if (comms_my_pe < partner) {
                    error = put_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on put (%d)\n", error);
                        return error;
                    }
                    error = get_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on get (%d)\n", error);
                        return error;
                    }
                    memset(comm_group.buffer_ptr, 0, sizeof(uint64_t) * 1024);
                    error = cswap_times(NR_ITERATIONS, 
                              partner, 
                              &agg, 
                              times,
                              &comm_group);
                    if (error < SHARP_OK) {
                        fprintf(stderr, "Failed on cswap (%d)\n", error);
                        return error;
                    }
                    fflush(stdout);
                }
                break;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    free(dist_pes);
    free(times);

    sharp_finalize();
    MPI_Finalize();
    return 0;
}
