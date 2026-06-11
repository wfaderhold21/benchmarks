/*
 * nb_sync_stencil.c
 *
 * Benchmark: nonblocking sync overlapped with interior stencil compute on a
 * 1D ring halo exchange.  Shows the crossover point where interior work is
 * large enough to fully hide sync latency.
 *
 * Per-PE buffer layout:  [left_halo | data[1..N] | right_halo]  (N+2 elements)
 *
 * Blocking:   PUT boundaries → fence → shmem_team_sync → full stencil
 * NB sync:    PUT boundaries → fence → shmem_sync_nb
 *                           → interior stencil (no halos needed)
 *             → shmem_req_wait → boundary stencil (halos now safe)
 *
 * Usage: shmrun -np <P> ./nb_sync_stencil [iterations] [warmup]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shmem.h>
#include <shmemx.h>
#include <sys/time.h>

#define DEFAULT_ITERATIONS 200
#define DEFAULT_WARMUP     20

static long   reduce_psync[SHMEM_REDUCE_SYNC_SIZE];
static double reduce_pwrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
static double g_val, l_val;

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* Report the slowest PE — wall-clock is gated by the laggard */
static double max_across_pes(double v, int n_pes) {
    l_val = v;
    shmem_double_max_to_all(&g_val, &l_val, 1, 0, 0, n_pes, reduce_pwrk, reduce_psync);
    return g_val;
}

/* Send boundary elements into neighbors' halo slots, then ensure delivery */
static inline void halo_exchange(double *buf, int n, int left_pe, int right_pe) {
    shmem_double_put(&buf[n + 1], &buf[1], 1, left_pe);  /* leftmost  → left_pe's right halo */
    shmem_double_put(&buf[0],     &buf[n], 1, right_pe); /* rightmost → right_pe's left halo  */
    shmem_quiet();
}

/* 3-point stencil: out[i-1] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1], i in [lo,hi] */
static inline void stencil(double *out, const double *in, int lo, int hi) {
    for (int i = lo; i <= hi; i++)
        out[i - 1] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
}

static double bench_blocking(double *buf, double *work, int n,
                              int left_pe, int right_pe,
                              int iters, int warmup, int n_pes) {
    for (int i = 0; i < warmup; i++) {
        halo_exchange(buf, n, left_pe, right_pe);
        shmem_sync_all();
        stencil(work, buf, 1, n);
    }

    shmem_barrier_all();
    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        halo_exchange(buf, n, left_pe, right_pe);
        shmem_sync_all();
        stencil(work, buf, 1, n);
    }
    return max_across_pes((get_time_us() - t0) / iters, n_pes);
}

static double bench_nb_sync(double *buf, double *work, int n,
                             int left_pe, int right_pe,
                             int iters, int warmup, int n_pes) {
    for (int i = 0; i < warmup; i++) {
        halo_exchange(buf, n, left_pe, right_pe);
        shmem_sync_all();
        if (n > 2)
            stencil(work, buf, 2, n - 1); /* interior: indices 2..n-1, no halos needed */
        stencil(work, buf, 1, 1);         /* left boundary: needs buf[0]   (left halo)  */
        stencil(work, buf, n, n);         /* right boundary: needs buf[n+1] (right halo) */
    }

    shmem_barrier_all();
    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        halo_exchange(buf, n, left_pe, right_pe);
        shmem_sync_all();
        if (n > 2)
            stencil(work, buf, 2, n - 1);
        stencil(work, buf, 1, 1);
        stencil(work, buf, n, n);
    }
    return max_across_pes((get_time_us() - t0) / iters, n_pes);
}

static double bench_sync_only(int iters, int warmup, int n_pes) {
    for (int i = 0; i < warmup; i++)
        shmem_sync_all();

    shmem_barrier_all();
    double t0 = get_time_us();
    for (int i = 0; i < iters; i++)
        shmem_sync_all();
    return max_across_pes((get_time_us() - t0) / iters, n_pes);
}

int main(int argc, char **argv) {
    int iterations = DEFAULT_ITERATIONS;
    int warmup     = DEFAULT_WARMUP;

    shmem_init();
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) warmup     = atoi(argv[2]);

    int left_pe  = (my_pe - 1 + n_pes) % n_pes;
    int right_pe = (my_pe + 1) % n_pes;

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++)
        reduce_psync[i] = SHMEM_SYNC_VALUE;
    shmem_barrier_all();

    double sync_us = bench_sync_only(iterations, warmup, n_pes);

    static const int sizes[] = {
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576, 0
    };

    if (my_pe == 0) {
        printf("# nb_sync_stencil: 1D ring halo exchange + 3-point stencil\n");
        printf("# PEs: %d  iterations: %d  warmup: %d\n", n_pes, iterations, warmup);
        printf("# Sync-only latency (max across PEs): %.3f us\n#\n", sync_us);
        printf("%-10s  %-14s  %-14s  %-10s\n",
               "# N/PE", "Blocking(us)", "NB_sync(us)", "Overlap(%)");
    }

    for (int si = 0; sizes[si] != 0; si++) {
        int n = sizes[si];

        double *buf  = (double *)shmem_malloc((n + 2) * sizeof(double));
        double *work = (double *)malloc(n * sizeof(double));
        if (!buf || !work) {
            if (my_pe == 0)
                printf("# shmem_malloc failed at N=%d, stopping\n", n);
            free(work);
            shmem_free(buf);
            break;
        }

        buf[0] = 0.0;
        for (int i = 1; i <= n; i++) buf[i] = (double)(my_pe * n + i);
        buf[n + 1] = 0.0;
        memset(work, 0, n * sizeof(double));

        shmem_barrier_all();
        double t_block = bench_blocking(buf, work, n, left_pe, right_pe,
                                        iterations, warmup, n_pes);
        shmem_barrier_all();
        double t_nb    = bench_nb_sync(buf, work, n, left_pe, right_pe,
                                       iterations, warmup, n_pes);

        if (my_pe == 0) {
            double overlap = 0.0;
            if (sync_us > 0.0)
                overlap = (t_block - t_nb) / sync_us * 100.0;
            if (overlap > 100.0) overlap = 100.0;
            if (overlap <   0.0) overlap =   0.0;
            printf("%-10d  %-14.3f  %-14.3f  %-10.1f\n",
                   n, t_block, t_nb, overlap);
            fflush(stdout);
        }

        shmem_free(buf);
        free(work);
    }

    shmem_finalize();
    return 0;
}
