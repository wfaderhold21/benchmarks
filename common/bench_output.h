/*
 * bench_output.h -- Header-only CSV output for UCC benchmarks (C99)
 *
 * Provides a fixed-column CSV format so results from every benchmark
 * variant can be merged into one data set.
 */

#ifndef BENCH_OUTPUT_H
#define BENCH_OUTPUT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *bench_name;     /* "ucc_a2a"        */
    const char *variant;        /* "mpi", "shrink"  */
    int         world_size, ppn;
    const char *tls;            /* from env or transport_detect()   */
} bench_meta_t;

/* ---- helpers (all inline static so this is header-only) ---- */

static inline void
bench_iso8601(char *buf, size_t len)
{
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%S%z", tm);
}

static inline void
bench_csv_header(FILE *f)
{
    fprintf(f,
            "timestamp,"
            "bench,"
            "variant,"
            "nprocs,"
            "ppn,"
            "tls,"
            "ucx_tls,"
            "msg_size,"
            "iters,"
            "avg_us,"
            "min_us,"
            "max_us,"
            "stddev_us,"
            "bw_mbps\n");
}

static inline void
bench_csv_row(FILE *f, const bench_meta_t *m,
              size_t msg_size, int iters,
              double avg_us, double min_us, double max_us,
              double stddev_us, double bw_mbps)
{
    char ts[64];
    bench_iso8601(ts, sizeof(ts));

    const char *tls  = m->tls  ? m->tls  : "";

    fprintf(f, "%s,%s,%s,%d,%d,%s,,%zu,%d,"
            "%.2f,%.2f,%.2f,%.2f,%.2f\n",
            ts,
            m->bench_name,
            m->variant,
            m->world_size,
            m->ppn,
            tls,
            msg_size, iters,
            avg_us, min_us, max_us, stddev_us, bw_mbps);
}

#endif /* BENCH_OUTPUT_H */
