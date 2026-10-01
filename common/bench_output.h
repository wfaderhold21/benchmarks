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
    const char *ucc_tls;        /* UCC TLs from transport_detect()  */
    const char *ucx_tls;        /* UCX transports from transport_detect() */
} bench_meta_t;

/* RFC-4180 quoting for free-text CSV fields (transport lists contain
 * commas). Escapes s into dst (dstsz bytes) and returns dst; if s needs
 * no quoting, returns s itself. NULL/empty renders as "". dst too small:
 * copies the prefix and closes the quote. Callers with several escaped
 * fields in one row MUST pass distinct buffers. */
static inline const char *bench_csv_escape(char *dst, size_t dstsz,
                                           const char *s)
{
    if (!s || !s[0])
        return "";
    if (!strpbrk(s, ",\"\n"))
        return s;
    size_t n = 0;
    dst[n++] = '"';
    for (const char *p = s; *p && n + 3 < dstsz; p++) {
        if (*p == '"') dst[n++] = '"';
        dst[n++] = *p;
    }
    dst[n++] = '"';
    dst[n] = '\0';
    return dst;
}

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
            "ucc_tls,"
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
    char ts[64], ucc_buf[512], ucx_buf[512];
    bench_iso8601(ts, sizeof(ts));

    const char *ucc_tls = bench_csv_escape(ucc_buf, sizeof(ucc_buf), m->ucc_tls);
    const char *ucx_tls = bench_csv_escape(ucx_buf, sizeof(ucx_buf), m->ucx_tls);

    fprintf(f, "%s,%s,%s,%d,%d,%s,%s,%zu,%d,"
            "%.2f,%.2f,%.2f,%.2f,%.2f\n",
            ts,
            m->bench_name,
            m->variant,
            m->world_size,
            m->ppn,
            ucc_tls, ucx_tls,
            msg_size, iters,
            avg_us, min_us, max_us, stddev_us, bw_mbps);
}
/* ---- extended CSV: extra key/value columns appended after the fixed set ----
 *
 * `extra_cols` / `extra_vals` are comma-separated strings of the same length
 * (same number of fields, in the same order).  A missing value is written as
 * an empty field.  Used by the ICMC benchmark for:
 *   qp_node,qp_rank,qp_types,iio_rd_txn,iio_rd_bytes,
 *   excess_rd_B_per_msg,icm_fetch_per_msg
 * Existing fixed columns are unchanged; consumers that don't know the extra
 * columns simply ignore the trailing fields.
 */

#define BENCH_EXT_COLS_ICMC \
    "qp_node,qp_rank,qp_types,iio_rd_txn,iio_rd_bytes," \
    "excess_rd_B_per_msg,icm_fetch_per_msg"

static inline void
bench_csv_header_ext(FILE *f, const char *extra_cols)
{
    fprintf(f,
            "timestamp,"
            "bench,"
            "variant,"
            "nprocs,"
            "ppn,"
            "ucc_tls,"
            "ucx_tls,"
            "msg_size,"
            "iters,"
            "avg_us,"
            "min_us,"
            "max_us,"
            "stddev_us,"
            "bw_mbps");
    if (extra_cols && extra_cols[0])
        fprintf(f, ",%s", extra_cols);
    fprintf(f, "\n");
}

static inline void
bench_csv_row_ext(FILE *f, const bench_meta_t *m,
                  size_t msg_size, int iters,
                  double avg_us, double min_us, double max_us,
                  double stddev_us, double bw_mbps,
                  const char *extra_vals)
{
    char ts[64], ucc_buf[512], ucx_buf[512];
    bench_iso8601(ts, sizeof(ts));

    const char *ucc_tls = bench_csv_escape(ucc_buf, sizeof(ucc_buf), m->ucc_tls);
    const char *ucx_tls = bench_csv_escape(ucx_buf, sizeof(ucx_buf), m->ucx_tls);

    fprintf(f, "%s,%s,%s,%d,%d,%s,%s,%zu,%d,"
            "%.2f,%.2f,%.2f,%.2f,%.2f",
            ts,
            m->bench_name,
            m->variant,
            m->world_size,
            m->ppn,
            ucc_tls, ucx_tls,
            msg_size, iters,
            avg_us, min_us, max_us, stddev_us, bw_mbps);
    if (extra_vals)
        fprintf(f, ",%s", extra_vals);
    fprintf(f, "\n");
}

#endif /* BENCH_OUTPUT_H */
