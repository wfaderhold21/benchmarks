/*
 * iio_pmu.h -- Intel uncore IIO PMU counters via perf_event_open
 *
 * Measures all non-payload NIC read traffic (QPC / MKey / MTT ICM cache
 * misses plus any other DMA reads) on the node.  This is an ESTIMATE of
 * ICMC misses: it cannot distinguish QPC fetches from other NIC reads.
 *
 * PMU<->NIC mapping is not exposed in sysfs on the target kernels, so the
 * library opens every uncore_iio_N PMU reachable from the requested NUMA
 * socket (wrong-socket PMUs fail to open and are skipped), sums them in
 * snapshots, and the caller can identify the dominant PMU.
 *
 * Available only where unprivileged perf_event_open on uncore works
 * (e.g. Intel SPR/EMR with perf_event_paranoid < 1).  On other machines
 * (e.g. NVIDIA Thor / Broadwell) iio_open() returns 0 opened PMUs and the
 * benchmark must print "n/a" for these columns.
 *
 */
#ifndef IIO_PMU_H
#define IIO_PMU_H

#include <stdint.h>

/* Intel IIO PMU event codes (x86). */
#define IIO_EVT_READ_TRANSACTIONS  0x84   /* 64 B read transactions */
#define IIO_EVT_READ_UNITS         0x83   /* 4 B read units */
#define IIO_UMASK_64B_READ         0x4    /* for event 0x84 */
#define IIO_UMASK_4B_READ          0x4    /* for event 0x83 */
#define IIO_CH_MASK_ALL            0xff   /* channels 0..7 */
#define IIO_FC_MASK_ALL            0x7    /* function complexes */

#define IIO_MAX_PMUS      64
#define IIO_PMU_NAME_LEN  24

/*
 * One field from /sys/bus/event_source/devices/uncore_iio_N/format/<name>.
 * `lsb`/`width` are the absolute bit range lo..hi in the PMU's virtual
 * config space (3x32-bit words: config, config1, config2); the word is
 * derived from the position, so a field may cross a word boundary.
 */
typedef struct {
    char name[16];   /* "event", "umask", "ch_mask", "fc_mask", ... */
    uint32_t lsb;
    uint32_t width;
} iio_field_spec_t;
/*
 * Parse one PMU format file.  fname = field name (the file's basename, e.g.
 * "ch_mask"); content = the file body, "<word>:<lo>-<hi>" where word is
 * config/config1/config2 (absolute bits W*32+lo..W*32+hi).  Returns 1 on
 * success, 0 on parse error.  Pure: unit-testable without sysfs.
 */
int iio_parse_format(const char *fname, const char *content, iio_field_spec_t *sp);
/*
 * Pure encoder: place `values[i]` at absolute bit range
 * specs[i].lsb .. specs[i].lsb+specs[i].width-1 of the PMU's virtual
 * config space (3x32-bit words: config, config1, config2), selecting the
 * word per bit (fields may cross word boundaries).  Returns 0 on success,
 * -1 if any field overflows its width or lands outside the 96-bit space.
 * Unit-testable without any sysfs access.
 */
int iio_encode(const iio_field_spec_t *specs, const uint32_t *values, int nfields,
               uint32_t out_config[3]);

/*
 * Open every uncore_iio_N PMU that accepts a counter on `cpu` (a CPU of
 * `numa_node`).  Each opened PMU gets two fds: 64 B read transactions
 * (event 0x84, umask 0x4) and 4 B read units (event 0x83, umask 0x4),
 * system-wide (pid=-1).  Returns the number of PMUs opened (0 = unavailable,
 * -1 = fatal).  Safe to call twice: previous state is closed first.
 */
int iio_open(int numa_node, int cpu);
void iio_close(void);

int  iio_pmu_count(void);                /* PMUs currently open */
int  iio_pmu_cpu(void);                  /* CPU the counters were opened on */
const char *iio_pmu_name(int idx);       /* "uncore_iio_N" or NULL */

/* Counter values summed over all open PMUs. */
typedef struct {
    uint64_t rd_trans_64b;   /* 64 B read transactions */
    uint64_t rd_units_4b;    /* 4 B read units; bytes = rd_units_4b * 4 */
} iio_snapshot_t;

/* Read all open counters into *s.  0 on success, -1 if none open/read err. */
int iio_snapshot(iio_snapshot_t *s);

/* d = b - a (unsigned). */
void iio_diff(const iio_snapshot_t *a, const iio_snapshot_t *b, iio_snapshot_t *d);

/* Approximate bytes of NIC read traffic in a delta. */
static inline uint64_t iio_read_bytes(const iio_snapshot_t *d)
{
    return d->rd_units_4b * 4u;
}

/*
 * Per-PMU breakdown of the most recent iio_snapshot(); idx in [0, count).
 * Used to identify the dominant PMU during warm-up.
 */
int iio_pmu_snapshot(int idx, iio_snapshot_t *s);
/* Dominant = PMU with the largest rd_trans_64b delta since last call; -1 none. */
int iio_dominant_pmu(void);

#endif /* IIO_PMU_H */
