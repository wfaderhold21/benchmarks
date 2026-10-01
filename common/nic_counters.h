/*
 * nic_counters.h -- node-wide NIC counters (pluggable backends)
 *
 * Backends:
 *   sysfs : /sys/class/infiniband/<dev>/ports/1/hw_counters/{np_cnp_sent,
 *           rp_cnp_handled, rp_cnp_ignored,np_ecn_marked_roce_packets}
 *   iio   : uncore IIO PMU NIC read-transaction counters (see iio_pmu.h;
 *           an ESTIMATE of ICMC misses, not a real miss counter)
 *
 * The caller (the per-node leader in an MPI job) reads one snapshot per
 * measurement window; deltas are computed between snapshots.  Node-wide
 * semantics: exactly ONE process per node owns a nic_counters_t.
 */

#ifndef NIC_COUNTERS_H
#define NIC_COUNTERS_H

#include <stdint.h>

#define NC_MASK_SYSCNP 0x1
#define NC_MASK_IIO    0x2

typedef struct {
    uint64_t np_cnp_sent;
    uint64_t rp_cnp_handled;
    uint64_t rp_cnp_ignored;
    uint64_t np_ecn_marked;
    uint64_t iio_rd_txn_64b;   /* 64 B read transactions (IIO) */
    uint64_t iio_rd_units_4b;  /* 4 B read units (IIO) */
    uint32_t valid;            /* NC_MASK bits for fields read OK */
} nc_snapshot_t;

typedef struct nic_counters nic_counters_t;

/*
 * Open backends named in `backends` ("sysfs,iio"; unknown names ignored).
 * device: RDMA device name (e.g. "mlx5_0"); the sysfs backend uses
 * /sys/class/infiniband/<device>/..., the iio backend opens PMUs on the
 * NIC's NUMA socket.  Returns NULL if no requested backend is available
 * (benchmark must then print "n/a" for the corresponding columns).
 */
nic_counters_t *nic_counters_create(const char *device, const char *backends);
void nic_counters_destroy(nic_counters_t *nc);

/* Which backends actually opened (NC_MASK bits). */
uint32_t nic_counters_backends(const nic_counters_t *nc);

/* Read all open counters.  *out->valid indicates which groups read OK. */
void nic_counters_snapshot(const nic_counters_t *nc, nc_snapshot_t *out);

/* d = b - a (unsigned; assumes monotonic counters). */
static inline void nic_counters_diff(const nc_snapshot_t *a,
                                     const nc_snapshot_t *b,
                                     nc_snapshot_t *d)
{
    d->np_cnp_sent    = b->np_cnp_sent    - a->np_cnp_sent;
    d->rp_cnp_handled = b->rp_cnp_handled - a->rp_cnp_handled;
    d->rp_cnp_ignored = b->rp_cnp_ignored - a->rp_cnp_ignored;
    d->np_ecn_marked  = b->np_ecn_marked  - a->np_ecn_marked;
    d->iio_rd_txn_64b = b->iio_rd_txn_64b - a->iio_rd_txn_64b;
    d->iio_rd_units_4b = b->iio_rd_units_4b - a->iio_rd_units_4b;
    d->valid = b->valid & a->valid;
}

/* Name of the dominant IIO PMU (the one the NIC traffic lands on). */
const char *nic_counters_iio_pmu(const nic_counters_t *nc);

#endif /* NIC_COUNTERS_H */
