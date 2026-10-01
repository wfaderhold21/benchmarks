/*
 * qp_count.h -- RDMA QP counts (node-wide and per-process)
 *
 * Backend: `rdma -j res show qp [pid <pid>]` (iproute2 >= 6.x), found via
 * PATH or a few known install locations (see qp_count.c).  The command
 * supports only a `pid` filter — there is no `dev` filter — so device
 * selection is applied client-side against each QP object's `ifname`.  A
 * raw RDMA netlink (RDMA_NLDEV_CMD_RES_QP_GET) path was considered and
 * dropped: the target kernel's (5.14) uapi predates the res_obj/res_qp_type
 * attrs and would be version-fragile.
 *
 * The JSON parser is a pure function (qp_parse_json) so it can be unit
 * tested offline with fixture text (see tools/icmc/test_parsers.c).
 */

#ifndef QP_COUNT_H
#define QP_COUNT_H

#include <stdint.h>

#define QP_TYPE_MAX 8
#define QP_DEV_NAME_MAX 32

/* One QP type bucket, e.g. RC, UC, UD, XRC, DCI, DCT, SRQ, MRR. */
typedef struct {
    char name[8];
    uint64_t count;
} qp_type_bucket_t;

typedef struct {
    char device[QP_DEV_NAME_MAX];
    uint64_t total;
    qp_type_bucket_t types[QP_TYPE_MAX];
    int ntypes;
} qp_count_t;

/*
 * Parse one or more `rdma -j res show qp` JSON arrays into *out, counting
 * QPs of every device (NULL device filter).  Accepts a single array,
 * concatenated arrays, or newline-delimited arrays (rdma prints one array
 * per device/pid).  Returns the number of QP objects counted, -1 on
 * malformed JSON.
 */
int qp_parse_json(const char *json, qp_count_t *out);

/*
 * Same as qp_parse_json, but only QP objects whose ifname matches `dev`
 * are counted (dev == NULL counts every device).  This is how the `rdma`
 * backend applies the `-c <dev>` selection, since the command has no dev
 * filter of its own.
 */
int qp_parse_json_dev(const char *json, const char *dev, qp_count_t *out);

/*
 * Query QP counts.
 *   pid < 0 : node-wide (all processes)
 *   pid >= 0: only QPs owned by that pid
 *   device  : ifname filter applied client-side (NULL = all devices)
 * Returns 0 on success, -1 if the `rdma` tool failed or was not found.
 * *used (optional) is set to 0 on success (backend id; 0 = `rdma -j`).
 */
int qp_query(const char *device, int pid, qp_count_t *out, int *used_netlink);

/*
 * Convenience: node-wide count plus the caller's own pid count.
 * Fills node-wide into *node and per-pid into *self (self->total is the
 * QPs this process currently owns; self->device may be empty).
 */
int qp_sample(const char *device, qp_count_t *node, qp_count_t *self);

/* "RC=1 UD=2" style rendering of the type buckets, empty string if none. */
const char *qp_type_string(const qp_count_t *c);

#endif /* QP_COUNT_H */
