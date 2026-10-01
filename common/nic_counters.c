/*
 * nic_counters.c -- node-wide NIC counters (pluggable backends)
 *
 * See nic_counters.h.
 */

#define _GNU_SOURCE
#include "nic_counters.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iio_pmu.h"

struct nic_counters {
    char device[64];
    uint32_t wanted;
    uint32_t active;
    char sysfs_base[288];       /* /sys/class/infiniband/<dev>/ports/1/hw_counters/ */
    char iio_pmu[32];           /* dominant IIO PMU name */
    int iio_started;            /* iio_open() was called */
};

/* ------------------------------------------------------------------ */
/* sysfs helpers                                                       */
/* ------------------------------------------------------------------ */

#define NC_SYSCNP_COUNT 4

static const char *nc_syscnp_files[NC_SYSCNP_COUNT] = {
    "np_cnp_sent",
    "rp_cnp_handled",
    "rp_cnp_ignored",
    "np_ecn_marked_roce_packets",
};

static int read_syscnp(const nic_counters_t *nc, uint64_t *v, int idx)
{
    char path[512];
    FILE *fp;
    snprintf(path, sizeof(path), "%s%s", nc->sysfs_base,
             nc_syscnp_files[idx]);
    fp = fopen(path, "r");
    if (!fp) return -1;
    if (fscanf(fp, "%lu", (unsigned long *)&v[idx]) != 1) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int sysfs_check(const char *base)
{
    for (int i = 0; i < NC_SYSCNP_COUNT; i++) {
        char path[512];
        FILE *fp;
        snprintf(path, sizeof(path), "%s%s", base, nc_syscnp_files[i]);
        fp = fopen(path, "r");
        if (!fp) return 0;
        fclose(fp);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* IIO helpers                                                         */
/* ------------------------------------------------------------------ */

static int nic_numa_node(const char *device)
{
    char path[384];
    FILE *fp;
    int node = -1;

    snprintf(path, sizeof(path),
             "/sys/class/infiniband/%s/device/numa_node", device);
    fp = fopen(path, "r");
    if (!fp) return -1;
    if (fscanf(fp, "%d", &node) != 1) node = -1;
    fclose(fp);
    if (node < 0) node = 0;   /* single-socket systems report -1 */
    return node;
}

static int first_cpu_on_node(int numa_node)
{
    char path[256], buf[1024];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist",
             numa_node);
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    fclose(fp);
    char *tok = strtok(buf, ",");
    if (!tok) return 0;
    return (int)strtol(tok, NULL, 10);   /* range low bound */
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

nic_counters_t *nic_counters_create(const char *device, const char *backends)
{
    nic_counters_t *nc = calloc(1, sizeof(*nc));
    if (!nc) return NULL;
    if (device)
        snprintf(nc->device, sizeof(nc->device), "%s", device);

    const char *p = backends ? backends : "sysfs";
    while (p && *p) {
        while (*p == ',' || *p == ' ') p++;
        const char *start = p;
        while (*p && *p != ',') p++;
        size_t len = (size_t)(p - start);
        if (len == 0) break;
        if (len == 5 && strncmp(start, "sysfs", 5) == 0)
            nc->wanted |= NC_MASK_SYSCNP;
        else if (len == 3 && strncmp(start, "iio", 3) == 0)
            nc->wanted |= NC_MASK_IIO;
    }

    if (nc->wanted & NC_MASK_SYSCNP) {
        snprintf(nc->sysfs_base, sizeof(nc->sysfs_base),
                 "/sys/class/infiniband/%s/ports/1/hw_counters/",
                 nc->device[0] ? nc->device : "mlx5_0");
        if (sysfs_check(nc->sysfs_base))
            nc->active |= NC_MASK_SYSCNP;
    }

    if (nc->wanted & NC_MASK_IIO) {
        int numa = nic_numa_node(nc->device);
        int cpu = first_cpu_on_node(numa);
        int n = iio_open(numa, cpu);
        if (n > 0) {
            nc->active |= NC_MASK_IIO;
            nc->iio_started = 1;
            int dom = iio_dominant_pmu();
            if (dom >= 0)
                snprintf(nc->iio_pmu, sizeof(nc->iio_pmu), "%s",
                         iio_pmu_name(dom));
        }
    }

    if (nc->active == 0) {
        if (nc->iio_started) iio_close();
        free(nc);
        return NULL;
    }
    return nc;
}

void nic_counters_destroy(nic_counters_t *nc)
{
    if (!nc) return;
    if (nc->iio_started) iio_close();
    free(nc);
}

uint32_t nic_counters_backends(const nic_counters_t *nc)
{
    return nc->active;
}

void nic_counters_snapshot(const nic_counters_t *nc, nc_snapshot_t *out)
{
    memset(out, 0, sizeof(*out));

    if (nc->active & NC_MASK_SYSCNP) {
        uint64_t v[NC_SYSCNP_COUNT];
        int ok = 1;
        for (int i = 0; i < NC_SYSCNP_COUNT; i++)
            if (read_syscnp(nc, v, i) != 0) { ok = 0; break; }
        if (ok) {
            out->np_cnp_sent    = v[0];
            out->rp_cnp_handled = v[1];
            out->rp_cnp_ignored = v[2];
            out->np_ecn_marked  = v[3];
            out->valid |= NC_MASK_SYSCNP;
        }
    }

    if (nc->active & NC_MASK_IIO) {
        iio_snapshot_t s;
        if (iio_snapshot(&s) == 0) {
            out->iio_rd_txn_64b  = s.rd_trans_64b;
            out->iio_rd_units_4b = s.rd_units_4b;
            out->valid |= NC_MASK_IIO;
        }
    }
}

const char *nic_counters_iio_pmu(const nic_counters_t *nc)
{
    return nc->iio_pmu[0] ? nc->iio_pmu : NULL;
}
