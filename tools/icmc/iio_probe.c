/*
 * iio_probe.c -- definitive test that the uncore IIO read counters
 * actually increment under RDMA read (device->host DMA) traffic.
 *
 * Usage (2 nodes):
 *   icmc_probe server            # node A: opens counters, runs ib_read_bw server, probes
 *   icmc_probe client <ip>       # node B: opens counters, runs ib_read_bw client, probes
 *
 * Both print per-PMU base/now/delta.  Links iio_pmu.c (same open path as
 * the benchmark).  Exit code 0 = at least one PMU counted >0 reads.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "iio_pmu.h"

int main(int argc, char **argv)
{
    const char *role = (argc > 1) ? argv[1] : "server";
    const char *peer = (argc > 2) ? argv[2] : NULL;

    int numa = 0;
    FILE *fp = fopen("/sys/class/infiniband/mlx5_0/device/numa_node", "r");
    if (fp) { if (fscanf(fp, "%d", &numa) != 1) numa = 0; fclose(fp); if (numa < 0) numa = 0; }
    printf("[probe] numa=%d role=%s peer=%s\n", numa, role, peer ? peer : "(none)");

    int n = iio_open(numa, 0);
    printf("[probe] iio_open -> %d pmus open\n", n);
    if (n <= 0) { printf("[probe] FATAL: no IIO pmus\n"); return 2; }

    /* baseline per-PMU */
    uint64_t base_t[IIO_MAX_PMUS], base_u[IIO_MAX_PMUS];
    for (int i = 0; i < n; i++) {
        iio_snapshot_t s;
        int r = iio_pmu_snapshot(i, &s);
        base_t[i] = (r == 0) ? s.rd_trans_64b : 0;
        base_u[i] = (r == 0) ? s.rd_units_4b : 0;
        printf("[probe]   pmu[%d] %s base txn=%llu units=%llu read_rc=%d\n",
               i, iio_pmu_name(i),
               (unsigned long long)base_t[i], (unsigned long long)base_u[i], r);
    }

    /* start RDMA read traffic (device -> host DMA reads) */
    if (strcmp(role, "client") == 0 && peer) {
        char c[256];
        snprintf(c, sizeof(c), "sleep 2; ib_read_bw -d mlx5_0 -F -D 20 %s  2>&1 &", peer);
        printf("[probe] launching: %s\n", c);
        system(c);
    } else {
        system("ib_read_bw -d mlx5_0 -F -D 25  2>&1 &");
        printf("[probe] launching ib_read_bw server (duration 25s)\n");
    }

    printf("[probe] traffic window: sleeping 12s\n");
    sleep(12);

    /* now per-PMU + aggregate snapshot rc */
    uint64_t tot_delta_t = 0, tot_delta_u = 0;
    for (int i = 0; i < n; i++) {
        iio_snapshot_t s;
        int r = iio_pmu_snapshot(i, &s);
        uint64_t dt = (r == 0) ? (s.rd_trans_64b - base_t[i]) : 0;
        uint64_t du = (r == 0) ? (s.rd_units_4b - base_u[i]) : 0;
        tot_delta_t += dt; tot_delta_u += du;
        printf("[probe]   pmu[%d] %s delta txn=%llu units=%llu (rc=%d)\n",
               i, iio_pmu_name(i), (unsigned long long)dt, (unsigned long long)du, r);
    }
    iio_snapshot_t agg;
    int arc = iio_snapshot(&agg);
    printf("[probe] iio_snapshot(aggregate) rc=%d  total delta txn=%llu units=%llu bytes=%llu\n",
           arc, (unsigned long long)agg.rd_trans_64b,
           (unsigned long long)agg.rd_units_4b,
           (unsigned long long)(agg.rd_units_4b * 4));
    printf("[probe] RESULT: %s\n", (tot_delta_t > 0 || tot_delta_u > 0) ? "COUNTERS COUNT" : "ALL ZERO");
    return (tot_delta_t > 0 || tot_delta_u > 0) ? 0 : 1;
}
