/* raw_iio.c -- open uncore_iio_0 exactly like iio_pmu.c (config packing:
 * low32=event+umask, high32=ch_mask+fc_mask) and read it for a fixed
 * duration, to compare against perf stat on the same PMU.
 * Usage: raw_iio <seconds>
 * Prints the txn (0x84) and units (0x83) deltas. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <errno.h>

static int open_ev(uint32_t type, uint64_t config, uint64_t config1,
                   uint64_t config2, int cpu)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.type = type;
    /* Intel uncore IIO reads ALL fields from the low 48 bits of
     * attr.config: low 32 = cfg[0] (event+umask), high 32 = cfg[1]
     * (ch_mask+fc_mask). attr.config1 is IGNORED by the IIO PMU. */
    a.config = config | (config1 << 32);
    (void)config2;
    a.disabled = 0;
    a.inherit = 1;
    int fd = (int)syscall(SYS_perf_event_open, &a, -1, cpu, -1, 0);
    if (fd < 0) {
        printf("  open FAIL: %s\n", strerror(errno));
    }
    return fd;
}

static uint64_t rd(int fd)
{
    uint64_t v = 0;
    ssize_t n = read(fd, &v, sizeof(v));
    if (n != (ssize_t)sizeof(v)) {
        printf("  read FAIL: %s\n", strerror(errno));
        return 0;
    }
    return v;
}

int main(int argc, char **argv)
{
    int secs = (argc > 1) ? atoi(argv[1]) : 6;
    uint32_t type = 81; /* uncore_iio_0 on these nodes */
    FILE *fp = fopen("/sys/bus/event_source/devices/uncore_iio_0/type", "r");
    if (fp) { if (fscanf(fp, "%u", &type) != 1) type = 81; fclose(fp); }

    /* txn: 0x84 umask 0x4; ch_mask 0xff@36-47, fc_mask 0x7@48-50 -> cfg[1] */
    int fd_t = open_ev(type, 0x00000484, 0x00070FF0, 0, 0);
    /* units: 0x83 umask 0x4; same masks */
    int fd_u = open_ev(type, 0x00000483, 0x00070FF0, 0, 0);
    if (fd_t < 0 || fd_u < 0) { printf("open failed\n"); return 2; }

    uint64_t bt = rd(fd_t), bu = rd(fd_u);
    fprintf(stderr, "raw_iio: type=%u sleeping %ds (fd_t=%d fd_u=%d)\n",
            type, secs, fd_t, fd_u);
    sleep(secs);
    uint64_t nt = rd(fd_t), nu = rd(fd_u);
    printf("RAW  uncore_iio_0 txn(0x84) delta=%llu\n", (unsigned long long)(nt - bt));
    printf("RAW  uncore_iio_0 units(0x83) delta=%llu\n", (unsigned long long)(nu - bu));
    return 0;
}
