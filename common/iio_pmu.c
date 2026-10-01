/*
 * iio_pmu.c -- Intel uncore IIO PMU counters via perf_event_open
 *
 * See iio_pmu.h for the model: system-wide counters on every reachable
 * uncore_iio_N PMU, summed over PMUs in snapshots.
 */

#define _GNU_SOURCE
#include "iio_pmu.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif /* __linux__ */

#define IIO_MAX_EVENTS_PER_PMU 2
#define ARRAYSIZE(a) ((int)(sizeof(a) / sizeof((a)[0])))

struct iio_pmu {
    char name[IIO_PMU_NAME_LEN];
    uint32_t type;
    int fd_txn;   /* event 0x84 umask 0x4 */
    int fd_unit;  /* event 0x83 umask 0x4 */
    /* fields of this PMU's format directory (order differs per kernel) */
    iio_field_spec_t specs[8];
    int nspecs;
};

static struct iio_pmu g_pmus[IIO_MAX_PMUS];
static int g_npmus = 0;
static int g_cpu = -1;
static int g_numa_node = -1;

/* per-PMU snapshot taken at open time, for iio_dominant_pmu() deltas */
static iio_snapshot_t g_prev_pmu[IIO_MAX_PMUS];
int iio_open(int numa_node, int cpu);
void iio_close(void);
int iio_pmu_count(void) { return g_npmus; }
int iio_pmu_cpu(void) { return g_cpu; }

const char *iio_pmu_name(int idx)
{
    if (idx < 0 || idx >= g_npmus) return NULL;
    return g_pmus[idx].name;
}

/* ------------------------------------------------------------------ */
/* Pure encoder (unit-tested offline)                                  */
/*                                                                     */
/* Fields are bit ranges in the PMU's virtual config space, which is   */
/* emitted as 3x32-bit words (config, config1, config2).  A field MAY  */
/* cross a word boundary (e.g. uncore_iio ch_mask: config:36-47) -     */
/* bits are placed per-bit, which is exactly what the kernel's         */
/* uncore format files describe.                                       */
/* ------------------------------------------------------------------ */

int iio_encode(const iio_field_spec_t *specs, const uint32_t *values, int nfields,
               uint32_t out_config[3])
{
    if (!specs || !values || nfields <= 0 || !out_config) return -1;
    memset(out_config, 0, sizeof(uint32_t) * 3);
    for (int i = 0; i < nfields; i++) {
        const iio_field_spec_t *sp = &specs[i];
        if (sp->lsb + sp->width > 96) return -1;   /* beyond config2 */
        if (sp->name[0] == '\0') continue;
        uint32_t mask = (sp->width >= 32) ? 0xffffffffu
                        : ((1u << sp->width) - 1u);
        if ((values[i] & ~mask) != 0u) return -1;   /* field overflow */
        for (uint32_t b = 0; b < sp->width; b++) {
            uint32_t bit = (values[i] >> b) & 1u;
            if (!bit) continue;
            uint32_t pos = sp->lsb + b;
            uint32_t word = pos / 32;
            if (word > 2) return -1;
            out_config[word] |= 1u << (pos % 32);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* format-file parsing                                                  */
/* ------------------------------------------------------------------ */

/*
 * Parse one /sys/bus/event_source/devices/uncore_iio_N/format/<name> file.
 *
 * The field NAME is the file's basename (e.g. "ch_mask"); the file
 * content is "<word>:<lo>-<hi>" where <word> is config / config1 /
 * config2, giving absolute bits W*32+lo .. W*32+hi (a field may cross
 * a 32-bit word boundary, e.g. ch_mask: config:36-47).  Optional
 * whitespace around the colon is tolerated.  Returns 1 on success and
 * fills *sp (name from fname), 0 on parse error.  Pure: unit-testable
 * without sysfs.
 */
int iio_parse_format(const char *fname, const char *content, iio_field_spec_t *sp)
{
    static const char *wordmap[3] = { "config", "config1", "config2" };
    if (!fname || !fname[0] || !content || !sp) return 0;

    const char *p = content;
    while (*p == ' ' || *p == '\t') p++;
    const char *colon = strchr(p, ':');
    if (!colon) return 0;

    /* word = 0/1/2 from the text before the colon (config/config1/config2);
     * reject anything that is not a known word name. */
    int word = -1;
    {
        const char *we = colon;
        while (we > p && (we[-1] == ' ' || we[-1] == '\t')) we--;
        size_t wl = (size_t)(we - p);
        for (int i = 0; i < 3; i++)
            if (wl == strlen(wordmap[i]) && strncmp(p, wordmap[i], wl) == 0) {
                word = i;
                break;
            }
        if (word < 0) return 0;
    }

    const char *bits = colon + 1;
    while (*bits == ' ' || *bits == '\t') bits++;
    const char *dash = strchr(bits, '-');
    int lo, hi;
    if (dash) {
        if (sscanf(bits, "%d", &lo) != 1) return 0;
        if (sscanf(dash + 1, "%d", &hi) != 1 || hi < lo) return 0;
    } else {
        /* single-bit field, e.g. "config:18" */
        char *end = NULL;
        long v = strtol(bits, &end, 10);
        while (end && *end == ' ') end++;
        if (end == bits || *end != '\0') return 0;
        lo = hi = (int)v;
        if (lo < 0) return 0;
    }
    if (word * 32 + hi >= 96) return 0;   /* must fit in 3x32-bit config */

    size_t nl = strlen(fname);
    if (nl >= sizeof(sp->name)) nl = sizeof(sp->name) - 1;
    memcpy(sp->name, fname, nl);
    sp->name[nl] = '\0';
    sp->lsb = (uint32_t)(word * 32 + lo);
    sp->width = (uint32_t)(hi - lo + 1);
    return 1;
}

static int field_is_essential(const char *name)
{
    return strcmp(name, "event") == 0 || strcmp(name, "umask") == 0;
}

static int load_pmu_format(struct iio_pmu *pmu, const char *devdir)
{
    char fpath[PATH_MAX], path[PATH_MAX + 64], line[128];
    struct dirent *de;
    int n = 0, bad = 0;

    snprintf(fpath, sizeof(fpath), "%s/format", devdir);
    DIR *d = opendir(fpath);
    if (!d) return 0;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "%s/%s", fpath, de->d_name);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;
        if (!fgets(line, sizeof(line), fp)) { fclose(fp); bad = 1; continue; }
        fclose(fp);
        line[strcspn(line, "\r\n")] = '\0';
        if (n < (int)ARRAYSIZE(pmu->specs) &&
            iio_parse_format(de->d_name, line, &pmu->specs[n])) {
            n++;
        } else if (field_is_essential(de->d_name)) {
            bad = 1;   /* essential field present but unparseable */
        }
    }
    closedir(d);
    pmu->nspecs = n;
    return bad ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* discovery                                                           */
/* ------------------------------------------------------------------ */

static int read_uint(const char *path, uint64_t *v)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    if (fscanf(fp, "%lu", (unsigned long *)v) != 1) { fclose(fp); return 0; }
    fclose(fp);
    return 1;
}

/* Return 1 if `cpu` belongs to `numa_node` via /sys/devices/system/node. */
static int cpu_on_node(int numa_node, int cpu)
{
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", numa_node);
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    fclose(fp);

    /* parse cpulist: "0-15,48-63" style */
    int ok = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok;
         tok = strtok_r(NULL, ",", &save)) {
        long lo, hi;
        char *dash = strchr(tok, '-');
        if (dash) {
            *dash = '\0';
            lo = strtol(tok, NULL, 10);
            hi = strtol(dash + 1, NULL, 10);
        } else {
            lo = hi = strtol(tok, NULL, 10);
        }
        if (cpu >= (int)lo && cpu <= (int)hi) { ok = 1; break; }
    }
    return ok;
}

#ifdef __linux__
/* All IIO fields are packed into the 64-bit `config` (see note below).
 * `config` = low 32 (event+umask), `config1` = high 32 (ch_mask+fc_mask),
 * `config2` unused. */
static int perf_open(uint32_t type, uint64_t config, uint64_t config1,
                     uint64_t config2, int cpu)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = type;
    /* Intel uncore IIO reads ALL fields (event, umask, ch_mask, fc_mask)
     * from the low 48 bits of attr.config: low 32 = cfg[0] (event+umask),
     * high 32 = cfg[1] (ch_mask+fc_mask). attr.config1 is IGNORED by the
     * IIO PMU -- putting the masks there yields ch_mask=fc_mask=0 and the
     * counters read 0 (verified: perf packs config=0x70ff000000484). */
    attr.config = config | (config1 << 32);
    (void)config2;
    attr.disabled = 0;
    attr.inherit = 1;
    return (int)syscall(SYS_perf_event_open, &attr, -1 /* all pids */, cpu,
                        -1, 0);
}
#else
static int perf_open(uint32_t type, uint64_t config, uint64_t config1,
                     uint64_t config2, int cpu)
{
    (void)type; (void)config; (void)config1; (void)config2; (void)cpu;
    errno = ENOSYS;
    return -1;
}
#endif /* __linux__ */

/* Open the two read counters for one PMU. 0 on success. */
static int open_pmu_counters(struct iio_pmu *pmu, int cpu)
{
    uint32_t cfg[3];
    uint32_t vals[8] = {0};
    int have_event = 0, have_umask = 0;

    for (int i = 0; i < pmu->nspecs; i++) {
        const iio_field_spec_t *sp = &pmu->specs[i];
        if (strcmp(sp->name, "event") == 0) have_event = 1;
        else if (strcmp(sp->name, "umask") == 0) have_umask = 1;
    }
    if (!have_event || !have_umask) return -1;

    /* field values: ch_mask/fc_mask pinned; event/umask per measurement */
    for (int i = 0; i < pmu->nspecs; i++) {
        const iio_field_spec_t *sp = &pmu->specs[i];
        if (strcmp(sp->name, "ch_mask") == 0) vals[i] = IIO_CH_MASK_ALL;
        else if (strcmp(sp->name, "fc_mask") == 0) vals[i] = IIO_FC_MASK_ALL;
        else if (strcmp(sp->name, "umask") == 0) vals[i] = IIO_UMASK_64B_READ;
        else vals[i] = 0;
    }
    for (int i = 0; i < pmu->nspecs; i++)
        if (strcmp(pmu->specs[i].name, "event") == 0) vals[i] = IIO_EVT_READ_TRANSACTIONS;
    if (iio_encode(pmu->specs, vals, pmu->nspecs, cfg) != 0) return -1;
    pmu->fd_txn = perf_open(pmu->type, cfg[0], cfg[1], cfg[2], cpu);
    if (pmu->fd_txn < 0) return -1;

    for (int i = 0; i < pmu->nspecs; i++)
        if (strcmp(pmu->specs[i].name, "event") == 0) vals[i] = IIO_EVT_READ_UNITS;
    if (iio_encode(pmu->specs, vals, pmu->nspecs, cfg) != 0) { close(pmu->fd_txn); pmu->fd_txn = -1; return -1; }
    pmu->fd_unit = perf_open(pmu->type, cfg[0], cfg[1], cfg[2], cpu);
    if (pmu->fd_unit < 0) { close(pmu->fd_txn); pmu->fd_txn = -1; pmu->fd_unit = -1; return -1; }
    return 0;
}

#ifdef __linux__
static int read_pmu_type(struct iio_pmu *pmu, const char *devdir)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/type", devdir);
    return read_uint(path, (uint64_t *)&pmu->type) ? 0 : -1;
}
#endif

/*
 * Discover uncore_iio_N devices under /sys/bus/event_source/devices, read
 * their format files, and open the two read counters on `cpu` (which
 * must be on `numa_node`; we still try, and a wrong-socket PMU simply
 * contributes its own socket's traffic -- see header note).
 */
int iio_open(int numa_node, int cpu)
{
    iio_close();
    if (cpu < 0) cpu = 0;
    if (numa_node < 0) numa_node = 0;
    g_numa_node = numa_node;
    g_cpu = cpu;

#ifdef __linux__
    if (!cpu_on_node(numa_node, cpu)) {
        /* fall back to first CPU of the node */
        char path[256], buf[1024];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d/cpulist", numa_node);
        FILE *fp = fopen(path, "r");
        if (fp) {
            size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
            buf[n] = '\0';
            fclose(fp);
            char *tok = strtok(buf, ",");
            if (tok) cpu = (int)strtol(tok, NULL, 10);
        }
        g_cpu = cpu;
    }

    DIR *devs = opendir("/sys/bus/event_source/devices");
    if (!devs) return 0;
    struct dirent *de;
    while ((de = readdir(devs)) != NULL) {
        if (strncmp(de->d_name, "uncore_iio_", 11) != 0) continue;
        if (!isdigit((unsigned char)de->d_name[11])) continue;
        if (g_npmus >= IIO_MAX_PMUS) break;

        char devdir[PATH_MAX];
        snprintf(devdir, sizeof(devdir), "/sys/bus/event_source/devices/%s",
                 de->d_name);

        struct iio_pmu *pmu = &g_pmus[g_npmus];
        memset(pmu, 0, sizeof(*pmu));
        snprintf(pmu->name, sizeof(pmu->name), "%s", de->d_name);
        if (read_pmu_type(pmu, devdir) != 0) continue;
        if (!load_pmu_format(pmu, devdir)) continue;
        if (open_pmu_counters(pmu, cpu) != 0) continue;
        g_npmus++;
        (void)iio_pmu_snapshot(g_npmus - 1, &g_prev_pmu[g_npmus - 1]);
    }
    closedir(devs);
#endif /* __linux__ */
    memset(g_prev_pmu, 0, sizeof(g_prev_pmu));
    return g_npmus;
}

void iio_close(void)
{
    for (int i = 0; i < g_npmus; i++) {
        if (g_pmus[i].fd_txn >= 0) close(g_pmus[i].fd_txn);
        if (g_pmus[i].fd_unit >= 0) close(g_pmus[i].fd_unit);
    }
    g_npmus = 0;
    g_cpu = -1;
    g_numa_node = -1;
}

/* ------------------------------------------------------------------ */
/* reads                                                               */
/* ------------------------------------------------------------------ */

static int read_one_fd(int fd, uint64_t *v)
{
    uint64_t val;
    ssize_t n;
    do {
        n = read(fd, &val, sizeof(val));
    } while (n == -1 && errno == EINTR);
    if (n != (ssize_t)sizeof(val)) return -1;
    *v = val;
    return 0;
}

int iio_pmu_snapshot(int idx, iio_snapshot_t *s)
{
    if (idx < 0 || idx >= g_npmus) return -1;
    memset(s, 0, sizeof(*s));
    if (read_one_fd(g_pmus[idx].fd_txn, &s->rd_trans_64b) != 0) return -1;
    if (read_one_fd(g_pmus[idx].fd_unit, &s->rd_units_4b) != 0) return -1;
    return 0;
}

int iio_snapshot(iio_snapshot_t *s)
{
    if (g_npmus == 0) return -1;
    memset(s, 0, sizeof(*s));
    for (int i = 0; i < g_npmus; i++) {
        iio_snapshot_t one;
        if (iio_pmu_snapshot(i, &one) != 0) return -1;
        s->rd_trans_64b += one.rd_trans_64b;
        s->rd_units_4b  += one.rd_units_4b;
    }
    return 0;
}

void iio_diff(const iio_snapshot_t *a, const iio_snapshot_t *b, iio_snapshot_t *d)
{
    d->rd_trans_64b = b->rd_trans_64b - a->rd_trans_64b;
    d->rd_units_4b  = b->rd_units_4b  - a->rd_units_4b;
}

int iio_dominant_pmu(void)
{
    int best = -1;
    uint64_t best_delta = 0;
    for (int i = 0; i < g_npmus; i++) {
        iio_snapshot_t now;
        if (iio_pmu_snapshot(i, &now) != 0) continue;
        uint64_t delta = now.rd_trans_64b >= g_prev_pmu[i].rd_trans_64b
                             ? now.rd_trans_64b - g_prev_pmu[i].rd_trans_64b
                             : 0;
        g_prev_pmu[i] = now;   /* rolling window */
        if (delta > best_delta) {
            best_delta = delta;
            best = i;
        }
    }
    return best;
}
