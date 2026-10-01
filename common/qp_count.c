/*
 * qp_count.c -- RDMA QP counts (node-wide and per-process)
 *
 * Backend: `rdma -j res show qp` (iproute2), which is unprivileged and
 * stable across kernel versions.  The JSON parser is a pure function
 * (qp_parse_json) so it is unit tested offline (tools/icmc/test_parsers.c).
 *
 * Why not raw RDMA netlink?  On the target kernel (5.14) the netlink
 * RES dump ABI predates RDMA_NLDEV_ATTR_RES_OBJ / RES_QP_TYPE, so a
 * hand-rolled netlink dumper is version-fragile.  `rdma -j` is the
 * supported unprivileged interface and produces the same data; we call
 * it once per measurement window (after warm-up), not per iteration.
 */

#define _GNU_SOURCE
#include "qp_count.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* JSON parsing (pure, unit-tested)                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *p;
    const char *dev;   /* ifname filter; NULL = count every device */
    int err;
} jp_t;

static void jp_ws(jp_t *j)
{
    while (*j->p && (unsigned char)*j->p <= ' ') j->p++;
}

static int jp_peek(jp_t *j)
{
    jp_ws(j);
    return (unsigned char)*j->p;
}

/* Copy the string value at the current position (must be at '"').
 * Ends at the closing '"' honoring \" escapes.  Returns chars written
 * (excl. NUL), -1 on error.  Leaves j->p after the quote. */
static int jp_string(jp_t *j, char *dst, size_t cap)
{
    if (*j->p != '"') { j->err = 1; return -1; }
    j->p++;
    size_t n = 0;
    for (;;) {
        char c = *j->p;
        if (c == '\0') { j->err = 1; return -1; }
        if (c == '\\') {
            j->p++;
            char e = *j->p++;
            if (e == '\0') { j->err = 1; return -1; }
            c = (e == 'n') ? '\n' : (e == 't') ? '\t' : (e == 'r') ? '\r'
                : (e == '"') ? '"' : (e == '\\') ? '\\'
                : (e == '/') ? '/' : e;
        } else if (c == '"') {
            j->p++;
            break;
        } else {
            j->p++;
        }
        if (n + 1 < cap) dst[n++] = c;
    }
    dst[n] = '\0';
    return (int)n;
}

/* Skip a JSON value of any kind.  0 on success. */
static int jp_skip_value(jp_t *j)
{
    int c = jp_peek(j);
    if (c == '"') {
        char dummy[64];
        return jp_string(j, dummy, sizeof(dummy)) < 0 ? 1 : 0;
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        while (*j->p) {
            char ch = *j->p;
            if (ch == '"') {
                char dummy[64];
                if (jp_string(j, dummy, sizeof(dummy)) < 0) return 1;
                continue;
            }
            if (ch == '{' || ch == '[') depth++;
            else if (ch == '}' || ch == ']') {
                depth--;
                j->p++;
                if (depth == 0) return 0;
                continue;
            }
            j->p++;
        }
        j->err = 1;
        return 1;
    }
    /* number / true / false / null */
    while (*j->p && *j->p != ',' && *j->p != '}' && *j->p != ']' &&
           *j->p != ':' && (unsigned char)*j->p > ' ')
        j->p++;
    return 0;
}

static int add_type(qp_count_t *out, const char *name)
{
    for (int i = 0; i < out->ntypes; i++)
        if (strcmp(out->types[i].name, name) == 0) {
            out->types[i].count++;
            return 0;
        }
    if (out->ntypes >= QP_TYPE_MAX) return 0;   /* total still exact */
    snprintf(out->types[out->ntypes].name, sizeof(out->types[0].name),
             "%s", name);
    out->types[out->ntypes].count = 1;
    out->ntypes++;
    return 0;
}

/* Parse one QP object (j->p at '{').  Extracts "type" and "ifname";
 * counts the QP only when the device filter (j->dev) is NULL or matches
 * the object's ifname. */
static int jp_parse_qp_obj(jp_t *j, qp_count_t *out)
{
    if (*j->p != '{') { j->err = 1; return -1; }
    j->p++;
    char ifname[QP_DEV_NAME_MAX] = "";
    char type[8] = "";
    for (;;) {
        jp_ws(j);
        if (*j->p == '}') { j->p++; break; }
        if (*j->p == ',') { j->p++; continue; }
        if (*j->p != '"') { j->err = 1; return -1; }
        char key[32];
        if (jp_string(j, key, sizeof(key)) < 0) return -1;
        jp_ws(j);
        if (*j->p != ':') { j->err = 1; return -1; }
        j->p++;
        int c = jp_peek(j);
        if (c == '"') {
            char val[QP_DEV_NAME_MAX + 8];
            if (jp_string(j, val, sizeof(val)) < 0) return -1;
            if (strcmp(key, "type") == 0) {
                if (val[0] && !type[0])
                    snprintf(type, sizeof(type), "%s", val);
            } else if (strcmp(key, "ifname") == 0 ||
                       strcmp(key, "device") == 0) {
                if (!ifname[0])
                    snprintf(ifname, sizeof(ifname), "%s", val);
            }
        } else {
            if (jp_skip_value(j)) return -1;
        }
    }
    /* Device filter: `rdma res show qp` has no dev filter, so the
     * command returns every device; drop QPs off the wanted interface. */
    if (j->dev && (ifname[0] == '\0' || strcmp(ifname, j->dev) != 0))
        return 0;
    if (type[0]) add_type(out, type);
    out->total++;
    if (ifname[0] && !out->device[0])
        snprintf(out->device, sizeof(out->device), "%s", ifname);
    return 0;
}

/* Parse a JSON array of QP objects.  j->p at '['.  0 on success. */
static int jp_parse_array(jp_t *j, qp_count_t *out)
{
    if (*j->p != '[') { j->err = 1; return -1; }
    j->p++;
    for (;;) {
        jp_ws(j);
        if (*j->p == ']') { j->p++; break; }
        if (*j->p == ',') { j->p++; continue; }
        if (*j->p == '\0') { j->err = 1; return -1; }
        if (jp_parse_qp_obj(j, out) < 0) return -1;
    }
    return 0;
}

int qp_parse_json(const char *json, qp_count_t *out)
{
    return qp_parse_json_dev(json, NULL, out);
}

int qp_parse_json_dev(const char *json, const char *dev, qp_count_t *out)
{
    if (!json || !out) return -1;
    memset(out, 0, sizeof(*out));
    jp_t j = { .p = json, .dev = dev, .err = 0 };
    for (;;) {
        jp_ws(&j);
        if (*j.p == '\0') break;
        if (*j.p != '[') return -1;      /* top level must be arrays only */
        if (jp_parse_array(&j, out) < 0) return -1;
        if (j.err) return -1;
    }
    return (int)out->total;
}

/* ------------------------------------------------------------------ */
/* `rdma -j` backend                                                   */
/* ------------------------------------------------------------------ */

static const char *find_rdma_bin(void)
{
    static const char *cands[] = {
        "/usr/sbin/rdma",
        "/opt/mellanox/iproute2/sbin/rdma",
        "/usr/local/hpcx/rdma/rdma",
        NULL
    };
    if (access("rdma", X_OK) == 0) return "rdma";
    for (int i = 0; cands[i]; i++)
        if (access(cands[i], X_OK) == 0) return cands[i];
    return NULL;
}

static int qp_query_cmd(const char *device, int pid, qp_count_t *out)
{
    const char *bin = find_rdma_bin();
    if (!bin) return -1;

    char cmd[256];
    int len = snprintf(cmd, sizeof(cmd), "%s -j res show qp", bin);
    if (pid >= 0)
        len += snprintf(cmd + len, sizeof(cmd) - (size_t)len, " pid %d", pid);

    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;

    static char buf[8 * 1024 * 1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    pclose(fp);

    /* `rdma res show qp` has no device filter; the device is applied
     * client-side against each object's ifname. */
    return qp_parse_json_dev(buf, device, out) >= 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

int qp_query(const char *device, int pid, qp_count_t *out, int *used_netlink)
{
    if (qp_query_cmd(device, pid, out) == 0) {
        if (used_netlink) *used_netlink = 0;   /* backend id: 0 = rdma -j */
        return 0;
    }
    return -1;
}

int qp_sample(const char *device, qp_count_t *node, qp_count_t *self)
{
    int used = 0;
    if (qp_query(device, -1, node, &used) != 0) return -1;
    if (qp_query(device, (int)getpid(), self, NULL) != 0) return -1;
    return 0;
}

const char *qp_type_string(const qp_count_t *c)
{
    static char buf[256];
    size_t n = 0;
    for (int i = 0; i < c->ntypes; i++) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n, "%s%s=%llu",
                              i ? " " : "", c->types[i].name,
                              (unsigned long long)c->types[i].count);
    }
    return buf;
}
