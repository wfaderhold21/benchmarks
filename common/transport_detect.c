/*
 * transport_detect.c -- Implementation of shared transport detection module.
 *
 * Lifted from ucc/timing-ucc/time-ucc.c so that every benchmark can report
 * which UCC TLs and UCX transports are loaded at run time without duplicating
 * the dl_iterate_phdr walk.
 */

#define _GNU_SOURCE

#include "transport_detect.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <link.h>

#define MAX_TLS 32

typedef struct {
    char names[MAX_TLS][64];
    int  count;
} tl_list_t;

/* ---------- UCC TLs (libucc_tl_*.so) ---------- */

static int find_ucc_tls_cb(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    tl_list_t *list = (tl_list_t *)data;
    const char *name = info->dlpi_name;

    if (!name || strlen(name) == 0) return 0;

    const char *base = strrchr(name, '/');
    base = base ? base + 1 : name;

    if (strncmp(base, "libucc_tl_", 10) == 0) {
        const char *s = base + 10;
        const char *e = strstr(s, ".so");
        if (e && list->count < MAX_TLS) {
            size_t len = (size_t)(e - s);
            if (len > 0 && len < sizeof(list->names[0])) {
                strncpy(list->names[list->count], s, len);
                list->names[list->count][len] = '\0';
                list->count++;
            }
        }
    }
    return 0;
}

/* ---------- UCX transports (libuct_*.so) ---------- */

static int find_ucx_tls_cb(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    tl_list_t *list = (tl_list_t *)data;
    const char *name = info->dlpi_name;

    if (!name || strlen(name) == 0) return 0;

    const char *base = strrchr(name, '/');
    base = base ? base + 1 : name;

    if (strncmp(base, "libuct_", 7) == 0) {
        const char *s = base + 7;
        const char *e = strstr(s, ".so");
        if (e && list->count < MAX_TLS) {
            size_t len = (size_t)(e - s);
            if (len > 0 && len < sizeof(list->names[0])) {
                strncpy(list->names[list->count], s, len);
                list->names[list->count][len] = '\0';
                list->count++;
            }
        }
    }
    return 0;
}

/* ---------- Public API ---------- */

char *transport_detect(void)
{
    tl_list_t ucc_tls, ucx_tls;
    ucc_tls.count = 0;
    ucx_tls.count = 0;

    dl_iterate_phdr(find_ucc_tls_cb, &ucc_tls);
    dl_iterate_phdr(find_ucx_tls_cb, &ucx_tls);

    /* worst-case: each name (63 chars) + comma, times (MAX_TLS*2), plus NUL */
    size_t cap = (size_t)(ucc_tls.count + ucx_tls.count) * 70 + 1;
    char  *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    int n = 0;
    for (int i = 0; i < ucc_tls.count; i++) {
        n += snprintf(buf + n, cap - (size_t)n, "%s%s",
                      n > 0 ? "," : "", ucc_tls.names[i]);
    }
    for (int i = 0; i < ucx_tls.count; i++) {
        n += snprintf(buf + n, cap - (size_t)n, "%s%s",
                      n > 0 ? "," : "", ucx_tls.names[i]);
    }

    return buf;
}

void transport_free(char *s)
{
    free(s);
}
