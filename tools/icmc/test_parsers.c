/*
 * test_parsers.c -- offline unit tests for the ICMC parsers (no NIC)
 *
 *   - qp_parse_json(): `rdma -j res show qp` JSON, verbatim schema
 *     captured on hpcac-internal spr001 (iproute2 6.17):
 *       {"ifindex":N,"ifname":"mlx5_0","port":1,"lqpn":1,"type":"GSI",
 *        "state":"RTS","sq-psn":N,"comm":"ib_core"}
 *     QP counts by type (RC/UC/UD/DCI/DCT/GSI) and totals must be
 *     exact; malformed input must be rejected.
 *
 *   - iio_encode(): perf_event_attr.config encoding from the real
 *     uncore_iio_N format files on spr001:
 *       ch_mask: config:36-47
 *       edge:    config:18
 *       event:   config:0-7
 *       fc_mask: config:48-50
 *       inv:     config:23
 *       thresh:  config:24-35
 *       umask:   config:8-15
 *     ch_mask crosses the config/config1 32-bit word boundary; the
 *     encoded words must match the kernel's expectation.
 *
 * Build:  cc -O2 -I../../common test_parsers.c ../../common/qp_count.c \
 *              ../../common/iio_pmu.c -o test_parsers && ./test_parsers
 * Exit 0 = all pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qp_count.h"
#include "iio_pmu.h"

static int nfail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("ok   %s\n", msg); } \
    else      { printf("FAIL %s  (%s:%d)\n", msg, __FILE__, __LINE__); nfail++; } \
} while (0)

/* ------------------------------------------------------------------ */
/* qp_parse_json                                                       */
/* ------------------------------------------------------------------ */

/* Verbatim capture: spr001, `rdma -j res show qp` (2 GSI QPs, two NICs). */
static const char *FIXTURE_GSI =
    "[{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":1,"
    "\"type\":\"GSI\",\"state\":\"RTS\",\"sq-psn\":10821,\"comm\":\"ib_core\"}\n"
    ",{\"ifindex\":1,\"ifname\":\"mlx5_1\",\"port\":1,\"lqpn\":1,"
    "\"type\":\"GSI\",\"state\":\"RTS\",\"sq-psn\":0,\"comm\":\"ib_core\"}\n"
    "]";

static const char *FIXTURE_MIXED =
    "[{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":8,"
    "\"type\":\"RC\",\"state\":\"RTS\",\"sq-psn\":1,\"comm\":\"ucc_bench\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":9,"
    "\"type\":\"RC\",\"state\":\"RTS\",\"sq-psn\":2,\"comm\":\"ucc_bench\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":10,"
    "\"type\":\"UD\",\"state\":\"RTS\",\"sq-psn\":3,\"comm\":\"ucx\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":11,"
    "\"type\":\"DCI\",\"state\":\"RTS\",\"sq-psn\":4,\"comm\":\"ucx\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":12,"
    "\"type\":\"DCT\",\"state\":\"RTS\",\"sq-psn\":5,\"comm\":\"ucx\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":13,"
    "\"type\":\"RC\",\"state\":\"ERR\",\"sq-psn\":6,\"comm\":\"ucx\"},"
    "{\"ifindex\":0,\"ifname\":\"mlx5_0\",\"port\":1,\"lqpn\":14,"
    "\"type\":\"UC\",\"state\":\"RTS\",\"sq-psn\":7,"
    "\"driver\":{\"nested\":true,\"num\":42},\"comm\":\"ucx\"}]";

static const char *FIXTURE_MULTIPLE_ARRAYS =
    "[{\"ifname\":\"mlx5_0\",\"type\":\"RC\",\"state\":\"RTS\"}]\n"
    "[{\"ifname\":\"mlx5_0\",\"type\":\"RC\",\"state\":\"RTS\"},"
     "{\"ifname\":\"mlx5_0\",\"type\":\"UD\",\"state\":\"RTS\"}]\n";

static void test_qp_parse_gsi(void)
{
    qp_count_t out;
    int n = qp_parse_json(FIXTURE_GSI, &out);
    CHECK(n == 2, "gsi: parsed 2 QPs");
    CHECK(out.total == 2, "gsi: total == 2");
    CHECK(out.ntypes == 1, "gsi: one type bucket");
    CHECK(out.ntypes == 1 && strcmp(out.types[0].name, "GSI") == 0,
          "gsi: type is GSI");
    CHECK(out.ntypes == 1 && out.types[0].count == 2, "gsi: GSI count == 2");
    CHECK(strcmp(out.device, "mlx5_0") == 0, "gsi: first device mlx5_0");
}

static void test_qp_parse_mixed(void)
{
    qp_count_t out;
    int n = qp_parse_json(FIXTURE_MIXED, &out);
    CHECK(n == 7, "mixed: parsed 7 QPs");
    CHECK(out.total == 7, "mixed: total == 7");
    CHECK(out.ntypes == 5, "mixed: 5 distinct types");

    uint64_t rc = 0, ud = 0, dci = 0, dct = 0, uc = 0;
    for (int i = 0; i < out.ntypes; i++) {
        if (strcmp(out.types[i].name, "RC") == 0) rc = out.types[i].count;
        else if (strcmp(out.types[i].name, "UD") == 0) ud = out.types[i].count;
        else if (strcmp(out.types[i].name, "DCI") == 0) dci = out.types[i].count;
        else if (strcmp(out.types[i].name, "DCT") == 0) dct = out.types[i].count;
        else if (strcmp(out.types[i].name, "UC") == 0) uc = out.types[i].count;
    }
    CHECK(rc == 3, "mixed: RC == 3");
    CHECK(ud == 1, "mixed: UD == 1");
    CHECK(dci == 1, "mixed: DCI == 1");
    CHECK(dct == 1, "mixed: DCT == 1");
    CHECK(uc == 1, "mixed: UC == 1");
}

static void test_qp_parse_multiple_arrays(void)
{
    qp_count_t out;
    int n = qp_parse_json(FIXTURE_MULTIPLE_ARRAYS, &out);
    CHECK(n == 3, "arrays: parsed 3 QPs across 2 arrays");
    CHECK(out.total == 3, "arrays: total == 3");
}

static void test_qp_parse_errors(void)
{
    qp_count_t out;
    CHECK(qp_parse_json("", &out) == 0, "ok: empty output -> 0 QPs");
    CHECK(qp_parse_json("not json at all", &out) == -1, "err: junk rejected");
    CHECK(qp_parse_json("[{\"type\":", &out) == -1, "err: truncated rejected");
    CHECK(qp_parse_json("42", &out) == -1, "err: bare number rejected");
    qp_count_t out2;
    int n = qp_parse_json("[]", &out2);
    CHECK(n == 0 && out2.total == 0, "ok: empty array -> 0 QPs");
}

static void test_qp_parse_dev_filter(void)
{
    /* NULL device -> all QPs counted (2 GSI across mlx5_0 + mlx5_1). */
    qp_count_t all;
    CHECK(qp_parse_json(FIXTURE_GSI, &all) == 2, "dev: NULL filter counts all");
    CHECK(all.total == 2, "dev: NULL filter total == 2");

    /* ifname == mlx5_0 -> only the first GSI QP. */
    qp_count_t d0;
    int n0 = qp_parse_json_dev(FIXTURE_GSI, "mlx5_0", &d0);
    CHECK(n0 == 1, "dev: mlx5_0 counts 1");
    CHECK(d0.total == 1, "dev: mlx5_0 total == 1");
    CHECK(d0.ntypes == 1 && strcmp(d0.types[0].name, "GSI") == 0,
          "dev: mlx5_0 type GSI");

    /* ifname == mlx5_1 -> only the second GSI QP. */
    qp_count_t d1;
    int n1 = qp_parse_json_dev(FIXTURE_GSI, "mlx5_1", &d1);
    CHECK(n1 == 1, "dev: mlx5_1 counts 1");
    CHECK(d1.total == 1, "dev: mlx5_1 total == 1");

    /* unknown device -> 0 QPs (not an error). */
    qp_count_t dn;
    int nn = qp_parse_json_dev(FIXTURE_GSI, "mlx5_9", &dn);
    CHECK(nn == 0 && dn.total == 0, "dev: unknown ifname -> 0 QPs");

    /* Mixed fixture: 7 RC/UD/DCI/DCT/UC all on mlx5_0 -> all kept. */
    qp_count_t dm;
    CHECK(qp_parse_json_dev(FIXTURE_MIXED, "mlx5_0", &dm) == 7,
          "dev: mixed all-on-mlx5_0 -> 7");
    CHECK(dm.total == 7, "dev: mixed total == 7");
}

/* ------------------------------------------------------------------ */
/* iio_encode                                                          */
/* ------------------------------------------------------------------ */

/* The real spr001 uncore_iio_0 format file contents (absolute bits). */
static iio_field_spec_t SPR_SPECS[] = {
    { .name = "ch_mask", .lsb = 36, .width = 12 },
    { .name = "edge",    .lsb = 18, .width =  1 },
    { .name = "event",   .lsb =  0, .width =  8 },
    { .name = "fc_mask", .lsb = 48, .width =  3 },
    { .name = "inv",     .lsb = 23, .width =  1 },
    { .name = "thresh",  .lsb = 24, .width = 12 },
    { .name = "umask",   .lsb =  8, .width =  8 },
};

static int spr_spec_index(const char *name)
{
    for (size_t i = 0; i < sizeof(SPR_SPECS)/sizeof(SPR_SPECS[0]); i++)
        if (strcmp(SPR_SPECS[i].name, name) == 0) return (int)i;
    return -1;
}

/* event=0x84, umask=0x4, ch_mask=0xff, fc_mask=0x7 -> expected words. */
static void test_iio_encode_read_txn(void)
{
    uint32_t vals[7] = {0};
    vals[spr_spec_index("event")]   = IIO_EVT_READ_TRANSACTIONS; /* 0x84 */
    vals[spr_spec_index("umask")]   = IIO_UMASK_64B_READ;        /* 0x4  */
    vals[spr_spec_index("ch_mask")] = IIO_CH_MASK_ALL;           /* 0xff */
    vals[spr_spec_index("fc_mask")] = IIO_FC_MASK_ALL;           /* 0x7  */

    uint32_t cfg[3];
    int rc = iio_encode(SPR_SPECS, vals, 7, cfg);
    CHECK(rc == 0, "iio: encode succeeds with real SPR layout");
    /* config  = event|umask<<8 = 0x84 | 0x400 = 0x00000484 */
    CHECK(cfg[0] == 0x00000484u, "iio: config == 0x00000484");
    /* ch_mask 0xff at abs bits 36..43 -> word1 bits 4..11  = 0x000FF0
     * fc_mask 0x7  at abs bits 48..50 -> word1 bits 16..18 = 0x70000 */
    CHECK(cfg[1] == 0x00070FF0u, "iio: config1 == 0x00070FF0 (ch+fc cross-word)");
    /* config2 unused */
    CHECK(cfg[2] == 0x00000000u, "iio: config2 == 0");
}

static void test_iio_encode_read_units(void)
{
    uint32_t vals[7] = {0};
    vals[spr_spec_index("event")]   = IIO_EVT_READ_UNITS;  /* 0x83 */
    vals[spr_spec_index("umask")]   = IIO_UMASK_4B_READ;   /* 0x4  */
    vals[spr_spec_index("ch_mask")] = IIO_CH_MASK_ALL;
    vals[spr_spec_index("fc_mask")] = IIO_FC_MASK_ALL;

    uint32_t cfg[3];
    int rc = iio_encode(SPR_SPECS, vals, 7, cfg);
    CHECK(rc == 0, "iio: 0x83 encode succeeds");
    CHECK(cfg[0] == 0x00000483u, "iio: 0x83 config == 0x00000483");
    CHECK(cfg[1] == 0x00070FF0u, "iio: 0x83 config1 == 0x00070FF0");
    CHECK(cfg[2] == 0x00000000u, "iio: 0x83 config2 == 0");
}

static void test_iio_encode_errors(void)
{
    uint32_t vals[7] = {0};
    uint32_t cfg[3];

    /* ch_mask is 12 bits wide (bits 36..47); 0x1000 (13 bits) must be
     * rejected */
    vals[spr_spec_index("ch_mask")] = 0x1000;
    CHECK(iio_encode(SPR_SPECS, vals, 7, cfg) == -1,
          "iio: ch_mask overflow rejected");

    /* field beyond config2 (bit >= 96) rejected */
    iio_field_spec_t bad[1] = { { .name = "x", .lsb = 96, .width = 1 } };
    uint32_t bvals[1] = { 1 };
    CHECK(iio_encode(bad, bvals, 1, cfg) == -1, "iio: beyond 96 bits rejected");

    /* field straddling the word0/word1 boundary: bits 31..32 */
    iio_field_spec_t straddle[1] = { { .name = "s", .lsb = 31, .width = 2 } };
    uint32_t svals[1] = { 0x3 };
    memset(cfg, 0xAA, sizeof(cfg));
    CHECK(iio_encode(straddle, svals, 1, cfg) == 0, "iio: straddle encodes");
    CHECK(cfg[0] == 0x80000000u, "iio: straddle sets word0 bit31");
    CHECK(cfg[1] == 0x00000001u, "iio: straddle sets word1 bit0");

    /* NULL args */
    CHECK(iio_encode(NULL, vals, 7, cfg) == -1, "iio: NULL specs rejected");
    CHECK(iio_encode(SPR_SPECS, NULL, 7, cfg) == -1, "iio: NULL vals rejected");
}

/* ------------------------------------------------------------------ */
/* iio_parse_format                                                    */
/* ------------------------------------------------------------------ */

/* The real spr001 uncore_iio_0 format files: name = basename, content =
 * "<word>:<lo>-<hi>".  The parser must reconstruct SPR_SPECS exactly. */
static void test_iio_parse_format(void)
{
    struct { const char *name; const char *content; int lsb, width; } cases[] = {
        { "event",   "config:0-7",     0,  8 },
        { "umask",   "config:8-15",    8,  8 },
        { "edge",    "config:18",      18, 1 },
        { "inv",     "config:23",      23, 1 },
        { "thresh",  "config:24-35",   24, 12 },
        { "ch_mask", "config:36-47",   36, 12 },
        { "fc_mask", "config:48-50",   48, 3 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        iio_field_spec_t sp;
        int rc = iio_parse_format(cases[i].name, cases[i].content, &sp);
        CHECK(rc == 1, "fmt: parses");
        CHECK(strcmp(sp.name, cases[i].name) == 0, "fmt: name from filename");
        CHECK(sp.lsb == (uint32_t)cases[i].lsb, "fmt: lsb");
        CHECK(sp.width == (uint32_t)cases[i].width, "fmt: width");
    }

    /* word1 / word2 content */
    iio_field_spec_t sp;
    CHECK(iio_parse_format("x", "config1:0-3", &sp) == 1 &&
          sp.lsb == 32 && sp.width == 4, "fmt: config1 word offset");
    CHECK(iio_parse_format("y", "config2:31", &sp) == 1 &&
          sp.lsb == 95 && sp.width == 1, "fmt: config2 bit 31 -> abs 95");

    /* whitespace tolerated */
    CHECK(iio_parse_format("event", " config : 0 - 7", &sp) == 1 &&
          sp.lsb == 0 && sp.width == 8, "fmt: whitespace tolerated");

    /* garbage word name rejected */
    CHECK(iio_parse_format("event", "cfg:0-7", &sp) == 0, "fmt: bad word rejected");
    /* range beyond 96 bits rejected */
    CHECK(iio_parse_format("big", "config2:32", &sp) == 0, "fmt: >=96 rejected");
    /* no colon rejected; a bare "config:7" is a valid single-bit field */
    CHECK(iio_parse_format("event", "config", &sp) == 0, "fmt: no colon rejected");
    CHECK(iio_parse_format("event", "config:7", &sp) == 1 &&
          sp.lsb == 7 && sp.width == 1, "fmt: single-bit config:7");
    /* reversed range rejected */
    CHECK(iio_parse_format("event", "config:7-0", &sp) == 0, "fmt: hi<lo rejected");
    /* NULL args */
    CHECK(iio_parse_format(NULL, "config:0-7", &sp) == 0, "fmt: NULL fname rejected");
    CHECK(iio_parse_format("event", NULL, &sp) == 0, "fmt: NULL content rejected");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("== qp_parse_json ==\n");
    test_qp_parse_gsi();
    test_qp_parse_mixed();
    test_qp_parse_multiple_arrays();
    test_qp_parse_dev_filter();
    test_qp_parse_errors();

    printf("== iio_parse_format ==\n");
    test_iio_parse_format();

    printf("== iio_encode ==\n");
    test_iio_encode_read_txn();
    test_iio_encode_read_units();
    test_iio_encode_errors();

    if (nfail) {
        printf("\n%d test(s) FAILED\n", nfail);
        return 1;
    }
    printf("\nall tests passed\n");
    return 0;
}
