#!/bin/bash
# qpsweep.sh -- IIO/ICMC proxy reference test (plan step 0.2)
#
# Self-contained single-node sweep: for each QP count, launch a fresh
# ib_write_bw server in the background and run a perf-measured client against
# the local IP, reading the Intel uncore IIO PMU read counters around each run.
# The 64B-read-transaction count is all non-payload NIC reads (QPC +
# MKey/MTT cache misses together) -- an ESTIMATE of ICMC activity, not a real
# miss counter.
#
# Acceptance (before a2a ICMC data is trusted):
#   QPs <= 64  : ~0 excess 64B read txns per message
#   QPs >= 128 : ~1 64B read txn per message (cache-miss knee)
#
# Usage:
#   qpsweep.sh [qps...]        (default: 1 2 4 8 16 32 64 128 256 512)
#
# Requirements (unprivileged, RHEL9 spr/emr): perf_event_paranoid <= 1,
# perf, perftest (ib_write_bw), RDMA device.  No sudo.  Run ON the node whose
# NIC you want to measure (the client/sender side fires the IIO read counters).

set -u
DEV=${DEV:-mlx5_0}
PORT=${PORT:-18515}
SIZE=${SIZE:-8}        # 8 B: pure header traffic, smallest payload
DUR=${DUR:-5}          # seconds per run (-D)
WAIT=${WAIT:-2}        # seconds for the bg server to come up
QPS_DEFAULT="1 2 4 8 16 32 64 128 256 512"
HOST=${HOST:-$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | awk -F/ '{print $1}' | head -1)}

QPS=$*
[ -n "$QPS" ] || QPS=$QPS_DEFAULT

command -v ib_write_bw >/dev/null 2>&1 || { echo "ERROR: ib_write_bw not found (perftest)"; exit 2; }
[ -n "$HOST" ] || { echo "ERROR: cannot determine local IP; set HOST=..."; exit 2; }

# All uncore_iio_N PMUs (perf wants one -e per PMU instance).
IIO_PMUS=$(ls -d /sys/bus/event_source/devices/uncore_iio_* 2>/dev/null \
           | xargs -n1 basename 2>/dev/null \
           | grep -E '^uncore_iio_[0-9]+$' | sort -t_ -k3 -n)
if [ -z "$IIO_PMUS" ]; then
    echo "ERROR: no uncore_iio_* PMUs found (need Intel spr/emr)."
    exit 2
fi
# ch_mask/fc_mask MUST be passed as named perf fields, NOT config1: the IIO
# PMU reads the masks from fixed config bit positions and perf only encodes
# them there via the named form -- `config1=0x70FF0` is silently dropped and
# the counter reads 0 (verified on spr: ch_mask/fc_mask -> 270, config1 -> 0).
# IIO event 0x84 (64B read txns), umask 0x04, all channels + function
# complexes (SPR/EMR), exactly matching common/iio_pmu.c.
IIO_MASKS="ch_mask=0xff,fc_mask=0x7"
# Both the leading and trailing / around the raw config are required
# (pmu/config/); a missing trailing / -> "event syntax error".
PERF_E=()
for p in $IIO_PMUS; do
    PERF_E+=("-e" "$p/event=0x84,umask=0x4,$IIO_MASKS/")
    PERF_E+=("-e" "$p/event=0x83,umask=0x4,$IIO_MASKS/")
done
if ! perf stat -a "${PERF_E[@]}" true >/dev/null 2>&1; then
    echo "ERROR: cannot read uncore_iio_* perf events (need perf_event_"
    echo "       paranoid <= 1 and a spr/emr-class CPU)."
    exit 2
fi

printf "%-6s %14s %14s %16s\n" "QPs" "Mmsg/s" "64B_rd_txn" "rd_txn/msg"
for q in $QPS; do
    # Fresh server for this QP count. NO -R: RDMA-CM loopback (server+client
    # on one node) fails with ADDR_ERROR; the plain TCP handshake works and
    # the data path is still RDMA.
    ib_write_bw -d "$DEV" -s "$SIZE" -D "$DUR" -q "$q" -p "$PORT" \
        > /dev/null 2>&1 &
    SRV=$!
    sleep "$WAIT"

    OUT=$(perf stat -a "${PERF_E[@]}" \
            ib_write_bw -d "$DEV" -s "$SIZE" -D "$DUR" -q "$q" \
            -p "$PORT" "$HOST" 2>&1)

    # The perftest summary line is:
    #   <bytes>  <#iterations>  <BW peak>  <BW avg>  <MsgRate>
    # #iterations = number of messages sent at SIZE bytes.
    MSGS=$(printf '%s\n' "$OUT" \
             | awk -v sz="$SIZE" 'NF>=4 && $1==sz {print $2; exit}')

    # perf stat prints "<count>  <event>"; sum the 0x84 (64B read txn)
    # counters across all PMUs.
    T=$(printf '%s\n' "$OUT" \
         | awk '$2 ~ /event=0x84,umask=0x4/ {s += $1} END {print s+0}')

    wait "$SRV" 2>/dev/null   # reap the server (exits after -D)

    if [ -n "${MSGS:-}" ] && [ "${MSGS:-0}" -gt 0 ] 2>/dev/null; then
        awk -v m="$MSGS" -v t="$T" -v q="$q" \
            'BEGIN{printf "%-6d %14.2f %14d %16.4f\n", q, m/1e6, t, t/m}'
    else
        printf "%-6d %14s %14s %16s\n  (failed run; see output above)\n" "$q" "-" "-" "-"
        printf '%s\n' "$OUT" | tail -5
    fi
done
echo
echo "Expect ~0 rd_txn/msg at QPs<=64 and ~1 at QPs>=128."
exit 0
