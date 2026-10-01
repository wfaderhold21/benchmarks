#!/bin/bash
# probe_icmc.sh -- capture ICMC tooling schemas on a compute node
set -u
echo "===NODE==="; hostname; uname -r
echo "===PERF_PARANOID==="; cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a
echo "===RDMA_JSON==="; rdma -j res show qp 2>&1 | head -c 2500; echo
echo "===IIO==="
ls -d /sys/bus/event_source/devices/uncore_iio_* 2>/dev/null | head -8
for d in $(ls -d /sys/bus/event_source/devices/uncore_iio_* 2>/dev/null | head -2); do
  echo "== $d"
  cat "$d/type" 2>/dev/null
  for f in "$d"/format/*; do
    printf "%s: " "$(basename "$f")"; cat "$f"; echo
  done
done
echo "===HDR==="
ls /usr/include/rdma/rdma_netlink.h /usr/include/linux/rdma.h 2>&1
grep -n "RES_QP_GET\|RES_QP_TYPE\|RES_PID\|RES_OBJ\|RES_QP_STATE\|RES_QP_PKEY" /usr/include/rdma/rdma_netlink.h 2>/dev/null | head -20
grep -n "rdma_res_obj\|rdma_res_attrs" /usr/include/linux/rdma.h 2>/dev/null | head -10
echo "===IB==="
ls /sys/class/infiniband/ 2>/dev/null
for d in /sys/class/infiniband/*/; do
  echo "$d numa=$(cat ${d}device/numa_node 2>/dev/null)"
done
which ib_write_bw ib_read_lat 2>/dev/null || echo no-perftest
echo "===CPULISTS==="
head -4 /sys/devices/system/node/node*/cpulist 2>/dev/null
