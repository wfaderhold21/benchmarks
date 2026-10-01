# Plan: ICMC miss rate vs. QP count in `ucc_bench_a2a_mpi.c`

Goal: keep the existing latency/bandwidth measurement and add, per message
size, the **node-wide QP count** and an **ICM-cache (ICMC) miss estimate**, so
miss rate can be plotted against QP count. Target runtime: HPC-X (UCX + OSHMEM,
built with `shmemcc`), cluster `hpcac-internal`, no sudo.

## Findings that shape the design (probed 2026-09-30)

| Source | Result |
|---|---|
| sysfs `hw_counters`, `rdma statistic`, `ethtool -S` | no ICM counters |
| mlx5 debugfs `diag_cnt` | root only; firmware exposes 0 diag counters anyway |
| DevX `QUERY/SET_DIAGNOSTIC_*` (0x819-0x821) | kernel whitelist -> `EINVAL`; FW `cap.debug=0` on thor/spr/emr |
| NEO-Host, DOCA SDK | not installed |
| `rdma resource show` (netlink) | works unprivileged -> QP counts |
| Intel uncore IIO PMU (`perf_event_paranoid=-1`) | works unprivileged on **spr/emr** (not thor/Broadwell) |

IIO proxy validation (spr, `ib_write_bw` 8 B, RoCE, 2M msgs, sender side):

| QPs | Mmsg/s | 64B read txn/msg | read bytes/msg |
|---|---|---|---|
| 1 | 4.59 | 1.00 | 8 |
| 64 | 4.03 | 1.07 | 12 |
| 128 | 2.38 | 1.96 | 70 |
| 4096 | 2.41 | 2.11 | 78 |

Knee between 64 and 128 QPs: ~one extra ~64 B context fetch per message and
~2x message-rate loss. The proxy measures **all** non-payload NIC reads (QPC
and MKey/MTT cache misses together), local node only; labelled as an estimate.

## Step 0 - Tests and validation (written first)

1. **Parser unit tests** (offline, no NIC): fixtures for `rdma -j res show qp`
   JSON and for perf/IIO sysfs `format/*` files; assert QP counts by type
   (RC/DCI/DCT/UD) and correct `perf_event_attr.config` encoding.
2. **Reference test** (`tools/icmc/qpsweep.sh`): perftest sweep above; must show
   ~0 excess reads at <=64 QPs and ~1/msg at >=128 before a2a data is trusted.
3. **QP-count sanity** in the benchmark: `UCX_TLS=rc_x` with N ranks must give
   >= N-1 RC QPs per rank; `UCX_TLS=dc_x UCX_DC_MLX5_NUM_DCI=k` must give k DCIs.
   Warn loudly on mismatch.
4. **Noise floor**: IIO deltas over an idle window (barriers only), reported and
   subtracted.
5. **Overhead check**: latency/BW with counters on vs. off must agree within the
   run's variance (all reads happen outside the timed region).
6. **ppn regression**: with ppn>1, node counters must not scale with ppn (see
   step 1).

## Step 1 - Fix existing counter aggregation bug

Today every rank reads device-wide sysfs counters and they are summed with
`MPI_Allreduce`, overcounting by ppn. Split with
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`; only the node leader reads NIC/IIO
counters; reduce across leaders (report sum and per-node max).

## Step 2 - `common/nic_counters.{c,h}` (pluggable backends)

Interface: `open(dev)`, `snapshot()`, `delta()`, `names()`.
- `sysfs`: existing CNP/ECN counters moved here.
- `iio` (`common/iio_pmu.{c,h}`): `perf_event_open` on `uncore_iio_N`
  (type from `/sys/bus/event_source/devices/uncore_iio_N/type`, config built
  from `format/*`), events `event=0x84,umask=0x4` (64 B read txns) and
  `event=0x83,umask=0x4` (4 B read units), `ch_mask=0xff,fc_mask=0x7`, opened
  on one CPU of the NIC's NUMA socket.
  PMU<->NIC mapping isn't in sysfs on this kernel: count all IIO PMUs on the
  socket; identify the dominant one during warm-up and report it.
  Unavailable (e.g. thor) -> columns print `n/a`, benchmark still runs.

## Step 3 - `common/qp_count.{c,h}` (node-wide)

RDMA netlink `RDMA_NLDEV_CMD_RES_QP_GET` (fallback `popen("rdma -j res show
qp")`). Node leader records node-wide QPs on the NIC by type; each rank also
records its own pid's QPs (for the UCC-vs-runtime breakdown: snapshot before/
after `ucc_team_create` and after the first collective). Sampled after SKIP
warm-up for every message size (UCX creates eps/DCIs lazily; lanes differ for
eager vs. rndv). Measured, not computed: OMPI `pml ucx`, OSHMEM `spml ucx` and
UCC `tl/ucp` each own UCP workers/QPs.

## Step 4 - Benchmark changes (`ucc_bench_a2a_mpi.c`)

- Options: `-c <dev>` (device, unchanged), `-C sysfs,iio` (backends),
  `-Q` (QP sampling).
- One counter window per message size around all timed iterations (not per
  iteration).
- Derived metrics per size:
  - `nic_msgs` = (npes-1) x ceil(msg_bytes / max_put_frag) x iter x ppn per node
  - `excess_read_B` = IIO read bytes - payload bytes read by NIC - noise floor
  - `icm_fetch_per_msg` = excess 64 B txns / `nic_msgs` (estimated miss rate)
- New stdout/CSV columns: `qp_node`, `qp_rank`, `qp_types`, `iio_rd_txn`,
  `iio_rd_bytes`, `excess_rd_B_per_msg`, `icm_fetch_per_msg`.
- `common/bench_output.h`: add `bench_csv_header_ext`/`bench_csv_row_ext`
  (extra key/value columns); existing fixed columns unchanged.

## Step 5 - Driving QP count (sweep, not the benchmark)

`scripts/sweep.py` `env_matrix`: `UCX_TLS` in {rc_x, dc_x},
`UCX_DC_MLX5_NUM_DCI` in {1,2,4,...,64}, `UCX_IB_NUM_PATHS`,
`UCX_MAX_RNDV_RAILS`, npes/ppn; pass via `-x` to HPC-X `oshrun`/`mpirun`.
Target partitions: spr, emr. `scripts/report.py`: plot `icm_fetch_per_msg`,
latency and BW vs. measured `qp_node`, one series per message size.

## Deliverables / order

1. `tools/icmc/` (DevX probe, `qpsweep.sh`) + parser unit tests
2. Step 1 bug fix
3. `nic_counters`, `iio_pmu`, `qp_count`
4. Benchmark integration + CSV extension
5. Sweep/report integration; validation run on spr and emr
