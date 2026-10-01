#!/usr/bin/env python3
"""Config-driven benchmark sweep driver — stdlib only."""

import argparse
import csv
import itertools
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# Environment variables that can be swept as independent dimensions.
# Any subset may be listed in the [env_matrix] config section (or set via
# CLI flags); unlisted vars are passed through as a single [None] dim.
ENV_DIMS = [
    "UCC_TLS",
    "UCX_TLS",
    "UCX_DC_MLX5_NUM_DCI",
    "UCX_IB_NUM_PATHS",
    "UCX_MAX_RNDV_RAILS",
]

# Extra CSV columns emitted by the ICMC-aware benchmark (bench_csv_row_ext).
ICMC_COLS = [
    "qp_node", "qp_rank", "qp_types", "iio_rd_txn", "iio_rd_bytes",
    "excess_rd_B_per_msg", "icm_fetch_per_msg",
]


def parse_size(token: str) -> int:
    """Parse a byte-size token with optional K/M suffix."""
    token = token.strip().upper()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([KMG]?)B?$', token)
    if not m:
        raise ValueError(f"Cannot parse size: {token!r}")
    value = float(m.group(1))
    suffix = m.group(2)
    if suffix == 'K':
        return int(value * 1024)
    if suffix == 'M':
        return int(value * 1048576)
    return int(value)


def load_toml(path: str) -> dict:
    """Minimal TOML parser for the sweep config format (stdlib only)."""
    cfg = {}
    current_section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Section header
            m = re.match(r'^\[([^\]]+)\]$', line)
            if m:
                current_section = m.group(1)
                cfg.setdefault(current_section, {})
                continue
            # Key-value pair
            m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            # Parse value type
            if val.startswith('["') or val.startswith("['"):
                # List of strings
                items = re.findall(r'["\']([^"\']*)["\']', val)
                parsed = items
            elif val.startswith('['):
                # List of numbers
                items = re.findall(r'\d+', val)
                parsed = [int(x) for x in items]
            elif val == 'true':
                parsed = True
            elif val == 'false':
                parsed = False
            else:
                try:
                    parsed = int(val)
                except ValueError:
                    try:
                        parsed = float(val)
                    except ValueError:
                        # String value, strip quotes if present
                        parsed = val.strip('"').strip("'")

            if current_section:
                cfg[current_section][key] = parsed
            else:
                cfg[key] = parsed
    return cfg


def env_dims_from_config(env_cfg: dict) -> dict:
    """Return {dim_name: [values]} for each swept environment dimension."""
    dims = {}
    for name in ENV_DIMS:
        if name in env_cfg:
            v = env_cfg[name]
            dims[name] = v if isinstance(v, list) else [v]
        else:
            dims[name] = [None]
    return dims


def build_launcher_cmd(
    launcher: str, nprocs: int, ppn: int | None,
    bench_path: str, msg_size: int, output_file: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Return the command list for a single benchmark invocation."""
    if launcher == "srun":
        cmd = ["srun"]
        cmd.extend(["-n", str(nprocs)])
        if ppn is not None:
            cmd.extend(["--cpus-per-task", str(ppn), "-c", str(ppn)])
    else:
        cmd = ["mpirun"]
        cmd.extend(["-np", str(nprocs)])
        if ppn is not None:
            cmd.extend(["--map-by", f"ppn:{ppn}"])

    cmd.append(bench_path)
    cmd.extend(["-s", str(msg_size), "-p", str(ppn) if ppn is not None else "-1", "-o", output_file])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def icmc_bench_args(icc_dev, icc_backends, icc_qp, icc_frag) -> list[str]:
    """Translate ICMC options into benchmark CLI flags (""/None -> omitted)."""
    args: list[str] = []
    if icc_dev:
        args += ["-c", str(icc_dev)]
        if icc_backends:
            args += ["-C", str(icc_backends)]
    if icc_qp:
        args += ["-Q"]
    if icc_frag:
        args += ["-F", str(icc_frag)]
    return args


def run_one(
    bench: str, msg_size: int, nprocs: int, ppn: int | None,
    env_vars: dict[str, object], launcher: str,
    timeout_s: int, repo_root: str,
    extra_args: list[str] | None = None,
) -> list[dict[str, object]]:
    """Execute a single benchmark run and return result row(s)."""
    ts = datetime.now(timezone.utc).isoformat()

    env = os.environ.copy()
    for k, v in env_vars.items():
        if v is not None:
            env[str(k)] = str(v)

    bench_full = os.path.join(repo_root, bench)
    ucc_tls = env_vars.get("UCC_TLS")
    ucx_tls = env_vars.get("UCX_TLS")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpf:
        output_file = tmpf.name

    base_row: dict[str, object] = {
        "timestamp": ts,
        "bench": os.path.basename(bench),
        "variant": "",
        "nprocs": nprocs,
        "ppn": ppn if ppn is not None else "",
        "ucc_tls": ucc_tls or "",
        "ucx_tls": ucx_tls or "",
        "ucx_dci": env_vars.get("UCX_DC_MLX5_NUM_DCI") or "",
        "ucx_num_paths": env_vars.get("UCX_IB_NUM_PATHS") or "",
        "ucx_rndv_rails": env_vars.get("UCX_MAX_RNDV_RAILS") or "",
        "msg_size": msg_size,
        "iters": "",
        "avg_us": "",
        "min_us": "",
        "max_us": "",
        "stddev_us": "",
        "bw_mbps": "",
        "status": "success",
        "config": f"bench={os.path.basename(bench)},msg_size={msg_size},nprocs={nprocs},"
                  f"ppn={ppn or 'any'},"
                  + ",".join(f"{k}={v}" for k, v in env_vars.items() if v is not None),
    }
    for c in ICMC_COLS:
        base_row[c] = ""

    try:
        cmd = build_launcher_cmd(launcher, nprocs, ppn, bench_full, msg_size,
                                 output_file, extra_args)
        proc = subprocess.run(
            cmd, env=env, timeout=timeout_s,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            base_row["status"] = f"failure(rc={proc.returncode})"
            return [base_row]

    except subprocess.TimeoutExpired:
        base_row["status"] = "failure(timeout)"
        return [base_row]
    except FileNotFoundError as exc:
        base_row["status"] = f"failure(not_found:{exc.filename})"
        return [base_row]
    except Exception as exc:
        base_row["status"] = f"failure({type(exc).__name__}:{exc})"
        return [base_row]

    # Parse all rows from benchmark CSV output
    try:
        with open(output_file, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if not rows:
            base_row["status"] = "failure(empty_output)"
            return [base_row]

        results = []
        for row in rows:
            r = dict(base_row)
            # Use the benchmark's actual msg_size if present
            row_msg_size = row.get("msg_size", "")
            if row_msg_size and str(row_msg_size).strip():
                try:
                    r["msg_size"] = int(row_msg_size)
                except ValueError:
                    pass
            for key in ("avg_us", "min_us", "max_us", "stddev_us"):
                val = row.get(key, "")
                if val and val.strip():
                    try:
                        r[key] = float(val)
                    except ValueError:
                        pass

            iters_val = row.get("iters") or row.get("# iters", "")
            if iters_val:
                try:
                    r["iters"] = int(iters_val)
                except ValueError:
                    try:
                        r["iters"] = float(iters_val)
                    except ValueError:
                        pass

            bw = row.get("bw_mbps", "")
            if bw and str(bw).strip():
                try:
                    r["bw_mbps"] = float(bw)
                except ValueError:
                    pass

            variant_raw = row.get("variant", "") or ""
            r["variant"] = str(variant_raw).strip()

            # Use benchmark's transport detection fields if available
            ucc_tls_val = row.get("ucc_tls", "")
            if ucc_tls_val and str(ucc_tls_val).strip():
                r["ucc_tls"] = str(ucc_tls_val).strip()
            ucx_tls_val = row.get("ucx_tls", "")
            if ucx_tls_val and str(ucx_tls_val).strip():
                r["ucx_tls"] = str(ucx_tls_val).strip()

            # ICMC extension columns (from bench_csv_row_ext)
            for c in ICMC_COLS:
                val = row.get(c, "")
                if val is not None and str(val).strip():
                    r[c] = str(val).strip()

            results.append(r)

        return results if results else [base_row]

    except Exception as exc:
        base_row["status"] = f"failure(parse:{type(exc).__name__})"
        return [base_row]
    finally:
        try:
            os.unlink(output_file)
        except OSError:
            pass


def generate_slurm_script(
    script_path: str,
    benchmarks: list[str],
    sizes: list[int],
    procs: list[int],
    ppns: list[int | None],
    env_dims: dict[str, list],
    launcher: str,
    timeout_s: int,
    repo_root: str,
    job_name: str,
    ntasks: str | None,
    ntasks_per_node: str | None,
    time_limit: str,
    partition: str | None,
    mail_type: str,
    mail_user: str,
    icc_args: str,
    output_dir: str = "slurm-output",
) -> None:
    """Generate a SLURM batch script that runs all sweep combinations."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
    ]
    if ntasks:
        lines.append(f"#SBATCH --ntasks={ntasks}")
    if ntasks_per_node:
        lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if mail_type:
        lines.append(f"#SBATCH --mail-type={mail_type}")
    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")
    lines += [
        "#SBATCH --output=%x_%A_%a.out",
        "#SBATCH --error=%x_%A_%a.err",
        "#SBATCH --time=" + time_limit,
        "",
        "set -e",
        "",
        f"REPO_ROOT='{repo_root}'",
        f"TIMEOUT={timeout_s}",
        "",
        "# Load environment modules if needed",
        "# module purge",
        "# module load ucx mpi cuda",
        "",
        f'COMBOS=(',
    ]

    combos = list(itertools.product(benchmarks, sizes, procs, ppns,
                                    *[env_dims[d] for d in ENV_DIMS]))
    for i, combo in enumerate(combos):
        bench, msg_sz, np_, ppn_val = combo[0], combo[1], combo[2], combo[3]
        env_vals = dict(zip(ENV_DIMS, combo[4:]))
        ppn_arg = str(ppn_val) if ppn_val is not None else "any"
        env_args = ",".join(f"{d}={env_vals[d]}" for d in ENV_DIMS
                            if env_vals[d] is not None)
        config_str = (f"bench={os.path.basename(bench)},msg_size={msg_sz},nprocs={np_},"
                      f"ppn={ppn_arg},{env_args or 'env=default'}")
        env_field = ",".join(str(env_vals[d]) if env_vals[d] is not None else ""
                             for d in ENV_DIMS)
        lines.append(
            f'  ("{bench}" "{msg_sz}" "{np_}" "{ppn_arg}" "{env_field}" "{config_str}")'
        )
    lines += [
        ")",
        "",
        f'INDEX=${{SLURM_ARRAY_TASK_ID:-1}}',
        f'TOTAL=${{#COMBOS[@]}}',
        "",
        'if [ "$INDEX" -gt "$TOTAL" ]; then',
        '    echo "Array task $INDEX exceeds total $TOTAL combinations"',
        '    exit 1',
        'fi',
        "",
        '# Array index is 1-based',
        'idx=$((INDEX - 1))',
        '',
        'bench="${COMBOS[$idx][0]}"',
        'msg_size="${COMBOS[$idx][1]}"',
        'nprocs="${COMBOS[$idx][2]}"',
        'ppn="${COMBOS[$idx][3]}"',
        'env_field="${COMBOS[$idx][4]}"',
        'config="${COMBOS[$idx][5]}"',
        '',
        'echo "Running [$INDEX/$TOTAL]: bench=$bench msg_size=$msg_size nprocs=$nprocs ppn=$ppn env=$env_field"',
        '',
        f'cd "$REPO_ROOT"',
        '',
        '# Set environment (comma-separated values for ' + ",".join(ENV_DIMS) + ')',
        'IFS=, read -r E1 E2 E3 E4 E5 <<< "$env_field"',
        '[ -n "$E1" ] && export UCC_TLS="$E1"',
        '[ -n "$E2" ] && export UCX_TLS="$E2"',
        '[ -n "$E3" ] && export UCX_DC_MLX5_NUM_DCI="$E3"',
        '[ -n "$E4" ] && export UCX_IB_NUM_PATHS="$E4"',
        '[ -n "$E5" ] && export UCX_MAX_RNDV_RAILS="$E5"',
        'export LAUNCHER="$launcher"',
        '',
        '# Create output directory',
        f'OUTPUT_DIR="{output_dir}"',
        'mkdir -p "$OUTPUT_DIR"',
        '',
        '# Build benchmark command',
        'BENCH_PATH="$REPO_ROOT/$bench"',
        'if ! [ -f "$BENCH_PATH" ]; then',
        '    echo "Benchmark not found: $BENCH_PATH"',
        '    exit 1',
        'fi',
        '',
        '# Run via launcher',
        'if [ "$LAUNCHER" = "srun" ]; then',
        '    LAUNCHER_CMD="srun -n $nprocs"',
        '    if [ "$ppn" != "any" ]; then',
        '        LAUNCHER_CMD="$LAUNCHER_CMD --cpus-per-task=$ppn -c $ppn"',
        '    fi',
        'else',
        '    LAUNCHER_CMD="mpirun -np $nprocs"',
        '    if [ "$ppn" != "any" ]; then',
        '        LAUNCHER_CMD="$LAUNCHER_CMD --map-by ppn:$ppn"',
        '    fi',
        'fi',
        '',
        '# Output CSV per-run',
        'OUTPUT_CSV="$OUTPUT_DIR/${bench##*/}_${msg_size}_${nprocs}_${ppn}_$(date +%Y%m%d-%H%M%S).csv"',
        '',
        '# Execute benchmark',
        'set +e',
        f'$LAUNCHER_CMD "$BENCH_PATH" -s "$msg_size" -p "$ppn" {icc_args} -o "$OUTPUT_CSV" 2>&1 || {{',
        '    echo "Benchmark failed with rc=$?"',
        '    # Write failure row',
        '    TS=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)',
        '    echo "$TS,$bench,,,$nprocs,$ppn,$E1,$E2,$E3,$E4,$E5,$msg_size,,,failure(rc=$?),,$config" >> "$OUTPUT_DIR/sweep_results.csv"',
        '    exit 1',
        '}}',
        'set -e',
        '',
        '# Append benchmark CSV rows to sweep results',
        'if [ -f "$OUTPUT_CSV" ]; then',
        '    tail -n +2 "$OUTPUT_CSV" >> "$OUTPUT_DIR/sweep_results.csv" 2>/dev/null || true',
        'fi',
        '',
        'echo "[$INDEX/$TOTAL] Done. Output: $OUTPUT_CSV"',
        '',
    ]

    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Cartesian-product benchmark sweep driver",
    )
    parser.add_argument("--benchmarks", nargs="+",
                        help="Benchmark binaries (paths relative to repo root)")
    parser.add_argument("--msg-sizes",
                        help='Comma-separated sizes e.g. "64,256,1K,4M"')
    parser.add_argument("--nprocs",
                        help='Comma-separated process counts')
    parser.add_argument("--ppn", default=None,
                        help="Optional comma-separated PPN values")
    parser.add_argument("--ucc-tls", nargs="+",
                        help='UCC_TLS values to try (repeat flag or space-separated)')
    parser.add_argument("--ucx-tls", nargs="+",
                        help='UCX_TLS values to try (repeat flag or space-separated)')
    parser.add_argument("--ucx-dci", nargs="+",
                        help='UCX_DC_MLX5_NUM_DCI values to try')
    parser.add_argument("--ucx-num-paths", nargs="+",
                        help='UCX_IB_NUM_PATHS values to try')
    parser.add_argument("--ucx-rndv-rails", nargs="+",
                        help='UCX_MAX_RNDV_RAILS values to try')
    # ICMC benchmark passthrough options
    parser.add_argument("--icc-dev", default=None,
                        help="RDMA device for node-wide NIC counters (-c)")
    parser.add_argument("--icc-backends", default="sysfs,iio",
                        help="Counter backends (-C, default sysfs,iio)")
    parser.add_argument("--icc-qp", action="store_true",
                        help="Enable QP sampling (-Q)")
    parser.add_argument("--icc-frag", default=None,
                        help="Put fragment size for msgs-per-node math (-F)")
    parser.add_argument("--launcher", choices=["mpirun", "srun"], default="mpirun")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: sweep-<ts>.csv)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--config", "-c", default=None,
                        help="TOML config file path")
    parser.add_argument("--slurm", action="store_true",
                        help="Generate SLURM batch scripts instead of running benchmarks")
    parser.add_argument("--slurm-output-dir", default="slurm-output",
                        help="Directory for SLURM output files (default: slurm-output)")
    parser.add_argument("--slurm-job-name", default="bench-sweep",
                        help="SLURM job name prefix")
    parser.add_argument("--slurm-ntasks", default=None,
                        help="Override SLURM --ntasks")
    parser.add_argument("--slurm-ntasks-per-node", default=None,
                        help="Override SLURM --ntasks-per-node")
    parser.add_argument("--slurm-time", default="01:00:00",
                        help="SLURM time limit (default: 01:00:00)")
    parser.add_argument("--slurm-partition", default=None,
                        help="SLURM partition")
    parser.add_argument("--slurm-mail-type", default="NONE",
                        help="SLURM mail type (default: NONE)")
    parser.add_argument("--slurm-mail-user", default="",
                        help="SLURM mail address")

    args = parser.parse_args()

    # Resolve repo root from workspace location
    repo_root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(repo_root, ".git")):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent

    icc_args = " ".join(icmc_bench_args(args.icc_dev, args.icc_backends,
                                       args.icc_qp, args.icc_frag))

    # Load config from TOML file or CLI args
    if args.config:
        cfg = load_toml(args.config)
        sweep_cfg = cfg.get("sweep", {})
        env_cfg = cfg.get("env_matrix", {})

        benchmarks = sweep_cfg.get("benchmarks", None)
        sizes = [parse_size(s) for s in sweep_cfg.get("msg_sizes", [])] if isinstance(sweep_cfg.get("msg_sizes"), list) else []
        procs = sweep_cfg.get("nprocs", [])
        ppns_raw = sweep_cfg.get("ppn", [None])
        env_dims = env_dims_from_config(env_cfg)
    else:
        if not args.msg_sizes or not args.benchmarks:
            parser.error("--msg-sizes and --benchmarks are required without --config")

        benchmarks = args.benchmarks
        sizes = [parse_size(s) for s in args.msg_sizes.split(",")]
        procs = [int(p.strip()) for p in args.nprocs.split(",")] if args.nprocs else []
        ppns_raw = ([int(x.strip()) for x in args.ppn.split(",")] if args.ppn else [None])
        def _split_commas(tokens: list[str] | None) -> list[str]:
            return [t for tok in (tokens or []) for t in tok.split(",") if t.strip()]
        env_dims = env_dims_from_config({
            "UCC_TLS": _split_commas(args.ucc_tls) or [None],
            "UCX_TLS": _split_commas(args.ucx_tls) or [None],
            "UCX_DC_MLX5_NUM_DCI": _split_commas(args.ucx_dci) or [None],
            "UCX_IB_NUM_PATHS": _split_commas(args.ucx_num_paths) or [None],
            "UCX_MAX_RNDV_RAILS": _split_commas(args.ucx_rndv_rails) or [None],
        })

    ppns = ppns_raw if isinstance(ppns_raw, list) else [ppns_raw]

    fieldnames = [
        "timestamp", "bench", "variant", "nprocs", "ppn",
        "ucc_tls", "ucx_tls", "ucx_dci", "ucx_num_paths", "ucx_rndv_rails",
        "msg_size", "iters",
        "avg_us", "min_us", "max_us", "stddev_us",
        "bw_mbps", "status", "config",
    ] + ICMC_COLS

    # SLURM mode: generate batch scripts instead of running
    if args.slurm:
        slurm_dir = args.slurm_output_dir
        os.makedirs(slurm_dir, exist_ok=True)
        script_path = os.path.join(slurm_dir, f"{args.slurm_job_name}.sh")
        generate_slurm_script(
            script_path=script_path,
            benchmarks=benchmarks,
            sizes=sizes,
            procs=procs,
            ppns=ppns,
            env_dims=env_dims,
            launcher=args.launcher,
            timeout_s=args.timeout,
            repo_root=repo_root,
            job_name=args.slurm_job_name,
            ntasks=args.slurm_ntasks,
            ntasks_per_node=args.slurm_ntasks_per_node,
            time_limit=args.slurm_time,
            partition=args.slurm_partition,
            mail_type=args.slurm_mail_type,
            mail_user=args.slurm_mail_user,
            icc_args=icc_args,
            output_dir=slurm_dir,
        )
        # Write sweep_results.csv header
        results_csv = os.path.join(slurm_dir, "sweep_results.csv")
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        combos = list(itertools.product(benchmarks, sizes, procs, ppns,
                                        *[env_dims[d] for d in ENV_DIMS]))
        total = len(combos)
        print(f"Generated SLURM script: {script_path}")
        print(f"Array job with {total} tasks (1 task = 1 combination)")
        print(f"Submit with: sbatch {script_path}")
        print(f"Results will be collected in: {results_csv}")
        return

    combos = list(itertools.product(benchmarks, sizes, procs, ppns,
                                    *[env_dims[d] for d in ENV_DIMS]))

    outpath = args.output or (
        f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    total = len(combos)
    written = 0

    with open(outpath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, combo in enumerate(combos):
            bench, msg_sz, np_, ppn_val = combo[0], combo[1], combo[2], combo[3]
            env_vars = dict(zip(ENV_DIMS, combo[4:]))
            sys.stderr.write(
                f"\r[{idx + 1}/{total}] bench={os.path.basename(bench)} "
                f"msg={msg_sz} np={np_}"
            )
            rows = run_one(
                bench, msg_sz, np_, ppn_val, env_vars,
                args.launcher, args.timeout, repo_root,
                icmc_bench_args(args.icc_dev, args.icc_backends,
                               args.icc_qp, args.icc_frag),
            )
            for row in rows:
                writer.writerow(row)
                written += 1
            csvfile.flush()

    sys.stderr.write(f"\nDone. {written}/{total} runs ({written} rows) written to {outpath}\n")


if __name__ == "__main__":
    main()
