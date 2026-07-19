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


def build_launcher_cmd(
    launcher: str, nprocs: int, ppn: int | None,
    bench_path: str, msg_size: int, output_file: str,
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
    return cmd


def run_one(
    bench: str, msg_size: int, nprocs: int, ppn: int | None,
    ucc_tls: str | None, ucx_tls: str | None, launcher: str,
    timeout_s: int, repo_root: str,
) -> list[dict[str, object]]:
    """Execute a single benchmark run and return result row(s)."""
    ts = datetime.now(timezone.utc).isoformat()

    env = os.environ.copy()
    if ucc_tls is not None:
        env["UCC_TLS"] = ucc_tls
    if ucx_tls is not None:
        env["UCX_TLS"] = ucx_tls

    bench_full = os.path.join(repo_root, bench)

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
        "msg_size": msg_size,
        "iters": "",
        "avg_us": "",
        "min_us": "",
        "max_us": "",
        "stddev_us": "",
        "bw_mbps": "",
        "status": "success",
        "config": f"bench={os.path.basename(bench)},msg_size={msg_size},nprocs={nprocs},"
                  f"ppn={ppn or 'any'},ucc_tls={ucc_tls or 'default'},"
                  f"ucx_tls={ucx_tls or 'default'}",
    }

    try:
        cmd = build_launcher_cmd(launcher, nprocs, ppn, bench_full, msg_size, output_file)
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
    ucc_tls_list: list[str | None],
    ucx_tls_list: list[str | None],
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

    combos = list(itertools.product(benchmarks, sizes, procs, ppns, ucc_tls_list, ucx_tls_list))
    for i, (bench, msg_sz, np_, ppn_val, ucc_t, ucx_t) in enumerate(combos):
        ppn_arg = str(ppn_val) if ppn_val is not None else "any"
        ucc_t_arg = ucc_t if ucc_t else "default"
        ucx_t_arg = ucx_t if ucx_t else "default"
        config_str = (f"bench={os.path.basename(bench)},msg_size={msg_sz},nprocs={np_},"
                      f"ppn={ppn_arg},ucc_tls={ucc_t_arg},ucx_tls={ucx_t_arg}")
        lines.append(
            f'  ("{bench}" "{msg_sz}" "{np_}" "{ppn_arg}" "{ucc_t_arg}" "{ucx_t_arg}" "{config_str}")'
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
        'ucc_tls="${COMBOS[$idx][4]}"',
        'ucx_tls="${COMBOS[$idx][5]}"',
        'config="${COMBOS[$idx][6]}"',
        '',
        'echo "Running [$INDEX/$TOTAL]: bench=$bench msg_size=$msg_size nprocs=$nprocs ppn=$ppn ucc_tls=$ucc_tls ucx_tls=$ucx_tls"',
        '',
        f'cd "$REPO_ROOT"',
        '',
        '# Set environment',
        'export UCC_TLS="$ucc_tls"',
        'export UCX_TLS="$ucx_tls"',
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
        f'$LAUNCHER_CMD "$BENCH_PATH" -s "$msg_size" -p "$ppn" -o "$OUTPUT_CSV" 2>&1 || {{',
        '    echo "Benchmark failed with rc=$?"',
        '    # Write failure row',
        '    TS=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)',
        '    echo "$TS,$bench,,,$nprocs,$ppn,$ucc_tls,$ucx_tls,$msg_size,,,failure(rc=$?),,$config" >> "$OUTPUT_DIR/sweep_results.csv"',
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

    # Load config from TOML file or CLI args
    if args.config:
        cfg = load_toml(args.config)
        sweep_cfg = cfg.get("sweep", {})
        env_cfg = cfg.get("env_matrix", {})

        benchmarks = sweep_cfg.get("benchmarks", None)
        sizes = [parse_size(s) for s in sweep_cfg.get("msg_sizes", [])] if isinstance(sweep_cfg.get("msg_sizes"), list) else []
        procs = sweep_cfg.get("nprocs", [])
        ppns_raw = sweep_cfg.get("ppn", [None])

        ucc_tls_list = env_cfg.get("UCC_TLS", [None])
        ucx_tls_list = env_cfg.get("UCX_TLS", [None])
    else:
        if not args.msg_sizes or not args.benchmarks:
            parser.error("--msg-sizes and --benchmarks are required without --config")

        benchmarks = args.benchmarks
        sizes = [parse_size(s) for s in args.msg_sizes.split(",")]
        procs = [int(p.strip()) for p in args.nprocs.split(",")] if args.nprocs else []
        ppns_raw = ([int(x.strip()) for x in args.ppn.split(",")] if args.ppn else [None])
        ucc_tls_list = args.ucc_tls or [None]
        ucx_tls_list = args.ucx_tls or [None]

    ppns = ppns_raw if isinstance(ppns_raw, list) else [ppns_raw]

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
            ucc_tls_list=ucc_tls_list,
            ucx_tls_list=ucx_tls_list,
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
            output_dir=slurm_dir,
        )
        # Write sweep_results.csv header
        results_csv = os.path.join(slurm_dir, "sweep_results.csv")
        fieldnames = [
            "timestamp", "bench", "variant", "nprocs", "ppn",
            "ucc_tls", "ucx_tls", "msg_size", "iters",
            "avg_us", "min_us", "max_us", "stddev_us",
            "bw_mbps", "status", "config",
        ]
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        combos = list(itertools.product(
            benchmarks, sizes, procs, ppns,
            ucc_tls_list, ucx_tls_list,
        ))
        total = len(combos)
        print(f"Generated SLURM script: {script_path}")
        print(f"Array job with {total} tasks (1 task = 1 combination)")
        print(f"Submit with: sbatch {script_path}")
        print(f"Results will be collected in: {results_csv}")
        return

    combos = list(itertools.product(
        benchmarks, sizes, procs, ppns,
        ucc_tls_list, ucx_tls_list,
    ))

    outpath = args.output or (
        f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    fieldnames = [
        "timestamp", "bench", "variant", "nprocs", "ppn",
        "ucc_tls", "ucx_tls", "msg_size", "iters",
        "avg_us", "min_us", "max_us", "stddev_us",
        "bw_mbps", "status", "config",
    ]

    total = len(combos)
    written = 0

    with open(outpath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (bench, msg_sz, np_, ppn_val, ucc_t, ucx_t) in enumerate(combos):
            sys.stderr.write(
                f"\r[{idx + 1}/{total}] bench={os.path.basename(bench)} "
                f"msg={msg_sz} np={np_}"
            )
            rows = run_one(
                bench, msg_sz, np_, ppn_val, ucc_t, ucx_t,
                args.launcher, args.timeout, repo_root,
            )
            for row in rows:
                writer.writerow(row)
                written += 1
            csvfile.flush()

    sys.stderr.write(f"\nDone. {written}/{total} runs ({written} rows) written to {outpath}\n")


if __name__ == "__main__":
    main()
