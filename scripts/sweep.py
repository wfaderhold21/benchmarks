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


def build_launcher_cmd(
    launcher: str, nprocs: int, ppn: int | None,
    bench_path: str, output_file: str, timeout_s: int | None = None,
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
    cmd.extend(["-o", output_file])
    return cmd


def run_one(
    bench: str, msg_size: int, nprocs: int, ppn: int | None,
    ucc_tls: str | None, ucx_tls: str | None, launcher: str,
    timeout_s: int, repo_root: str,
) -> dict[str, object]:
    """Execute a single benchmark run and return a result row dict."""
    ts = datetime.now(timezone.utc).isoformat()

    env = os.environ.copy()
    if ucc_tls is not None:
        env["UCC_TLS"] = ucc_tls
    if ucx_tls is not None:
        env["UCX_TLS"] = ucx_tls

    bench_full = os.path.join(repo_root, bench)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpf:
        output_file = tmpf.name

    row: dict[str, object] = {
        "timestamp": ts,
        "bench": os.path.basename(bench),
        "variant": "",
        "nprocs": nprocs,
        "ppn": ppn if ppn is not None else "",
        "tls": ucc_tls or ucx_tls or "",
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
        cmd = build_launcher_cmd(launcher, nprocs, ppn, bench_full, output_file)
        proc = subprocess.run(
            cmd, env=env, timeout=timeout_s,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            row["status"] = f"failure(rc={proc.returncode})"
            return row

    except subprocess.TimeoutExpired:
        row["status"] = "failure(timeout)"
        return row
    except FileNotFoundError as exc:
        row["status"] = f"failure(not_found:{exc.filename})"
        return row
    except Exception as exc:
        row["status"] = f"failure({type(exc).__name__}:{exc})"
        return row

    # Parse benchmark CSV output
    try:
        with open(output_file, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if not rows:
            row["status"] = "failure(empty_output)"
            return row

        last = rows[-1]
        for key in ("avg_us", "min_us", "max_us", "stddev_us"):
            val = last.get(key)
            if val is not None and val.strip():
                try:
                    row[key] = float(val)
                except ValueError:
                    pass

        iters_val = last.get("iters") or last.get("# iters")
        if iters_val:
            try:
                row["iters"] = int(iters_val)
            except ValueError:
                try:
                    row["iters"] = float(iters_val)
                except ValueError:
                    pass

        bw = last.get("bw_mbps")
        if bw is not None and bw.strip():
            try:
                row["bw_mbps"] = float(bw)
            except ValueError:
                pass

        variant_raw = last.get("variant", "") or ""
        row["variant"] = variant_raw.strip()

    except Exception as exc:
        row["status"] = f"failure(parse:{type(exc).__name__})"
    finally:
        try:
            os.unlink(output_file)
        except OSError:
            pass

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Cartesian-product benchmark sweep driver",
    )
    parser.add_argument("--benchmarks", required=True, nargs="+",
                        help="Benchmark binaries (paths relative to repo root)")
    parser.add_argument("--msg-sizes", required=True,
                        help='Comma-separated sizes e.g. "64,256,1K,4M"')
    parser.add_argument("--nprocs", required=True,
                        help='Comma-separated process counts')
    parser.add_argument("--ppn", default=None,
                        help="Optional comma-selected PPN values")
    parser.add_argument("--ucc-tls", default=None, nargs="+",
                        help='UCC_TLS values to try (repeat flag or space-separated)')
    parser.add_argument("--ucx-tls", default=None, nargs="+",
                        help='UCX_TLS values to try (repeat flag or space-separated)')
    parser.add_argument("--launcher", choices=["mpirun", "srun"], default="mpirun")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: sweep-<ts>.csv)")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    # Resolve repo root from workspace location
    repo_root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(repo_root, ".git")):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent

    sizes = [parse_size(s) for s in args.msg_sizes.split(",")]
    procs = [int(p.strip()) for p in args.nprocs.split(",")]
    ppns = ([int(x.strip()) for x in args.ppn.split(",")] if args.ppn else [None])
    ucc_tls_list = args.ucc_tls or [None]
    ucx_tls_list = args.ucx_tls or [None]

    combos = list(itertools.product(
        args.benchmarks, sizes, procs, ppns,
        ucc_tls_list, ucx_tls_list,
    ))

    outpath = args.output or (
        f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    fieldnames = [
        "timestamp", "bench", "variant", "nprocs", "ppn",
        "tls", "msg_size", "iters",
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
            row = run_one(
                bench, msg_sz, np_, ppn_val, ucc_t, ucx_t,
                args.launcher, args.timeout, repo_root,
            )
            writer.writerow(row)
            csvfile.flush()
            written += 1

    sys.stderr.write(f"\nDone. {written}/{total} runs written to {outpath}\n")


if __name__ == "__main__":
    main()
