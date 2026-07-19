#!/usr/bin/env python3
"""Quick sanity report from a sweep CSV — degrades gracefully without pandas."""

import argparse
import csv
import sys
from collections import defaultdict


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows: list[dict], group_col: str, metric: str = "avg_us"):
    """Return per-group min / avg / max of *metric*, including failure-only groups."""
    groups: dict[str, list[float]] = defaultdict(list)
    all_groups: set[str] = set()
    failures_by_group: dict[str, int] = defaultdict(int)

    for r in rows:
        key = r.get(group_col, "").strip() or "(empty)"
        all_groups.add(key)

        status = r.get("status", "")
        if "failure" in status:
            failures_by_group[key] += 1

        val = r.get(metric, "").strip()
        if val:
            try:
                groups[key].append(float(val))
            except ValueError:
                pass

    summary: list[dict[str, object]] = []
    for name in sorted(all_groups):
        vals = groups.get(name, [])
        n_fail = failures_by_group.get(name, 0)
        summary.append({
            "group": name,
            "runs": len(vals),
            "failures": n_fail,
            f"min_{metric}": round(min(vals), 3) if vals else "",
            f"avg_{metric}": round(sum(vals) / len(vals), 3) if vals else "",
            f"max_{metric}": round(max(vals), 3) if vals else "",
        })
    return summary


def print_table(summary: list[dict]) -> None:
    headers = list(summary[0].keys()) if summary else []
    col_w = {h: max(len(h), max((len(str(r[h])) for r in summary), default=4)) + 2
             for h in headers}

    line = "".join(h.ljust(col_w[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for row in summary:
        print("".join(str(row[h]).ljust(col_w[h]) for h in headers))


def try_plot(rows: list[dict], group_col: str, out_path: str = "sweep_report.png"):
    """Attempt to produce a box-plot / bar chart if pandas+matplotlib are available."""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        return False

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in ("avg_us", "min_us", "max_us") if c in df.columns]
    if not numeric_cols:
        return False

    grouped = df.groupby(group_col)[numeric_cols].agg(["mean", "min", "max"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in numeric_cols:
        means = [grouped[col]["mean"].iloc[i] if i < len(grouped) else 0
                 for i in range(len(grouped))]
        ax.plot(grouped.index, means, marker="o", label=col)

    ax.set_xlabel(group_col)
    ax.set_ylabel("us")
    ax.set_title("Sweep Summary")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Sanity report from sweep CSV")
    parser.add_argument("--input", required=True, help="Path to the sweep output CSV")
    parser.add_argument("--group-by", default="bench",
                        help="Column to group by (default: bench)")
    args = parser.parse_args()

    rows = read_csv(args.input)
    if not rows:
        print("No data in input file.", file=sys.stderr)
        sys.exit(1)

    summary = summarize(rows, args.group_by)
    print_table(summary)

    plotted = try_plot(rows, args.group_by)
    if plotted:
        print("\nPlot saved to sweep_report.png")
    else:
        print("(pandas/matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
