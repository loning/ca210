#!/usr/bin/env python3
"""
Long-run experiment entrypoint for CA210.

Runs the CI workflow and writes all artifacts into a fixed output directory
(`artifacts`). Produces a compatibility timeseries CSV with spec-aligned
columns. All messages and outputs are in English.
"""

import argparse
import os
import sys


def _write_compat_timeseries(ts_csv_path: str, out_dir: str) -> str:
    import pandas as pd  # type: ignore

    df = pd.read_csv(ts_csv_path)
    colmap = {"H_x_block3": "H3_x", "MI_x_a": "MI_xa", "budget_residual": "R"}
    for src, dst in colmap.items():
        if src in df.columns:
            df[dst] = df[src]
    cols = [c for c in ["name", "t", "H3_x", "H_a", "MI_xa", "R"] if c in df.columns]
    compat_path = os.path.join(out_dir, "ca210_timeseries_compat.csv")
    df[cols].to_csv(compat_path, index=False)
    return compat_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run CA210 experiment and generate artifacts.")
    # Fixed output directory by design
    p.add_argument("--N", type=int, default=64, help="Universe size (ring length)")
    p.add_argument("--T", type=int, default=96, help="Number of steps for timeseries")
    p.add_argument(
        "--no-compat-ts",
        dest="compat_ts",
        action="store_false",
        help="Disable writing spec-compatible timeseries CSV",
    )
    p.set_defaults(compat_ts=True)
    args = p.parse_args(argv)

    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    # Import after headless setup
    from src.ci import run_ci  # type: ignore

    res = run_ci(output_dir=out_dir, N=args.N, T=args.T)
    print("Run complete.")
    for k in ["timeseries_csv", "report_csv", "figure", "report_md"]:
        if k in res:
            print(f"{k}: {res[k]}")

    if args.compat_ts and "timeseries_csv" in res:
        compat_path = _write_compat_timeseries(res["timeseries_csv"], out_dir)
        print(f"compat_timeseries: {compat_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
