#!/usr/bin/env python3
"""Build a CellViT-Inference WSI filelist CSV (path, optional wsi_mpp, wsi_magnification)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Slide table with a WSI path column")
    parser.add_argument("--path-column", default="wsi_path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mpp-column", default=None)
    parser.add_argument("--magnification-column", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.path_column not in df.columns:
        raise SystemExit(f"Missing path column {args.path_column}")
    out = pd.DataFrame({"path": df[args.path_column]})
    if args.mpp_column and args.mpp_column in df.columns:
        out["wsi_mpp"] = df[args.mpp_column]
    if args.magnification_column and args.magnification_column in df.columns:
        out["wsi_magnification"] = df[args.magnification_column]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
