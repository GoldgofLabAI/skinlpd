#!/usr/bin/env python3
"""Write synthetic GeoJSON, nuclei JSON, and attention H5 for density smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def write_slide(out: Path, stem: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    geojson_dir = out / "geojson"
    attention_dir = out / "attention"
    nuclei_dir = out / "nuclei" / stem
    for d in (geojson_dir, attention_dir, nuclei_dir):
        d.mkdir(parents=True, exist_ok=True)

    patch_size = 256
    n_patches = 36
    grid = int(np.sqrt(n_patches))
    coords = np.array([[x * patch_size, y * patch_size] for y in range(grid) for x in range(grid)], dtype=np.int64)
    attention = rng.random(n_patches).astype(np.float32)
    attention[:8] += 1.0

    with h5py.File(attention_dir / f"{stem}.h5", "w") as f:
        f.create_dataset("attention", data=attention)
        dset = f.create_dataset("coords", data=coords)
        dset.attrs["patch_size_level0"] = patch_size

    max_x = int(coords[:, 0].max() + patch_size)
    max_y = int(coords[:, 1].max() + patch_size)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [max_x, 0], [max_x, max_y], [0, max_y], [0, 0]]],
                },
            }
        ],
    }
    (geojson_dir / f"{stem}.geojson").write_text(json.dumps(geojson), encoding="utf-8")

    cells = []
    for _ in range(80):
        cells.append(
            {
                "centroid": [float(rng.integers(0, max_x)), float(rng.integers(0, max_y))],
                "type": 0,
            }
        )
    payload = {
        "cells": cells,
        "type_map": {"0": "nuclei"},
        "wsi_metadata": {"base_mpp": 0.25},
    }
    (nuclei_dir / "cell_detection.json").write_text(json.dumps(payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic")
    parser.add_argument("--n-slides", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    for i in range(args.n_slides):
        write_slide(out, f"synth_{i:03d}", seed=i)
    print(f"Wrote {args.n_slides} synthetic slides under {out.resolve()}")


if __name__ == "__main__":
    main()
