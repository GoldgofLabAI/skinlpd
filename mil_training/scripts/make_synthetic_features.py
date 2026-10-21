#!/usr/bin/env python3
"""Write synthetic UNI-sized H5 bags and a master CSV for MIL smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic")
    parser.add_argument("--n-patients", type=int, default=20)
    parser.add_argument("--slides-per-patient", type=int, default=2)
    parser.add_argument("--patches", type=int, default=24)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    slide_idx = 0
    for p in range(args.n_patients):
        patient_id = f"P{p:03d}"
        diagnosis = "positive" if p < args.n_patients // 2 else "negative"
        mean = 0.4 if diagnosis == "positive" else -0.4
        specimen_id = f"SPEC{p:03d}"
        part = "Skin, left arm, punch biopsy, photo"
        for s in range(args.slides_per_patient):
            stem = f"slide_{slide_idx:03d}"
            slide_name = f"{stem}.svs"
            features = rng.normal(loc=mean, scale=1.0, size=(args.patches, args.dim)).astype(np.float32)
            coords = np.stack(
                [np.arange(args.patches) * 256, np.zeros(args.patches, dtype=int)],
                axis=1,
            )
            with h5py.File(feat_dir / f"{stem}.h5", "w") as f:
                f.create_dataset("features", data=features)
                f.create_dataset("coords", data=coords)
                f.create_dataset("patch_size_level0", data=256)
            rows.append(
                {
                    "slide_name": slide_name,
                    "diagnosis": diagnosis,
                    "patient_id": patient_id,
                    "MRN": patient_id,
                    "specimen_id": specimen_id,
                    "sp_specimen_id": specimen_id,
                    "part_description": part,
                    "sl_part_description": part,
                }
            )
            slide_idx += 1

    csv_path = out / "master.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} slides for {args.n_patients} patients to {out.resolve()}")


if __name__ == "__main__":
    main()
