#!/usr/bin/env python3
"""Write a tiny synthetic SQLite database for the LLM pipeline demo."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

REPORTS = [
    # Positive, definitive, classic MF
    (
        "P001",
        "SYN-MF-01",
        "2020-03-12",
        "surgical",
        "Persistent patches on the trunk.",
        "Skin, left trunk, punch biopsy, photo:\nDiagnosis: mycosis fungoides, patch/plaque stage, without large-cell transformation.",
    ),
    # Positive, likely, folliculotropic
    (
        "P002",
        "SYN-FMF-01",
        "2020-04-01",
        "surgical",
        "Follicular-based papules on the face.",
        "Skin, right cheek, punch biopsy, photo:\nThe findings are consistent with folliculotropic mycosis fungoides with follicular mucinosis.",
    ),
    # Positive, possible, CD30
    (
        "P003",
        "SYN-CD30-01",
        "2020-05-20",
        "surgical",
        "Waxing and waning papules.",
        "Skin, left thigh, punch biopsy, photo:\nThe findings are suggestive of lymphomatoid papulosis. Cannot exclude a CD30-positive T-cell lymphoproliferative disorder.",
    ),
    # Negative, definitive, spongiotic
    (
        "P004",
        "SYN-ECZ-01",
        "2019-11-02",
        "surgical",
        "Pruritic plaques, favor eczema.",
        "Skin, left arm, punch biopsy, photo:\nDiagnosis: spongiotic dermatitis, consistent with eczematous dermatitis.",
    ),
    # Negative, likely, lichenoid
    (
        "P005",
        "SYN-LP-01",
        "2021-01-15",
        "surgical",
        "Violaceous papules on the wrist.",
        "Skin, right wrist, punch biopsy, photo:\nThe findings are consistent with lichenoid dermatitis, including lichen planus.",
    ),
    # Negative, possible, interface/lupus
    (
        "P006",
        "SYN-LE-01",
        "2021-06-08",
        "surgical",
        "Photodistributed plaques.",
        "Skin, left cheek, punch biopsy, photo:\nInterface dermatitis; cannot exclude cutaneous lupus erythematosus.",
    ),
    # Multi-part case: two sites, same specimen family conceptually as two specimen_ids sharing patient? 
    # Methods: multi-part = slides sharing specimen ID and part description.
    # Use one specimen with two parts via two slide rows; report mentions both sites.
    (
        "P007",
        "SYN-MULTI-01",
        "2022-02-10",
        "surgical",
        "Two biopsies submitted.",
        "A. Skin, left back, punch biopsy, photo:\nB. Skin, right back, punch biopsy, photo:\nDiagnosis: mycosis fungoides involving both sites.",
    ),
    # Molecular companion for P001 within 30 days
    (
        "P001",
        "SYN-MOL-01",
        "2020-03-18",
        "molecular",
        None,
        "TCR gene rearrangement: clonal T-cell population detected, supporting the morphologic impression of mycosis fungoides.",
    ),
    # Neither
    (
        "P008",
        "SYN-AK-01",
        "2018-07-22",
        "surgical",
        "Rough patch on the helix.",
        "Skin, left ear, shave biopsy, photo:\nDiagnosis: actinic keratosis. No lymphoproliferative or inflammatory dermatosis is identified.",
    ),
    # Psoriasiform
    (
        "P009",
        "SYN-PSO-01",
        "2022-08-01",
        "surgical",
        "Scaly plaques on elbows.",
        "Skin, right elbow, punch biopsy, photo:\nThe findings are consistent with psoriasiform dermatitis.",
    ),
    # Sezary-ish
    (
        "P010",
        "SYN-SS-01",
        "2023-01-09",
        "surgical",
        "Erythroderma.",
        "Skin, abdomen, punch biopsy, photo:\nThe findings are consistent with Sezary syndrome in the appropriate clinical context.",
    ),
    # Malformed-output specimen (mock backend returns broken JSON)
    (
        "P011",
        "SYN-MALFORMED-01",
        "2015-03-03",
        "surgical",
        "Rule out cutaneous lymphoma.",
        "Skin, left arm, punch biopsy, photo:\nAtypical lymphoid infiltrate; cannot exclude mycosis fungoides.",
    ),
    # Extra negative folliculitis
    (
        "P012",
        "SYN-FOL-01",
        "2017-05-14",
        "surgical",
        "Pustules on the scalp.",
        "Skin, scalp, punch biopsy, photo:\nDiagnosis: neutrophilic folliculitis, bacterial folliculitis favored.",
    ),
]

SLIDES = [
    ("SYN-MF-01", "slide_mf_a", "Skin, left trunk, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-MF-01", "slide_mf_b", "Skin, left trunk, punch biopsy, photo", "H&E recut", "20x", None),
    ("SYN-FMF-01", "slide_fmf_a", "Skin, right cheek, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-CD30-01", "slide_cd30_a", "Skin, left thigh, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-ECZ-01", "slide_ecz_a", "Skin, left arm, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-ECZ-01", "slide_ecz_b", "Skin, left arm, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-LP-01", "slide_lp_a", "Skin, right wrist, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-LE-01", "slide_le_a", "Skin, left cheek, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-MULTI-01", "slide_multi_a", "Skin, left back, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-MULTI-01", "slide_multi_b", "Skin, right back, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-AK-01", "slide_ak_a", "Skin, left ear, shave biopsy, photo", "H&E", "20x", None),
    ("SYN-PSO-01", "slide_pso_a", "Skin, right elbow, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-SS-01", "slide_ss_a", "Skin, abdomen, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-MALFORMED-01", "slide_bad_a", "Skin, left arm, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-FOL-01", "slide_fol_a", "Skin, scalp, punch biopsy, photo", "H&E", "20x", None),
    # 40x companion that should cause a specimen to be dropped if we added one; keep a decoy unused specimen
    ("SYN-DROP40-01", "slide_drop_20", "Skin, left arm, punch biopsy, photo", "H&E", "20x", None),
    ("SYN-DROP40-01", "slide_drop_40", "Skin, left arm, punch biopsy, photo", "H&E", "40x", None),
]

# Need a report for SYN-DROP40 so the join works and then 40x exclusion drops it
REPORTS.append(
    (
        "P013",
        "SYN-DROP40-01",
        "2016-01-01",
        "surgical",
        "Control case with a 40x scan.",
        "Skin, left arm, punch biopsy, photo:\nDiagnosis: spongiotic dermatitis.",
    )
)


def write_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE reports (
                patient_id TEXT NOT NULL,
                specimen_id TEXT NOT NULL,
                accession_date TEXT,
                case_type TEXT NOT NULL,
                clinical_note TEXT,
                report_text TEXT,
                PRIMARY KEY (specimen_id, case_type)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE slides (
                specimen_id TEXT NOT NULL,
                slide_id TEXT PRIMARY KEY,
                part_description TEXT,
                stain_name TEXT,
                magnification TEXT,
                wsi_path TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?)",
            REPORTS,
        )
        conn.executemany(
            "INSERT INTO slides VALUES (?, ?, ?, ?, ?, ?)",
            SLIDES,
        )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="sample.db", help="Output SQLite path")
    args = parser.parse_args()
    out = Path(args.out)
    write_db(out)
    print(f"Wrote {out.resolve()} with {len(REPORTS)} reports and {len(SLIDES)} slides")


if __name__ == "__main__":
    main()
