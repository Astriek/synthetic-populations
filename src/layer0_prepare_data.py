#!/usr/bin/env python3
"""
JANASANKHYA — Layer 0: Prepare real training data
==================================================

Extracts a clean table of REAL informal wage-workers from the IHDS-II
individual microdata file (ICPSR 36151, DS0001 — 204,569 people x 337 vars)
and writes it to results/training_data.csv.

This is the real data that the CTGAN in Layer 1 learns from. Nothing here is
synthetic. Every column maps to a documented IHDS-II variable; every value
label is taken from the official codebook
(36151-0001-Codebook.pdf), referenced inline.

Informal-worker definition used here
------------------------------------
We restrict to wage / salary workers (variable WS13 is non-missing — this is
the 53,465 people who reported a wage or salaried job) and then DROP the
clearly-formal public sector (WS14 == 1, "Govt/PSU"). What remains is the
informal wage workforce: casual daily labour, piecework, short contracts and
regular-but-informal private jobs. We also require working age (18-65) and a
positive recorded annual wage.

Self-employed / own-farm workers live in different IHDS sections (FM/NF) and
are out of scope for this first CTGAN; they are a documented extension.

Run:
    python3 src/layer0_prepare_data.py
"""

from __future__ import annotations

import os
import sys
import json
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IHDS_TSV = os.path.join(
    ROOT, "data", "raw", "ihds2_icpsr", "ICPSR_36151",
    "DS0001", "36151-0001-Data.tsv",
)
OUT_DIR = os.path.join(ROOT, "results")
OUT_CSV = os.path.join(OUT_DIR, "training_data.csv")
OUT_META = os.path.join(OUT_DIR, "training_data_meta.json")

# ─── Codebook value-label maps (36151-0001-Codebook.pdf) ──────────────────────
# Each map is copied verbatim from the codebook; page/variable noted in comment.

SEX = {1: "Male", 2: "Female"}  # RO3 (codebook p.~, "HQ4 2.3 Sex")

MARITAL = {  # RO6 "HQ4 2.6 Marital Status" — collapsed to 3 classes
    0: "Married", 1: "Married", 5: "Married",
    2: "Unmarried", 3: "Widowed", 4: "Separated/Divorced",
}

RELIGION = {  # ID11 "HQ3 1.11 Religion"
    1: "Hindu", 2: "Muslim", 3: "Christian", 4: "Sikh",
    5: "Buddhist", 6: "Jain", 7: "Tribal", 8: "Other", 9: "None",
}

SOCIAL_GROUP = {  # GROUPS "HQ3 1.13-15 Caste & religion"
    1: "Brahmin", 2: "Forward caste", 3: "OBC", 4: "Dalit (SC)",
    5: "Adivasi (ST)", 6: "Muslim", 7: "Christian/Sikh/Jain",
}

URBAN = {0: "Rural", 1: "Urban"}  # URBAN2011

EDUCATION = {  # EDUC7 "Education: Completed Years, 7cats"
    0: "None", 3: "1-4 yrs", 5: "Primary (5)", 8: "Middle (6-9)",
    10: "Secondary (10-11)", 12: "Higher Sec (12-14)",
    15: "Graduate (15)", 16: "Post-graduate (16+)",
}

# WS13 "HQ13 7.13 Casual -job1" → our employment_type
EMPLOYMENT_TYPE = {
    1: "casual_daily",        # Casual daily
    2: "casual_piecework",    # Casual piecework
    3: "short_term_contract",  # Contract < 1yr
    4: "regular_informal",    # Regular/permanent (kept only when employer != Govt/PSU)
}

# WS14 "HQ13 7.14 Government, NREGA, private" → employer_type
EMPLOYER_TYPE = {
    1: "govt_psu",            # excluded by the informal filter
    2: "private_firm",
    3: "private_individual",
    4: "mgnrega",
    5: "other_govt_program",
    6: "other",
}

# STATEID "State code" — full IHDS-II state map (used to give workers a real state)
STATE = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "Delhi", 8: "Rajasthan",
    9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh",
    13: "Nagaland", 14: "Manipur", 15: "Mizoram", 16: "Tripura",
    17: "Meghalaya", 18: "Assam", 19: "West Bengal", 20: "Jharkhand",
    21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Daman & Diu", 26: "Dadra & Nagar Haveli", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep",
    32: "Kerala", 33: "Tamil Nadu", 34: "Puducherry", 35: "Andaman & Nicobar",
}

# Columns we read from the 337-wide file (usecols keeps loading fast).
RAW_COLS = [
    "STATEID", "RO3", "RO5", "RO6", "ID11", "GROUPS", "URBAN2011",
    "EDUC7", "WS13", "WS14", "WSEARN", "WKHOURS", "WKDAYS", "WT",
]

# Final, human-readable training columns.
CATEGORICAL_COLS = [
    "state", "sex", "marital_status", "religion", "social_group",
    "urban_rural", "education", "employment_type", "employer_type",
]
CONTINUOUS_COLS = ["age", "annual_wage_inr", "work_hours_year"]


def load_and_filter() -> pd.DataFrame:
    """Load IHDS-II individuals and reduce to informal wage workers."""
    if not os.path.exists(IHDS_TSV):
        sys.exit(f"IHDS data not found at {IHDS_TSV}")

    print(f"Reading IHDS-II microdata ({os.path.getsize(IHDS_TSV)//1_000_000} MB)...")
    df = pd.read_csv(
        IHDS_TSV, sep="\t", usecols=RAW_COLS,
        na_values=[" ", ""], low_memory=False,
    )
    print(f"  loaded {len(df):,} individuals")

    # Coerce numerics (blanks → NaN).
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Informal wage-worker filter ──────────────────────────────────────────
    n0 = len(df)
    df = df[df["WS13"].notna()]                                   # wage/salary workers
    print(f"  WS13 valid (wage/salary workers):        {len(df):,}  (-{n0-len(df):,})")
    df = df[df["WS14"] != 1]                                      # drop Govt/PSU (formal)
    print(f"  after dropping Govt/PSU (WS14==1):        {len(df):,}")
    df = df[(df["RO5"] >= 18) & (df["RO5"] <= 65)]               # working age
    print(f"  after working-age (18-65) filter:        {len(df):,}")
    df = df[df["WSEARN"] > 0]                                     # positive recorded wage
    print(f"  after positive-wage filter:              {len(df):,}")

    # Drop rows missing any field we need for clean training.
    need = ["STATEID", "RO3", "RO5", "ID11", "GROUPS", "URBAN2011",
            "EDUC7", "WS13", "WS14", "WSEARN", "WKHOURS"]
    df = df.dropna(subset=need)
    print(f"  after dropping rows with missing fields:  {len(df):,}")
    return df


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw codes to labelled columns; clip outliers."""
    out = pd.DataFrame()
    out["state"] = df["STATEID"].map(STATE)
    out["sex"] = df["RO3"].map(SEX)
    out["marital_status"] = df["RO6"].map(MARITAL).fillna("Unmarried")
    out["religion"] = df["ID11"].map(RELIGION)
    out["social_group"] = df["GROUPS"].map(SOCIAL_GROUP)
    out["urban_rural"] = df["URBAN2011"].map(URBAN)
    out["education"] = df["EDUC7"].map(EDUCATION)
    out["employment_type"] = df["WS13"].map(EMPLOYMENT_TYPE)
    out["employer_type"] = df["WS14"].map(EMPLOYER_TYPE)

    out["age"] = df["RO5"].astype(int)
    # WSEARN is annual; clip the long tail (top 1%) so CTGAN's GMM behaves.
    wage_cap = df["WSEARN"].quantile(0.99)
    out["annual_wage_inr"] = df["WSEARN"].clip(upper=wage_cap).round().astype(int)
    # Annual work-hours; clip to a sane range.
    out["work_hours_year"] = df["WKHOURS"].clip(0, 4000).round().astype(int)

    out = out.dropna().reset_index(drop=True)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_filter()
    train = build_training_table(df)

    train.to_csv(OUT_CSV, index=False)

    meta = {
        "source": "IHDS-II individual file, ICPSR 36151 DS0001",
        "n_rows": int(len(train)),
        "categorical_columns": CATEGORICAL_COLS,
        "continuous_columns": CONTINUOUS_COLS,
        "informal_filter": "WS13 non-missing (wage workers); WS14 != Govt/PSU; "
                           "age 18-65; WSEARN > 0",
        "wage_mean_inr": float(train["annual_wage_inr"].mean()),
        "wage_median_inr": float(train["annual_wage_inr"].median()),
        "female_share": float((train["sex"] == "Female").mean()),
        "rural_share": float((train["urban_rural"] == "Rural").mean()),
        "employment_type_dist": train["employment_type"].value_counts(normalize=True).round(4).to_dict(),
        "social_group_dist": train["social_group"].value_counts(normalize=True).round(4).to_dict(),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Wrote {len(train):,} real informal workers -> {OUT_CSV}")
    print("=" * 60)
    print("\nEmployment type:")
    print(train["employment_type"].value_counts(normalize=True).round(3).to_string())
    print("\nSocial group:")
    print(train["social_group"].value_counts(normalize=True).round(3).to_string())
    print(f"\nFemale share:      {meta['female_share']:.1%}")
    print(f"Rural share:       {meta['rural_share']:.1%}")
    print(f"Mean annual wage:  Rs {meta['wage_mean_inr']:,.0f}")
    print(f"Median annual wage:Rs {meta['wage_median_inr']:,.0f}")


if __name__ == "__main__":
    main()
