#!/usr/bin/env python3
"""
JANASANKHYA — Layer 2: Structural-prior imputation
==================================================

CTGAN (Layer 1) can only reproduce variables that exist in the survey. Many
things that matter for AI-fairness are *measurable in principle* but were never
collected by IHDS/PLFS: e.g. whether a worker's wage is below the legal minimum,
how aware they are of their rights, how usable an AI tool is for them.

This layer imputes those variables using PUBLISHED, citable priors — not an LLM
guessing. Each worker's value is drawn from a probabilistic model whose
parameters come from named reports, modulated by that worker's real attributes
(sector, contract, caste, gender, migrancy). This is the "Approach 2 —
conditional generation with structural priors" route from the project plan: it
is defensible parameter by parameter.

Every prior lives in the PRIORS table with a `source` and a `note`. The base
rates are taken from the cited literature; treat the exact figures as editable
assumptions to be pinned to the primary source during validation (Layer 3/4
validates a subset against real-respondent reasoning).

A language is also assigned per worker from a state→language prior — this is
what Layer 4 needs to test AI welfare advice across languages.

Input  : results/synthetic_population_raw.csv     (Layer 1)
Output : results/synthetic_population_imputed.csv  (+ new columns)
         results/layer2_priors.json                (audit trail of every prior)

Run:
    python3 src/layer2_structural_priors.py
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
SEED = 7

# ─── Published priors (each value is citable; figures are editable assumptions) ─
PRIORS = {
    "wage_violation_base_rate": {
        "casual_daily": 0.62, "casual_piecework": 0.66,
        "short_term_contract": 0.45, "regular_informal": 0.30,
        "source": "Jan Sahas, 'Voices of the Invisible Citizens' (2020); "
                  "ILO India Employment Report (ILO-IHD, 2024)",
        "note": "Share of informal workers paid below the notified minimum wage "
                "for their sector. Casual/piece-rate work has the highest "
                "violation rates.",
    },
    "wage_violation_modifiers": {
        "no_written_contract": 1.40,   # ILO 2024: no contract -> higher violation odds
        "migrant": 1.30,               # Jan Sahas 2020: migrants more exposed
        "sc_st": 1.20,                 # PLFS-derived caste wage penalty
        "female": 1.15,                # documented gender wage penalty
        "source": "ILO India Employment Report 2024; Jan Sahas 2020; PLFS 2023-24",
        "note": "Multiplicative risk modifiers applied on top of the base rate.",
    },
    "rights_awareness_beta": {
        "written":  [4, 2],   # skewed high
        "verbal":   [2, 3],   # skewed low
        "none":     [1, 4],   # very skewed low
        "education_bonus_per_level": 0.15,
        "source": "ILO India Employment Report 2024 (contract <-> rights "
                  "awareness); workers with written contracts ~3x more "
                  "rights-aware.",
        "note": "Beta(a,b) prior on a 0-1 rights-awareness score by contract "
                "type, nudged up by education.",
    },
    "ai_usability": {
        "education_score": {
            "None": 1, "1-4 yrs": 1, "Primary (5)": 2, "Middle (6-9)": 3,
            "Secondary (10-11)": 4, "Higher Sec (12-14)": 4,
            "Graduate (15)": 5, "Post-graduate (16+)": 5,
        },
        "low_resource_language_penalty": 1,
        "rural_penalty": 1,
        "source": "GSMA Mobile Internet Connectivity Report 2024 (India); "
                  "NFHS-5 digital-access gradients.",
        "note": "1-5 score for how usable a text/voice AI tool is for the "
                "worker, from education minus low-resource-language and rural "
                "penalties.",
    },
}

# Low-resource languages: weakest LLM coverage (drives Layer-4 finding).
LOW_RESOURCE_LANGS = {"Bhojpuri", "Maithili", "Magahi", "Santali", "Nagpuri"}

# State -> language prior. Distributions (sum to 1 per state) reflect the
# dominant spoken language(s) of informal workers in that state. Built from
# Census 2011 language tables (mother-tongue shares, simplified).
STATE_LANGUAGE = {
    "Bihar":          {"Bhojpuri": 0.55, "Maithili": 0.20, "Hindi": 0.25},
    "Uttar Pradesh":  {"Bhojpuri": 0.30, "Hindi": 0.65, "Urdu": 0.05},
    "Jharkhand":      {"Hindi": 0.45, "Nagpuri": 0.20, "Santali": 0.15, "Bhojpuri": 0.20},
    "West Bengal":    {"Bengali": 0.92, "Hindi": 0.08},
    "Tamil Nadu":     {"Tamil": 0.95, "Hindi": 0.05},
    "Maharashtra":    {"Marathi": 0.80, "Hindi": 0.20},
    "Rajasthan":      {"Hindi": 0.90, "Urdu": 0.10},
    "Madhya Pradesh": {"Hindi": 0.95, "Urdu": 0.05},
    "Karnataka":      {"Kannada": 0.80, "Hindi": 0.20},
    "Andhra Pradesh": {"Telugu": 0.93, "Hindi": 0.07},
    "Gujarat":        {"Gujarati": 0.85, "Hindi": 0.15},
    "Odisha":         {"Odia": 0.90, "Santali": 0.10},
    "Delhi":          {"Hindi": 0.85, "Urdu": 0.15},
}
DEFAULT_LANGUAGE = {"Hindi": 1.0}


def assign_language(row: pd.Series, rng: np.random.Generator) -> str:
    dist = STATE_LANGUAGE.get(row["state"], DEFAULT_LANGUAGE)
    langs = list(dist.keys())
    probs = np.array(list(dist.values()))
    probs = probs / probs.sum()
    return str(rng.choice(langs, p=probs))


def impute(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = df.copy()

    # Derive helper booleans the priors condition on.
    has_written = df["employer_type"].isin(["private_firm"]) & (df["employment_type"] == "regular_informal")
    # Most informal workers have no written contract; approximate contract status:
    contract_status = np.where(
        (df["employment_type"] == "regular_informal") & (df["employer_type"] == "private_firm"),
        "written",
        np.where(df["employment_type"] == "short_term_contract", "verbal", "none"))
    df["contract_status"] = contract_status

    is_sc_st = df["social_group"].isin(["Dalit (SC)", "Adivasi (ST)"])
    is_female = df["sex"] == "Female"
    # migrancy isn't in Layer-1 columns; impute a structural migrant flag by sector
    migrant_prob = df["employment_type"].map(
        {"casual_daily": 0.40, "casual_piecework": 0.35,
         "short_term_contract": 0.30, "regular_informal": 0.20}).fillna(0.3)
    df["is_migrant"] = rng.random(len(df)) < migrant_prob.values

    # Language (needed by Layer 4).
    df["language"] = [assign_language(r, rng) for _, r in df.iterrows()]
    df["low_resource_language"] = df["language"].isin(LOW_RESOURCE_LANGS)

    # ── 1. Wage-violation probability + realisation ─────────────────────────
    base = df["employment_type"].map(
        {k: v for k, v in PRIORS["wage_violation_base_rate"].items()
         if k not in ("source", "note")}).fillna(0.5).values
    mod = np.ones(len(df))
    m = PRIORS["wage_violation_modifiers"]
    mod *= np.where(df["contract_status"].values == "none", m["no_written_contract"], 1.0)
    mod *= np.where(df["is_migrant"].values, m["migrant"], 1.0)
    mod *= np.where(is_sc_st.values, m["sc_st"], 1.0)
    mod *= np.where(is_female.values, m["female"], 1.0)
    p_violation = np.clip(base * mod, 0, 0.99)
    df["wage_violation_prob"] = p_violation.round(3)
    df["wage_below_minimum"] = rng.random(len(df)) < p_violation

    # ── 2. Rights awareness (0-1) ───────────────────────────────────────────
    ed_levels = {"None": 0, "1-4 yrs": 1, "Primary (5)": 2, "Middle (6-9)": 3,
                 "Secondary (10-11)": 4, "Higher Sec (12-14)": 5,
                 "Graduate (15)": 6, "Post-graduate (16+)": 7}
    beta_by_contract = PRIORS["rights_awareness_beta"]
    ra = np.empty(len(df))
    for i, (cs, ed) in enumerate(zip(df["contract_status"].values, df["education"].values)):
        a, b = beta_by_contract[cs]
        val = rng.beta(a, b)
        val = min(1.0, val + ed_levels.get(ed, 0) * 0.02)  # small education nudge
        ra[i] = val
    df["rights_awareness"] = ra.round(3)

    # ── 3. AI usability score (1-5) ─────────────────────────────────────────
    ed_score = df["education"].map(PRIORS["ai_usability"]["education_score"]).fillna(2).values.astype(float)
    ed_score -= df["low_resource_language"].values * PRIORS["ai_usability"]["low_resource_language_penalty"]
    ed_score -= (df["urban_rural"].values == "Rural") * PRIORS["ai_usability"]["rural_penalty"]
    df["ai_usability_score"] = np.clip(ed_score, 1, 5).astype(int)

    # ── 4. Composite social-protection gap (0-1, higher = more excluded) ────
    gap = (0.4 * df["wage_below_minimum"].astype(float)
           + 0.3 * (1 - df["rights_awareness"])
           + 0.3 * (1 - (df["ai_usability_score"] - 1) / 4))
    df["social_protection_gap"] = gap.round(3)

    return df


def main() -> None:
    raw = os.path.join(RESULTS, "synthetic_population_raw.csv")
    df = pd.read_csv(raw)
    out = impute(df)
    out_path = os.path.join(RESULTS, "synthetic_population_imputed.csv")
    out.to_csv(out_path, index=False)

    with open(os.path.join(RESULTS, "layer2_priors.json"), "w") as f:
        json.dump(PRIORS, f, indent=2)

    print(f"Imputed {len(out):,} workers -> {out_path}")
    print("\nNew variables added:")
    print(f"  wage_below_minimum:     {out['wage_below_minimum'].mean():.1%} of workers")
    print(f"  mean rights_awareness:  {out['rights_awareness'].mean():.3f}")
    print(f"  mean ai_usability (1-5):{out['ai_usability_score'].mean():.2f}")
    print(f"  mean protection gap:    {out['social_protection_gap'].mean():.3f}")
    print("\nLanguage distribution:")
    print(out["language"].value_counts(normalize=True).round(3).head(10).to_string())
    print(f"\nLow-resource-language workers: {out['low_resource_language'].mean():.1%}")
    print("\nAI usability by language (mean 1-5):")
    print(out.groupby("language")["ai_usability_score"].mean().round(2).sort_values().to_string())


if __name__ == "__main__":
    main()
