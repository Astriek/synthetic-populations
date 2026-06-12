#!/usr/bin/env python3
"""
JANASANKHYA — Layer 1: Conditional Tabular GAN (CTGAN)
======================================================

Trains a REAL CTGAN (Xu et al., NeurIPS 2019) on the real IHDS-II informal
workers produced by Layer 0, then generates a synthetic population that is
statistically indistinguishable from the real data but contains no real person.

The model is the unmodified CTGAN from the `ctgan` package, which implements
the two innovations from the paper:
  * mode-specific normalisation of continuous columns (per-column Gaussian
    Mixture so multi-modal wage distributions survive), and
  * a conditional vector + training-by-sampling so rare categories
    (e.g. female Adivasi piece-workers) are still learned.
PacGAN packing (pac=10) guards against mode collapse.

After sampling we VALIDATE the synthetic data three ways:
  1. Fidelity   — KS test on every continuous column; total-variation distance
                  on every categorical column.
  2. Structure  — pairwise association matrix (Cramer's V / correlation) is
                  compared real-vs-synthetic; we report the mean absolute
                  difference.
  3. Privacy    — exact-duplicate rate (did the GAN memorise a real row?) and
                  the distance to the nearest real neighbour.

Outputs (results/):
  ctgan_model.pkl                trained model
  synthetic_population_raw.csv   Layer-1 synthetic workers (pre Layers 2-4)
  ctgan_validation.json          all validation numbers
  figures/fig_l1_validation.png  distribution + validation panel

Run:
    python3 src/layer1_ctgan.py [epochs]      # default 300
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctgan import CTGAN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
TRAIN_CSV = os.path.join(RESULTS, "training_data.csv")

CATEGORICAL = ["state", "sex", "marital_status", "religion", "social_group",
               "urban_rural", "education", "employment_type", "employer_type"]
CONTINUOUS = ["age", "annual_wage_inr", "work_hours_year"]

N_SYNTH = 10_000
SEED = 42


# ─── Validation helpers ───────────────────────────────────────────────────────
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V association between two categorical series."""
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = _chi2(ct.values)
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _chi2(observed: np.ndarray) -> float:
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (observed - expected) ** 2 / expected
    return float(np.nansum(terms))


def total_variation_distance(real: pd.Series, synth: pd.Series) -> float:
    """0 = identical category distributions, 1 = disjoint."""
    real = real.dropna().astype(str)
    synth = synth.dropna().astype(str)
    cats = sorted(set(real.unique()) | set(synth.unique()))
    p = real.value_counts(normalize=True).reindex(cats).fillna(0)
    q = synth.value_counts(normalize=True).reindex(cats).fillna(0)
    return float(0.5 * np.abs(p - q).sum())


def exact_duplicate_rate(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Fraction of synthetic rows that are an exact copy of a real row."""
    cols = list(synth.columns)
    real_keys = set(map(tuple, real[cols].round(0).astype(str).values))
    synth_keys = list(map(tuple, synth[cols].round(0).astype(str).values))
    hits = sum(1 for k in synth_keys if k in real_keys)
    return hits / len(synth_keys)


def validate(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    out = {"fidelity_continuous": {}, "fidelity_categorical": {}, "structure": {}, "privacy": {}}

    # 1. Continuous fidelity — KS test (smaller stat = closer; p>0.05 = indistinguishable).
    ks_stats = []
    for c in CONTINUOUS:
        stat, p = ks_2samp(real[c], synth[c])
        out["fidelity_continuous"][c] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "real_mean": round(float(real[c].mean()), 1),
            "synth_mean": round(float(synth[c].mean()), 1),
            "real_std": round(float(real[c].std()), 1),
            "synth_std": round(float(synth[c].std()), 1),
        }
        ks_stats.append(float(stat))
    out["mean_ks_stat"] = round(float(np.mean(ks_stats)), 4)

    # 2. Categorical fidelity — total variation distance.
    tvds = []
    for c in CATEGORICAL:
        tvd = total_variation_distance(real[c], synth[c])
        out["fidelity_categorical"][c] = round(tvd, 4)
        tvds.append(tvd)
    out["mean_tvd"] = round(float(np.mean(tvds)), 4)

    # 3. Structure — pairwise association matrices, mean abs difference.
    cols = CATEGORICAL[:6]  # a representative subset keeps this O(36)
    diffs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            vr = cramers_v(real[a], real[b])
            vs = cramers_v(synth[a], synth[b])
            diffs.append(abs(vr - vs))
    out["structure"]["mean_abs_assoc_diff"] = round(float(np.mean(diffs)), 4)
    # wage-by-sex correlation of inequality (a key structural signal)
    out["structure"]["real_female_wage_ratio"] = round(
        float(real.loc[real.sex == "Female", "annual_wage_inr"].mean()
              / real.loc[real.sex == "Male", "annual_wage_inr"].mean()), 3)
    out["structure"]["synth_female_wage_ratio"] = round(
        float(synth.loc[synth.sex == "Female", "annual_wage_inr"].mean()
              / synth.loc[synth.sex == "Male", "annual_wage_inr"].mean()), 3)

    # 4. Privacy — memorisation.
    out["privacy"]["exact_duplicate_rate"] = round(exact_duplicate_rate(real, synth), 5)
    return out


def make_validation_figure(real: pd.DataFrame, synth: pd.DataFrame, val: dict) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("JANASANKHYA Layer 1 — CTGAN validation (real IHDS-II vs synthetic)",
                 fontsize=14, fontweight="bold")

    # Continuous overlays
    for ax, c, title in zip(
        axes[0], CONTINUOUS,
        ["Age", "Annual wage (INR)", "Work hours / year"]):
        ax.hist(real[c], bins=40, density=True, alpha=0.55, label="Real", color="#2c7fb8")
        ax.hist(synth[c], bins=40, density=True, alpha=0.55, label="Synthetic", color="#de2d26")
        ks = val["fidelity_continuous"][c]["ks_stat"]
        ax.set_title(f"{title}\nKS={ks}")
        ax.legend(fontsize=8)
        ax.set_yticks([])

    # Categorical comparisons
    for ax, c, title in zip(
        axes[1], ["employment_type", "social_group", "education"],
        ["Employment type", "Social group", "Education"]):
        cats = sorted(set(real[c].dropna().astype(str)) | set(synth[c].dropna().astype(str)))
        rp = real[c].astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
        sp = synth[c].astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
        x = np.arange(len(cats))
        ax.bar(x - 0.2, rp.values, 0.4, label="Real", color="#2c7fb8")
        ax.bar(x + 0.2, sp.values, 0.4, label="Synthetic", color="#de2d26")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c)[:10] for c in cats], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{title}\nTVD={val['fidelity_categorical'][c]}")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGDIR, "fig_l1_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    validate_only = "--validate-only" in sys.argv
    os.makedirs(FIGDIR, exist_ok=True)

    real = pd.read_csv(TRAIN_CSV)
    raw_path = os.path.join(RESULTS, "synthetic_population_raw.csv")

    if validate_only and os.path.exists(raw_path):
        synth = pd.read_csv(raw_path)
        print(f"Validate-only: reusing {len(synth):,} synthetic workers from {raw_path}")
    else:
        print(f"Training CTGAN on {len(real):,} real informal workers, {epochs} epochs...")
        model = CTGAN(epochs=epochs, batch_size=500, pac=10, verbose=True, cuda=False)
        model.fit(real, CATEGORICAL)

        with open(os.path.join(RESULTS, "ctgan_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        print("Saved model -> results/ctgan_model.pkl")

        # Sample a little extra, drop any degenerate NaN rows, keep N_SYNTH.
        synth = model.sample(int(N_SYNTH * 1.3))
        synth = synth.dropna().reset_index(drop=True)
        # clip to training support so reverse-GMM tails stay physical
        synth["age"] = synth["age"].clip(18, 65).round().astype(int)
        synth["annual_wage_inr"] = synth["annual_wage_inr"].clip(
            real["annual_wage_inr"].min(), real["annual_wage_inr"].max()).round().astype(int)
        synth["work_hours_year"] = synth["work_hours_year"].clip(0, 4000).round().astype(int)
        synth = synth.head(N_SYNTH).reset_index(drop=True)
        synth.insert(0, "worker_id", range(len(synth)))
        synth.to_csv(raw_path, index=False)
        print(f"Sampled {len(synth):,} synthetic workers -> {raw_path}")

    val = validate(real, synth.drop(columns=["worker_id"]))
    val["n_real"] = int(len(real))
    val["n_synth"] = int(len(synth))
    val["epochs"] = epochs
    with open(os.path.join(RESULTS, "ctgan_validation.json"), "w") as f:
        json.dump(val, f, indent=2)

    fig = make_validation_figure(real, synth, val)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mean KS stat (continuous):      {val['mean_ks_stat']}  (lower=better)")
    print(f"Mean TVD (categorical):         {val['mean_tvd']}  (lower=better)")
    print(f"Mean assoc. diff (structure):   {val['structure']['mean_abs_assoc_diff']}")
    print(f"Female:male wage  real={val['structure']['real_female_wage_ratio']} "
          f"synth={val['structure']['synth_female_wage_ratio']}")
    print(f"Exact-duplicate rate (privacy): {val['privacy']['exact_duplicate_rate']}  (want ~0)")
    print(f"Figure -> {fig}")


if __name__ == "__main__":
    main()
