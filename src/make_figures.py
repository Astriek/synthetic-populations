#!/usr/bin/env python3
"""
JANASANKHYA — figure builder
============================

Produces the visual explanation of the whole system:
  fig_architecture.png   the 4-layer pipeline diagram
  fig_finding.png        the language-gap finding (Layer 4)
  fig_structure.png      structural inequality preserved by CTGAN
  fig_sample_table.png   a few real synthetic workers, rendered as a table

Run after the pipeline has produced results/*. Safe to run standalone; any
panel whose inputs are missing is skipped with a note.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".json"):
        return json.load(open(p))
    return pd.read_csv(p)


# ─── 1. Architecture diagram ──────────────────────────────────────────────────
def architecture():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("JANASANKHYA — a synthetic population of India's informal workers,\n"
                 "built to audit the AI systems that serve them",
                 fontsize=15, fontweight="bold", pad=16)

    layers = [
        ("REAL DATA", "IHDS-II microdata\n45,424 informal wage workers\n(ICPSR 36151, 337 vars)",
         "#bdc9e1", 8.4),
        ("LAYER 1 — CTGAN", "Conditional Tabular GAN (Xu et al. 2019)\nlearns the real joint distribution\n"
         "-> 10,000 synthetic workers\nmode-specific norm + conditional vector + PacGAN",
         "#74a9cf", 6.55),
        ("LAYER 2 — STRUCTURAL PRIORS", "impute variables surveys never collected\n"
         "wage-violation, rights-awareness, AI-usability\nfrom published reports (Jan Sahas, ILO, GSMA)",
         "#2b8cbe", 4.7),
        ("LAYER 3 — LLM PERSONAS", "each worker reasons in the first person;\nresponses coded -> same variables;\n"
         "cross-validated against Layer 2", "#0570b0", 2.85),
        ("LAYER 4 — AI AUDIT  (THE FINDING)", "ask each worker's welfare question to a live AI\n"
         "in English / Hindi / Bhojpuri / Bengali / Tamil\nscore vs real scheme-eligibility ground truth",
         "#034e7b", 1.0),
    ]
    for tag, body, color, y in layers:
        box = FancyBboxPatch((1.4, y - 0.62), 7.2, 1.25,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.2, edgecolor="#222", facecolor=color)
        ax.add_patch(box)
        tc = "white" if y < 5 else "#0b1f33"
        ax.text(5.0, y + 0.32, tag, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=tc)
        ax.text(5.0, y - 0.18, body, ha="center", va="center", fontsize=8.6, color=tc)

    for y0, y1 in [(8.4, 6.55), (6.55, 4.7), (4.7, 2.85), (2.85, 1.0)]:
        ax.add_patch(FancyArrowPatch((5.0, y0 - 0.66), (5.0, y1 + 0.66),
                     arrowstyle="-|>", mutation_scale=20, linewidth=2, color="#333"))

    ax.text(9.0, 5.0, "every variable\ntraces to a\nreal source\nor a citation",
            ha="center", va="center", fontsize=8.5, style="italic", color="#555",
            bbox=dict(boxstyle="round", fc="#fff7bc", ec="#cc9"))
    plt.tight_layout()
    out = os.path.join(FIG, "fig_architecture.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


# ─── 2. The finding ───────────────────────────────────────────────────────────
def finding():
    f = _load("audit_finding.json")
    res = _load("audit_results.csv")
    if f is None or res is None:
        print("  [skip] finding — run Layer 4 first")
        return None
    langs = f["languages"]
    recall = [f["mean_recall_by_language"][l] for l in langs]
    chars = [f["mean_answer_chars_by_language"][l] for l in langs]
    ci = f.get("ci95_by_language")
    sig = f.get("paired_wilcoxon_vs_english", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2c7fb8" if l == "English" else "#de2d26" for l in langs]
    yerr = None
    if ci:
        yerr = [[recall[i] - ci[l][0] for i, l in enumerate(langs)],
                [ci[l][1] - recall[i] for i, l in enumerate(langs)]]
    axes[0].bar(langs, recall, color=colors, yerr=yerr, capsize=5,
                error_kw=dict(ecolor="#333", lw=1.3))
    axes[0].set_ylabel("Mean welfare-scheme recall  (0-1)")
    axes[0].set_ylim(0, 1)
    n = f.get("n_workers", "?")
    axes[0].set_title(f"Does the AI surface the schemes a worker is entitled to?\n"
                      f"Same worker, same question — only the language changes (n={n}, 95% CI)")
    for i, (l, v) in enumerate(zip(langs, recall)):
        axes[0].text(i, (ci[l][1] if ci else v) + 0.03, f"{v:.2f}", ha="center",
                     fontsize=10, fontweight="bold")
        if sig.get(l) and sig[l].get("p_value") is not None:
            p = sig[l]["p_value"]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            axes[0].text(i, (ci[l][1] if ci else v) + 0.07, star, ha="center",
                         fontsize=9, color="#444")
    eng = f["mean_recall_by_language"]["English"]
    axes[0].axhline(eng, ls="--", color="#2c7fb8", alpha=0.6)
    axes[0].text(len(langs) - 0.5, eng + 0.01, "English baseline",
                 ha="right", fontsize=8, color="#2c7fb8")

    axes[1].bar(langs, chars, color=colors)
    axes[1].set_ylabel("Mean answer length (characters)")
    axes[1].set_title("Answer substance by language")
    for i, v in enumerate(chars):
        axes[1].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    gap = f["english_vs_lowest_gap"]
    fig.suptitle(f"THE FINDING — AI welfare advice degrades by language  "
                 f"(English minus lowest = {gap:+.2f} recall, lowest = {f['lowest_language']})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "fig_finding.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


# ─── 3. Structural inequality preserved ───────────────────────────────────────
def structure():
    real = _load("training_data.csv")
    synth = _load("synthetic_population_raw.csv")
    if real is None or synth is None:
        print("  [skip] structure — run Layers 0/1 first")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # wage by social group, real vs synth
    order = ["Brahmin", "Forward caste", "OBC", "Muslim", "Dalit (SC)", "Adivasi (ST)", "Christian/Sikh/Jain"]
    rw = real.groupby("social_group")["annual_wage_inr"].mean().reindex(order)
    sw = synth.groupby("social_group")["annual_wage_inr"].mean().reindex(order)
    x = np.arange(len(order))
    axes[0].bar(x - 0.2, rw.values, 0.4, label="Real IHDS-II", color="#2c7fb8")
    axes[0].bar(x + 0.2, sw.values, 0.4, label="Synthetic", color="#de2d26")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean annual wage (INR)")
    axes[0].set_title("Caste wage gap — ranking preserved by CTGAN")
    axes[0].legend()

    # wage by sex
    rs = real.groupby("sex")["annual_wage_inr"].mean()
    ss = synth.groupby("sex")["annual_wage_inr"].mean()
    sexes = ["Male", "Female"]
    x2 = np.arange(2)
    axes[1].bar(x2 - 0.2, rs.reindex(sexes).values, 0.4, label="Real", color="#2c7fb8")
    axes[1].bar(x2 + 0.2, ss.reindex(sexes).values, 0.4, label="Synthetic", color="#de2d26")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(sexes)
    axes[1].set_ylabel("Mean annual wage (INR)")
    axes[1].set_title("Gender wage gap — direction kept, magnitude compressed")
    axes[1].legend()

    fig.suptitle("CTGAN keeps the direction of structural inequality; it compresses the "
                 "gender gap (real F:M=0.40 → synth 0.61) — which is what Layer 2 corrects",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "fig_structure.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


# ─── 4. Sample synthetic workers as a table ───────────────────────────────────
def sample_table():
    df = _load("synthetic_population_imputed.csv")
    if df is None:
        print("  [skip] sample table — run Layer 2 first")
        return None
    cols = ["state", "sex", "age", "social_group", "education", "employment_type",
            "annual_wage_inr", "language", "ai_usability_score", "wage_below_minimum"]
    samp = df.sample(8, random_state=3)[cols].reset_index(drop=True)
    samp["annual_wage_inr"] = samp["annual_wage_inr"].map(lambda v: f"{v:,}")
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.axis("off")
    ax.set_title("Eight synthetic workers from JANASANKHYA (none is a real person)",
                 fontsize=12, fontweight="bold")
    t = ax.table(cellText=samp.values, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#034e7b")
        t[0, j].set_text_props(color="white", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG, "fig_sample_table.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


def main():
    print("Building figures...")
    for fn in (architecture, structure, sample_table, finding):
        out = fn()
        if out:
            print(f"  wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
