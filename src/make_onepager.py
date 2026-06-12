#!/usr/bin/env python3
"""
JANASANKHYA — client one-pager
==============================

Builds a single slide-ready poster (docs/JANASANKHYA_onepager.png and .pdf):
thesis, the four layers, the headline KPIs, and the finding with confidence
intervals — all on one 16:9 page. Drop it into a deck or print it.

Run after the pipeline (reads the result JSONs).
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT_PNG = os.path.join(ROOT, "docs", "JANASANKHYA_onepager.png")
OUT_PDF = os.path.join(ROOT, "docs", "JANASANKHYA_onepager.pdf")

NAVY = "#034e7b"
MID = "#2b8cbe"
RED = "#de2d26"


def jload(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def build():
    meta = jload("training_data_meta.json")
    val = jload("ctgan_validation.json")
    fnd = jload("audit_finding.json")

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.9, 0.55, 2.4],
                           width_ratios=[1.05, 1.0], hspace=0.32, wspace=0.16,
                           left=0.04, right=0.97, top=0.99, bottom=0.06)

    # ── Title band ──
    axt = fig.add_subplot(gs[0, :])
    axt.axis("off")
    axt.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                 transform=axt.transAxes, facecolor=NAVY, edgecolor="none"))
    axt.text(0.02, 0.66, "JANASANKHYA", transform=axt.transAxes, color="white",
             fontsize=30, fontweight="bold", va="center")
    axt.text(0.02, 0.24, "A synthetic population of India's informal workers — built to "
             "test whether AI serves them.",
             transform=axt.transAxes, color="#d6e6f2", fontsize=13, va="center")
    axt.text(0.985, 0.5, "real data  →  CTGAN  →  cited priors  →  AI audit",
             transform=axt.transAxes, color="white", fontsize=11, va="center",
             ha="right", style="italic")

    # ── KPI strip ──
    axk = fig.add_subplot(gs[1, :])
    axk.axis("off")
    kpis = [
        (f"{meta.get('n_rows',0):,}", "real workers", "IHDS-II microdata"),
        (f"{val.get('n_synth',0):,}", "synthetic workers", "from a real CTGAN"),
        (f"{val.get('mean_ks_stat','-')}", "mean KS", "fidelity (lower=better)"),
        (f"{val.get('privacy',{}).get('exact_duplicate_rate','-')}", "memorised rows", "no real person copied"),
        (f"{int(round((fnd.get('english_vs_lowest_gap',0))*100))} pts", "language gap", "in AI welfare advice"),
    ]
    n = len(kpis)
    for i, (v, lab, sub) in enumerate(kpis):
        x0 = i / n
        axk.add_patch(FancyBboxPatch((x0 + 0.006, 0.05), 1 / n - 0.012, 0.9,
                      boxstyle="round,pad=0,rounding_size=0.04", transform=axk.transAxes,
                      facecolor="#eef4f9", edgecolor="#cfe0ee"))
        cx = x0 + 0.5 / n
        axk.text(cx, 0.66, v, transform=axk.transAxes, ha="center", fontsize=21,
                 fontweight="bold", color=NAVY)
        axk.text(cx, 0.36, lab, transform=axk.transAxes, ha="center", fontsize=11)
        axk.text(cx, 0.15, sub, transform=axk.transAxes, ha="center", fontsize=8.5,
                 color="#7d8b99")

    # ── Left: the four layers ──
    axl = fig.add_subplot(gs[2, 0])
    axl.axis("off")
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.text(0.5, 0.98, "How it is built", ha="center", fontsize=14, fontweight="bold",
             color=NAVY)
    layers = [
        ("LAYER 1 — CTGAN", "learns the real joint distribution of 45,424 IHDS-II\n"
         "informal workers → 10,000 synthetic workers", MID),
        ("LAYER 2 — STRUCTURAL PRIORS", "imputes wage-violation, rights-awareness,\n"
         "AI-usability from cited reports (Jan Sahas, ILO, GSMA)", "#2b8cbe"),
        ("LAYER 3 — LLM PERSONAS", "each worker reasons in first person; cross-validates\n"
         "Layer 2 (agreement within ±1 for 100% of workers)", "#0570b0"),
        ("LAYER 4 — AI AUDIT", "asks each worker's welfare question to a live AI in\n"
         "5 languages; scores vs real scheme-eligibility", "#034e7b"),
    ]
    ys = [0.80, 0.575, 0.35, 0.125]
    for (tag, body, col), y in zip(layers, ys):
        axl.add_patch(FancyBboxPatch((0.04, y - 0.085), 0.92, 0.17,
                      boxstyle="round,pad=0.01,rounding_size=0.02",
                      facecolor=col, edgecolor="#222", linewidth=1))
        axl.text(0.5, y + 0.045, tag, ha="center", va="center", fontsize=11,
                 fontweight="bold", color="white")
        axl.text(0.5, y - 0.035, body, ha="center", va="center", fontsize=8.6,
                 color="white")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        axl.add_patch(FancyArrowPatch((0.5, y0 - 0.09), (0.5, y1 + 0.09),
                      arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color="#555"))

    # ── Right: the finding ──
    axr = fig.add_subplot(gs[2, 1])
    langs = fnd.get("languages", [])
    recall = [fnd["mean_recall_by_language"][l] for l in langs]
    ci = fnd.get("ci95_by_language")
    sig = fnd.get("paired_wilcoxon_vs_english", {})
    colors = [MID if l == "English" else RED for l in langs]
    yerr = None
    if ci:
        yerr = [[recall[i] - ci[l][0] for i, l in enumerate(langs)],
                [ci[l][1] - recall[i] for i, l in enumerate(langs)]]
    axr.bar(langs, recall, color=colors, yerr=yerr, capsize=5,
            error_kw=dict(ecolor="#333", lw=1.2))
    axr.set_ylim(0, 1)
    axr.set_ylabel("Welfare-scheme recall (0–1)", fontsize=11)
    nW = fnd.get("n_workers", "?")
    axr.set_title(f"THE FINDING — AI welfare advice degrades by language\n"
                  f"same worker, same question (n={nW}, 95% CI)",
                  fontsize=13, fontweight="bold", color=NAVY)
    for i, (l, v) in enumerate(zip(langs, recall)):
        top = ci[l][1] if ci else v
        axr.text(i, top + 0.03, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        if sig.get(l) and sig[l].get("p_value") is not None:
            p = sig[l]["p_value"]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            axr.text(i, top + 0.075, star, ha="center", fontsize=9, color="#444")
    if recall:
        axr.axhline(recall[0], ls="--", color=MID, alpha=0.6)
    low = fnd.get("lowest_language", "")
    gap = fnd.get("english_vs_lowest_gap", 0)
    axr.text(0.5, -0.16, f"English speakers are told about "
             f"{recall[0]*100:.0f}% of the schemes they are owed; "
             f"{low} speakers about {fnd['mean_recall_by_language'].get(low,0)*100:.0f}% "
             f"— a {gap*100:.0f}-point gap from language alone.",
             transform=axr.transAxes, ha="center", fontsize=9.5, color="#333", wrap=True)

    fig.text(0.04, 0.012, "Data: IHDS-II (ICPSR 36151).  Model: CTGAN (Xu et al., NeurIPS "
             "2019).  Audit: live Claude model, scored vs real scheme eligibility.  "
             "Layer-2 priors: Jan Sahas 2020, ILO India 2024, GSMA 2024 — editable.",
             fontsize=7.5, color="#7d8b99")

    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote one-pager -> {OUT_PNG}")
    print(f"               -> {OUT_PDF}")


if __name__ == "__main__":
    build()
