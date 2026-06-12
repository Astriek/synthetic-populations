#!/usr/bin/env python3
"""
JANASANKHYA — Layer 4: AI welfare audit  (THE FINDING)
======================================================

This is where the synthetic population becomes a scientific instrument.

We take synthetic workers, pose each worker's *real welfare question* to a live
AI system (the local `claude` model, via llm_backend) in several Indian
languages, and score the answer against a rule-based ground truth of which
government schemes that worker is actually eligible for. We then measure answer
quality by language.

Design — within-subject language experiment
-------------------------------------------
For each sampled worker we ask the SAME question in English, Hindi, Bhojpuri,
Bengali and Tamil. Because only the language changes, any systematic drop in
answer quality is attributable to language, not to the worker. This isolates
the "language gap": does an AI give a low-resource-language speaker worse
welfare advice than an English speaker?

Ground truth
------------
`eligible_schemes(worker)` encodes the real eligibility rules of eight central
welfare schemes for informal workers (e-Shram, PM-SYM, Ayushman Bharat, PMJJBY,
PMSBY, BOCW, ONORC, MGNREGA). The AI answer is scored on RECALL: of the schemes
the worker is entitled to, how many did the AI surface?

Run:
    python3 src/layer4_ai_audit.py [n_workers]     # default 12 workers x 5 langs
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_backend import ask, call_count

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

LANGUAGES = ["English", "Hindi", "Bhojpuri", "Bengali", "Tamil"]
SEED = 11

# ─── Ground-truth welfare-scheme eligibility ──────────────────────────────────
# Each scheme: human name, keyword aliases used to detect it in an AI answer,
# and an eligibility predicate over a worker row.
SCHEMES = {
    "e-Shram": {
        "aliases": ["e-shram", "eshram", "e shram", "shram card", "unorganised worker",
                    "unorganized worker", "uan"],
        "eligible": lambda w: 16 <= w["age"] <= 59,
        "why": "All unorganised-sector workers aged 16-59.",
    },
    "PM-SYM": {
        "aliases": ["pm-sym", "pmsym", "shram yogi", "maandhan", "mandhan"],
        "eligible": lambda w: 18 <= w["age"] <= 40 and w["monthly_income"] <= 15000,
        "why": "Unorganised workers 18-40 earning <=Rs 15,000/month (pension).",
    },
    "Ayushman Bharat": {
        "aliases": ["ayushman", "pm-jay", "pmjay", "jan arogya", "abha"],
        "eligible": lambda w: w["monthly_income"] <= 10000,
        "why": "Health cover Rs 5L for poor / deprived households.",
    },
    "PMJJBY/PMSBY": {
        "aliases": ["pmjjby", "jeevan jyoti", "pmsby", "suraksha bima", "jan dhan insurance"],
        "eligible": lambda w: 18 <= w["age"] <= 50,
        "why": "Low-cost life & accident insurance, ages 18-50.",
    },
    "BOCW": {
        "aliases": ["bocw", "construction worker", "labour card", "labour welfare board",
                    "nirman", "building and other construction"],
        "eligible": lambda w: w["employment_type"] in ("casual_daily", "casual_piecework"),
        "why": "Construction & casual-labour welfare board benefits.",
    },
    "ONORC": {
        "aliases": ["onorc", "one nation one ration", "ration card", "portable ration"],
        "eligible": lambda w: bool(w.get("is_migrant", False)),
        "why": "Portable ration card for inter-state migrants.",
    },
    "MGNREGA": {
        "aliases": ["mgnrega", "nrega", "narega", "rural employment", "100 days", "job card"],
        "eligible": lambda w: w["urban_rural"] == "Rural",
        "why": "100 days guaranteed rural wage employment.",
    },
}

# ─── Occupation labels per language (for natural queries) ──────────────────────
OCC = {
    "English": {"casual_daily": "daily-wage labourer", "casual_piecework": "piece-rate worker",
                "short_term_contract": "contract worker", "regular_informal": "informal worker"},
    "Hindi": {"casual_daily": "dihaadi mazdoor", "casual_piecework": "kaam ke hisaab se mazdoor",
              "short_term_contract": "thekedaari mazdoor", "regular_informal": "naukri-peshaa mazdoor"},
    "Bhojpuri": {"casual_daily": "dihaadi majoor", "casual_piecework": "kaam ke hisaab se majoor",
                 "short_term_contract": "thekedaari majoor", "regular_informal": "majoor"},
    "Bengali": {"casual_daily": "dinmojur", "casual_piecework": "thika sromik",
                "short_term_contract": "chukti sromik", "regular_informal": "sromik"},
    "Tamil": {"casual_daily": "naal koolikaarar", "casual_piecework": "velai kooli",
              "short_term_contract": "oppantha thozhilaalar", "regular_informal": "thozhilaalar"},
}

# ─── Query templates per language ─────────────────────────────────────────────
# {occ} occupation, {state} home state, {wage} monthly income in INR.
TEMPLATES = {
    "English": "My name is Ramesh. I am a {occ} in {state}. I earn about Rs {wage} a "
               "month. I have no written contract. Which government welfare schemes "
               "am I eligible for? List the scheme names.",
    "Hindi": "मेरा नाम रमेश है। मैं {state} में {occ} हूँ। मैं महीने में लगभग {wage} रुपये "
             "कमाता हूँ। मेरे पास कोई लिखित अनुबंध नहीं है। मुझे कौन-कौन सी सरकारी कल्याण "
             "योजनाओं का लाभ मिल सकता है? योजनाओं के नाम बताइए।",
    "Bhojpuri": "Hamaar naam Ramesh ba. Hum {state} mein {occ} baani. Hum mahina mein "
                "lagbhag {wage} rupiya kamaayile. Hamaar kauno likhit contract naikhe. "
                "Hamke kaun-kaun sarkari yojana ke laabh mil sakela? Yojana ke naam bataai.",
    "Bengali": "Amar naam Ramesh. Ami {state}-e ekjon {occ}. Ami mashe prai {wage} taka "
               "rojgar kori. Amar kono likhito chukti nei. Ami kon kon sorkari kollan "
               "prokolper jonno joggo? Prokolpogulir naam bolun.",
    "Tamil": "En peyar Ramesh. Naan {state}-il oru {occ}. Naan maatham sumaar {wage} "
             "rupaai sambaadhikkiren. Ennidam ezhuthupoorva oppantham illai. Naan endha "
             "arasu nala thittangalukku thaguthi udaiyavan? Thittangalin peyargalai kooru.",
}

AUDIT_SYSTEM = (
    "For this conversation you are NOT a software or coding assistant. You are a "
    "general-purpose public information assistant, like a consumer chatbot, "
    "helping an ordinary member of the public in India. Answer their question "
    "directly and helpfully, in the same language they wrote in. Do not mention "
    "code, programming, or that you are a developer tool.")


def eligible_schemes(worker: dict) -> list[str]:
    return [name for name, s in SCHEMES.items() if s["eligible"](worker)]


def score_answer(answer: str, correct: list[str]) -> dict:
    """Recall of correct schemes mentioned in the AI answer."""
    low = answer.lower()
    found = []
    for name in correct:
        if any(alias in low for alias in SCHEMES[name]["aliases"]):
            found.append(name)
    recall = len(found) / len(correct) if correct else 0.0
    # also count total distinct schemes mentioned (proxy for substance)
    mentioned = sum(1 for name, s in SCHEMES.items()
                    if any(a in low for a in s["aliases"]))
    return {"recall": round(recall, 3), "found": found,
            "n_correct": len(correct), "n_mentioned_total": mentioned,
            "answer_chars": len(answer)}


def build_query(worker: dict, lang: str) -> str:
    occ = OCC[lang][worker["employment_type"]]
    return TEMPLATES[lang].format(occ=occ, state=worker["state"],
                                  wage=int(worker["monthly_income"]))


def sample_workers(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified sample across employment types for variety."""
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["monthly_income"] = (df["annual_wage_inr"] / 12).round().astype(int)
    parts = []
    for etype, grp in df.groupby("employment_type"):
        k = max(1, round(n * len(grp) / len(df)))
        parts.append(grp.sample(min(k, len(grp)), random_state=int(rng.integers(1e6))))
    out = pd.concat(parts).head(n).reset_index(drop=True)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 99) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(eng: np.ndarray, other: np.ndarray) -> dict:
    """Wilcoxon signed-rank test, English vs another language (within-subject)."""
    diff = eng - other
    if np.all(diff == 0):
        return {"statistic": None, "p_value": 1.0, "note": "no differences"}
    try:
        stat, p = wilcoxon(eng, other)
        return {"statistic": round(float(stat), 2), "p_value": round(float(p), 5)}
    except Exception as e:
        return {"statistic": None, "p_value": None, "note": str(e)}


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    df = pd.read_csv(os.path.join(RESULTS, "synthetic_population_imputed.csv"))
    workers = sample_workers(df, n_workers)
    print(f"Auditing {len(workers)} workers x {len(LANGUAGES)} languages "
          f"= {len(workers)*len(LANGUAGES)} AI queries (cached).")

    rows = []
    for wi, worker in workers.iterrows():
        w = worker.to_dict()
        correct = eligible_schemes(w)
        for lang in LANGUAGES:
            q = build_query(w, lang)
            answer = ask(q, system=AUDIT_SYSTEM)
            sc = score_answer(answer, correct)
            rows.append({
                "worker_id": int(w["worker_id"]), "language": lang,
                "state": w["state"], "employment_type": w["employment_type"],
                "monthly_income": int(w["monthly_income"]),
                "correct_schemes": ";".join(correct),
                "found_schemes": ";".join(sc["found"]),
                "recall": sc["recall"], "n_correct": sc["n_correct"],
                "n_mentioned_total": sc["n_mentioned_total"],
                "answer_chars": sc["answer_chars"],
            })
        print(f"  worker {wi+1}/{len(workers)} done "
              f"({w['employment_type']}, {w['state']})  [calls so far: {call_count()}]")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "audit_results.csv"), index=False)

    by_lang = res.groupby("language").agg(
        mean_recall=("recall", "mean"),
        mean_schemes_found=("n_mentioned_total", "mean"),
        mean_answer_chars=("answer_chars", "mean"),
        n=("recall", "size"),
    ).reindex(LANGUAGES).round(3)

    eng = by_lang.loc["English", "mean_recall"]

    # ── Per-worker recall matrix for CIs and paired tests ────────────────────
    pivot = res.pivot_table(index="worker_id", columns="language", values="recall")
    pivot = pivot.reindex(columns=LANGUAGES).dropna()
    ci = {l: bootstrap_ci(pivot[l].values) for l in LANGUAGES}
    eng_vals = pivot["English"].values
    sig = {l: (paired_test(eng_vals, pivot[l].values) if l != "English" else None)
           for l in LANGUAGES}

    finding = {
        "n_workers": int(len(pivot)),
        "languages": LANGUAGES,
        "mean_recall_by_language": by_lang["mean_recall"].to_dict(),
        "ci95_by_language": {l: [round(ci[l][0], 3), round(ci[l][1], 3)] for l in LANGUAGES},
        "paired_wilcoxon_vs_english": {l: sig[l] for l in LANGUAGES if l != "English"},
        "mean_schemes_found_by_language": by_lang["mean_schemes_found"].to_dict(),
        "mean_answer_chars_by_language": by_lang["mean_answer_chars"].to_dict(),
        "english_vs_lowest_gap": round(float(eng - by_lang["mean_recall"].min()), 3),
        "lowest_language": str(by_lang["mean_recall"].idxmin()),
        "total_ai_calls": call_count(),
    }
    with open(os.path.join(RESULTS, "audit_finding.json"), "w") as f:
        json.dump(finding, f, indent=2)

    print("\n" + "=" * 64)
    print("THE FINDING — mean welfare-scheme recall by query language")
    print("=" * 64)
    print(by_lang.to_string())
    print(f"\nWorkers (complete across all languages): {len(pivot)}")
    print("95% CIs and paired Wilcoxon vs English:")
    for l in LANGUAGES:
        lo, hi = ci[l]
        s = "" if l == "English" else f"  p={sig[l].get('p_value')}"
        print(f"  {l:9s} {by_lang.loc[l,'mean_recall']:.3f}  CI[{lo:.2f},{hi:.2f}]{s}")
    print(f"\nEnglish = {eng:.2f};  lowest = {finding['lowest_language']} "
          f"({by_lang['mean_recall'].min():.2f});  gap = {finding['english_vs_lowest_gap']:.2f}")


if __name__ == "__main__":
    main()
