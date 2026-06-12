#!/usr/bin/env python3
"""
JANASANKHYA — Layer 3: LLM-persona reasoning + cross-validation
===============================================================

The structural priors in Layer 2 impute "soft" variables (rights awareness, AI
usability) from population statistics. Layer 3 derives the SAME variables a
second, independent way — by simulating the worker as an LLM persona, having
that persona reason in the first person about its own situation, then CODING
those qualitative answers into numeric scores. This is the
"LLM-based synthetic respondent generation" route from the project plan.

Two-step method (mirrors real qualitative coding):
  1. PERSONA call — claude, prompted to *be* this specific worker, answers three
     situational questions (minimum-wage awareness, app usability, contract
     comprehension) in character.
  2. CODER call — claude, prompted as an independent rater, reads the persona's
     answers and returns numeric codes (rights_awareness 0-1,
     ai_usability 1-5, contract_comprehension 1-5) as JSON.

Validation — convergent validity:
We compare the persona-derived scores against Layer 2's structural-prior scores
for the same workers. High agreement means two independent methods converge on
the same value, which is the evidence the plan asks for. (In a full study the
gold standard would be real workers interviewed via a partner such as Jan Sahas;
here we report method-to-method convergent validity and flag that next step.)

Run:
    python3 src/layer3_personas.py [n_workers]    # default 8
"""

from __future__ import annotations

import os
import re
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_backend import ask, call_count

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
SEED = 5

OCC_PLAIN = {"casual_daily": "daily-wage labourer", "casual_piecework": "piece-rate worker",
             "short_term_contract": "short-contract worker", "regular_informal": "informal worker"}


def persona_prompt(w: dict) -> str:
    occ = OCC_PLAIN[w["employment_type"]]
    return f"""You are role-playing a real informal worker in India. Stay fully in character.

You are a {w['age']}-year-old {w['sex'].lower()} {occ} in {w['state']}.
Your home language is {w['language']}. Your highest education is {w['education']}.
You earn about Rs {int(w['monthly_income'])} a month. You have no written contract.
Your social group is {w['social_group']}.

Answer these three questions AS YOURSELF, honestly, in 2-3 sentences each:

1. The government has set a minimum wage for your work that is higher than what
   you are paid. Do you know what the minimum wage is, and would you do anything
   about being underpaid?
2. Someone hands you a smartphone with the e-Shram registration app, which is in
   formal Hindi/English. Could you complete the registration on your own? What
   would confuse you?
3. Your employer gives you a 2-page contract in formal language to sign. How much
   of it do you actually understand?"""


CODER_SYSTEM = ("You are a careful social-science research assistant who codes "
                "interview transcripts into numeric scores. Return ONLY JSON.")


def coder_prompt(persona_answers: str) -> str:
    return f"""Read this informal worker's interview answers and code them.

TRANSCRIPT:
{persona_answers}

Return ONLY a JSON object with exactly these keys:
- "rights_awareness": a number from 0.0 to 1.0 (0 = no awareness of wage rights,
  1 = fully aware and would act)
- "ai_usability": an integer 1 to 5 (1 = cannot use a phone app at all,
  5 = fully independent)
- "contract_comprehension": an integer 1 to 5 (1 = understands almost nothing of
  a written contract, 5 = understands fully)
Example: {{"rights_awareness": 0.3, "ai_usability": 2, "contract_comprehension": 1}}"""


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    df = pd.read_csv(os.path.join(RESULTS, "synthetic_population_imputed.csv"))
    df["monthly_income"] = (df["annual_wage_inr"] / 12).round().astype(int)
    workers = df.sample(n, random_state=SEED).reset_index(drop=True)

    print(f"Generating {n} worker personas (2 AI calls each)...")
    records = []
    for i, w in workers.iterrows():
        wd = w.to_dict()
        answers = ask(persona_prompt(wd))
        coded = parse_json(ask(coder_prompt(answers), system=CODER_SYSTEM))
        if coded is None:
            print(f"  worker {i+1}: coder JSON parse failed, skipping")
            continue
        records.append({
            "worker_id": int(wd["worker_id"]),
            "employment_type": wd["employment_type"], "state": wd["state"],
            "education": wd["education"], "language": wd["language"],
            # structural-prior (Layer 2) values:
            "prior_rights_awareness": float(wd["rights_awareness"]),
            "prior_ai_usability": int(wd["ai_usability_score"]),
            # persona-derived (Layer 3) values:
            "persona_rights_awareness": float(coded.get("rights_awareness", np.nan)),
            "persona_ai_usability": int(coded.get("ai_usability", 0) or 0),
            "persona_contract_comprehension": int(coded.get("contract_comprehension", 0) or 0),
            "persona_transcript": answers,
        })
        print(f"  worker {i+1}/{n}: prior_ai={wd['ai_usability_score']} "
              f"persona_ai={coded.get('ai_usability')}  [calls: {call_count()}]")

    out = pd.DataFrame(records)
    out.to_csv(os.path.join(RESULTS, "persona_results.csv"), index=False)

    # ── Convergent validity ──────────────────────────────────────────────────
    valid = out.dropna(subset=["persona_rights_awareness"])
    ai_within1 = float((np.abs(valid["prior_ai_usability"] - valid["persona_ai_usability"]) <= 1).mean())
    ra_mae = float(np.abs(valid["prior_rights_awareness"] - valid["persona_rights_awareness"]).mean())
    ai_corr = float(np.corrcoef(valid["prior_ai_usability"], valid["persona_ai_usability"])[0, 1]) \
        if len(valid) > 2 and valid["persona_ai_usability"].std() > 0 else float("nan")

    validation = {
        "n_personas": int(len(out)),
        "ai_usability_agreement_within_1": round(ai_within1, 3),
        "ai_usability_correlation": round(ai_corr, 3) if ai_corr == ai_corr else None,
        "rights_awareness_mae": round(ra_mae, 3),
        "note": "Convergent validity between Layer-2 structural priors and "
                "Layer-3 LLM-persona coding. Gold-standard next step: validate "
                "against real workers interviewed via a field partner.",
    }
    with open(os.path.join(RESULTS, "persona_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print("\n" + "=" * 60)
    print("CONVERGENT VALIDITY — structural priors vs persona coding")
    print("=" * 60)
    print(f"AI-usability agreement (within +/-1 on 1-5): {ai_within1:.0%}")
    print(f"AI-usability correlation:                    {validation['ai_usability_correlation']}")
    print(f"Rights-awareness mean abs error (0-1):       {ra_mae:.3f}")


if __name__ == "__main__":
    main()
