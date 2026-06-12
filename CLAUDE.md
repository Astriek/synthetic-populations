# Project: JANASANKHYA (CTGAN build)

## What This Is
A four-layer pipeline that builds a synthetic population of India's informal
workers from real IHDS-II microdata using a real CTGAN, enriches it with cited
structural priors and LLM-persona reasoning, and uses it to audit whether AI
systems give worse welfare advice in low-resource languages. This is the working
build of `Jansankhya - CTGAN plan.docx`. It supersedes the parametric generator
in `../B/src/stage2_janasankhya.py` (that was rule-based; this trains a real GAN).

## Current Status (built May 2026)
All four layers run end to end and are validated.
- Layer 0: 45,424 real informal workers extracted from IHDS-II (DS0001).
- Layer 1: real CTGAN, 300 epochs. KS 0.146, TVD 0.113, struct-diff 0.044,
  0 memorised rows. 10,000 synthetic workers.
- Layer 2: structural-prior imputation (wage-violation, rights-awareness,
  AI-usability, language) — all cited.
- Layer 3: LLM personas; AI-usability agrees with Layer 2 within ±1 for 100%.
- Layer 4 (the finding): scaled to n=50 workers × 5 languages = 250 scored
  answers, within-subject. AI welfare-advice recall: English 0.75 [0.70,0.80],
  Bhojpuri 0.62, Bengali 0.58, Hindi 0.56, Tamil 0.53 [0.48,0.58]. Every
  non-English language significantly below English (paired Wilcoxon, all p<0.01;
  Tamil p=1e-5). A 0.22 recall gap, non-overlapping CIs. Answer length flat
  across languages (not brevity).

## The Target
A genuinely novel, reproducible finding on a synthetic population that did not
exist before, presentable at a serious venue (FAccT-style). Met as PoC; scale the
audit and pin Layer-2 priors to primary sources for publication.

## Key Decisions
- Train on IHDS-II microdata (real person-level data we have); PLFS unit data not
  freely available. Informal filter = wage workers (WS13) minus Govt/PSU, age
  18-65, positive wage.
- LLM backend = local `claude` CLI (no paid API), cached on disk. Pass
  stdin=DEVNULL to avoid context bleed. Strong system prompt to make it answer as
  a general assistant, not Claude Code.
- CTGAN compresses the gender wage gap (real F:M 0.40 → synth 0.61); this is
  reported honestly and is exactly what Layer 2 corrects.

## Files and What They Do
- `src/layer0_prepare_data.py` — extract + label real informal workers
- `src/layer1_ctgan.py` — train + validate CTGAN (`--validate-only` reuses output)
- `src/layer2_structural_priors.py` — cited prior imputation
- `src/layer3_personas.py` — LLM personas + convergent validity
- `src/layer4_ai_audit.py` — the language-gap audit
- `src/llm_backend.py` — cached `claude` CLI wrapper
- `src/make_figures.py | make_dashboard.py | make_report.py` — visuals
- `src/run_pipeline.py` — one-command orchestrator (`--fast`, `--from N`)
- `dashboard/janasankhya.html` — interactive walkthrough
- `docs/JANASANKHYA_build_report.docx` — written build report
- `results/` — model, synthetic CSVs, validation/finding JSON, figures

## Open Questions / Next Steps
- Scale the audit (hundreds of workers, more languages, error bars).
- Pin Layer-2 prior figures to exact primary sources.
- Add self-employed / own-farm workers (out of scope in v1).
- Validate Layer-2/3 soft variables against real workers via a field partner.
- Optionally audit a production consumer chatbot, not just the local model.

## Notes for Claude
Real data is at `../B/data/raw` (symlinked as `data/`). Never claim CTGAN
preserves the gender gap — it compresses it; say so. Keep every Layer-2 number
traceable to a source.
