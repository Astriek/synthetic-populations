# JANASANKHYA — a synthetic population of India's informal workers

> Build a realistic population of India's informal workers from real survey data,
> then use it to test whether AI systems give them the welfare advice they are owed.

This is the working build of the CTGAN plan. It is not a sketch and not a
rule-based mock — it trains a real generative model on real microdata, validates
it, and runs a real AI audit that produces a measurable finding.

## The four layers

| Layer | What it does | Real / synthetic | Code |
|-------|--------------|------------------|------|
| 0 | Extract real informal workers from IHDS-II microdata | **real** (45,424 workers) | `src/layer0_prepare_data.py` |
| 1 | Train a CTGAN, generate 10,000 synthetic workers, validate | model is real | `src/layer1_ctgan.py` |
| 2 | Impute uncollected variables from cited published priors | synthetic, cited | `src/layer2_structural_priors.py` |
| 3 | LLM personas reason in first person; cross-validate Layer 2 | uses `claude` CLI | `src/layer3_personas.py` |
| 4 | Ask each worker's welfare question to a live AI in 5 languages | **the finding** | `src/layer4_ai_audit.py` |

## Run it

```bash
pip install -r requirements.txt        # ctgan, sdv, scipy, python-docx, pdfplumber
python3 src/run_pipeline.py            # full run (CTGAN 300 epochs, ~20 min CPU)
python3 src/run_pipeline.py --fast     # quick run (60 epochs, small audit)
python3 src/run_pipeline.py --from 2   # resume from a given layer
```

Layers 3 and 4 call the local `claude` command-line tool as their language model
(no paid API key). Every AI call is cached under `results/cache/llm/`, so re-runs
are instant and reproducible.

## What you get

- `results/training_data.csv` — the real informal workers (Layer 0)
- `results/ctgan_model.pkl` — the trained generative model
- `results/synthetic_population_imputed.csv` — 10,000 synthetic workers, all layers
- `results/ctgan_validation.json` — fidelity, structure and privacy numbers
- `results/audit_finding.json` — welfare-advice recall by language
- `results/figures/` — architecture, validation, structure and finding figures
- `dashboard/janasankhya.html` — self-contained interactive walkthrough
- `docs/JANASANKHYA_build_report.docx` — the written build report

## Data

IHDS-II individual file (India Human Development Survey, ICPSR 36151, DS0001):
204,569 people, 337 variables. The raw data lives under `data/raw/` (symlinked
from the previous working folder `B/`). Codebook value labels are quoted inline in
`src/layer0_prepare_data.py`.

## Provenance and honesty

- The generative model is the unmodified CTGAN from the `ctgan` package.
- The audited AI is the local Claude model via its CLI; a production consumer
  chatbot may score differently.
- Layer 2 priors use figures from Jan Sahas (2020), ILO India (2024), NITI Aayog
  (2022) and GSMA (2024); treat the exact numbers as editable assumptions to be
  pinned to primary sources before publication.
- The audit sample here is a proof-of-concept size; the same code scales.
