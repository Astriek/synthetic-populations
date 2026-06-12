#!/usr/bin/env python3
"""
JANASANKHYA — end-to-end pipeline runner
========================================

Runs the four layers in order, then builds the figures, dashboard and report.
Each step is a separate module so it can also be run on its own.

    Layer 0  prepare real IHDS-II training data
    Layer 1  train + validate CTGAN, sample synthetic population
    Layer 2  structural-prior imputation
    Layer 3  LLM-persona reasoning + convergent validity   (uses claude CLI)
    Layer 4  AI welfare audit -> the language-gap finding   (uses claude CLI)
    Visuals  architecture + finding figures, HTML dashboard, Word report

Usage:
    python3 src/run_pipeline.py                 # full run (CTGAN 300 epochs)
    python3 src/run_pipeline.py --fast          # CTGAN 60 epochs, small audit
    python3 src/run_pipeline.py --from 2        # resume from Layer 2
"""

from __future__ import annotations

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, os.path.join(HERE, module), *args]
    print(f"\n{'#'*70}\n# {module} {' '.join(args)}\n{'#'*70}")
    subprocess.run(cmd, check=True)


def main() -> None:
    fast = "--fast" in sys.argv
    start = 0
    if "--from" in sys.argv:
        start = int(sys.argv[sys.argv.index("--from") + 1])

    epochs = "60" if fast else "300"
    audit_n = "6" if fast else "12"
    persona_n = "4" if fast else "8"

    if start <= 0:
        run("layer0_prepare_data.py")
    if start <= 1:
        run("layer1_ctgan.py", epochs)
    if start <= 2:
        run("layer2_structural_priors.py")
    if start <= 3:
        run("layer3_personas.py", persona_n)
    if start <= 4:
        run("layer4_ai_audit.py", audit_n)
    run("make_figures.py")
    run("make_dashboard.py")
    run("make_report.py")
    run("make_process_doc.py")
    run("make_onepager.py")
    print("\nDone. See results/, dashboard/janasankhya.html, "
          "docs/JANASANKHYA_build_report.docx and docs/JANASANKHYA_process_document.docx")


if __name__ == "__main__":
    main()
