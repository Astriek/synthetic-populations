#!/usr/bin/env python3
"""
JANASANKHYA — LLM backend
=========================

A thin, cached wrapper around the local `claude` CLI so Layers 3 and 4 can call
a real language model without any paid API key. Every call is cached on disk
(keyed by a hash of the prompt + system + model) so re-runs are instant and
reproducible, and so an interrupted audit can resume for free.

This is the "real AI system" that Layer 4 audits: we send a worker's welfare
question to `claude -p` exactly as a real user would, and score the answer.

Usage:
    from llm_backend import ask
    text = ask("What welfare schemes can a Bihar construction worker get?")
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(ROOT, "results", "cache", "llm")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap; good enough to audit
_CALL_COUNT = 0


def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{h}.json")


def ask(prompt: str, system: str | None = None, model: str = DEFAULT_MODEL,
        timeout: int = 150, max_retries: int = 2) -> str:
    """Send a prompt to the local `claude` CLI; return its text reply (cached)."""
    global _CALL_COUNT
    key = json.dumps({"p": prompt, "s": system, "m": model}, sort_keys=True)
    cpath = _cache_path(key)
    if os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)["response"]

    cmd = ["claude", "-p", prompt, "--model", model]
    if system:
        cmd += ["--append-system-prompt", system]

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            # run from /tmp so the CLI doesn't load this project's context
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp",
                stdin=subprocess.DEVNULL)  # never inherit caller stdin (avoids context bleed)
            out = proc.stdout.strip()
            if out:
                _CALL_COUNT += 1
                with open(cpath, "w") as f:
                    json.dump({"prompt": prompt, "system": system,
                               "model": model, "response": out}, f, indent=2)
                return out
            last_err = proc.stderr.strip() or "empty response"
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout}s"
        time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"claude CLI failed after {max_retries+1} tries: {last_err}")


def call_count() -> int:
    return _CALL_COUNT


if __name__ == "__main__":
    # smoke test
    print("Backend smoke test:", ask("Reply with exactly one word: READY"))
