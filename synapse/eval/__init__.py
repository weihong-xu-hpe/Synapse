"""Evaluation harness for Synapse retrieval (golden-set driven).

The harness runs queries through the real retrieval pipeline against whatever
database the active config points at and reports ranking/quality metrics.
Golden sets live OUTSIDE the repo by default (e.g. ``~/.synapse/eval/``) when
they contain real queries or real node ids; ``synapse/eval/golden.example.json``
ships a tiny synthetic example for smoke tests only.
"""

from __future__ import annotations

from synapse.eval.golden import GoldenQuery, load_golden
from synapse.eval.harness import EvalReport, run_eval, write_report

__all__ = ["EvalReport", "GoldenQuery", "load_golden", "run_eval", "write_report"]
