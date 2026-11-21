"""Does the configured System One model detect a contradiction between a memory claim and its cited evidence?

Planted-negation test: for memories with an ``## Evidence`` section, ask the model about the real Takeaway and about the
same Takeaway with one clause negated ("must" -> "must not", "是" -> "不是", ...), against the memory's own evidence
quotes. A model that can verify memories scores the negated claim lower on "do the quotes support it" and higher on
"does it contradict the quotes" in (nearly) every pair. Chance level is 50% of pairs.

Result on StartLux-Decision-v1.1-4B-Instant-MLX-4bit (12 pairs, 2026-11-21): support 6/12, contradict 7/12, i.e. chance.
A Dreamer "transcript contradiction" check built on that model would produce noise, so none is shipped. Re-run this
against any new first endpoint (e.g. a remote Jev / Karpo node) before building such a check; require a clear majority
(>= 90% of pairs in the right direction) over >= 30 pairs.

Run against a snapshot, never the live DB:
    python - <<'EOF'
    import sqlite3; s = sqlite3.connect("file:/Users/me/synapse/synapse.db?mode=ro", uri=True)
    d = sqlite3.connect("/tmp/snap.db"); s.backup(d)
    EOF
    .venv/bin/python scripts/calibrate_contradiction_check.py --db /tmp/snap.db --pairs 30
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from pathlib import Path

from synapse.config import load_config
from synapse.decision import DecisionClient

_NEGATIONS = [
    (r"\bmust not\b", "must"), (r"\bmust\b", "must not"), (r"\bnever\b", "always"), (r"\balways\b", "never"),
    (r"\bdoes not\b", "does"), (r"\bis not\b", "is"), (r"\bis\b", "is not"), (r"\bare\b", "are not"),
    (r"\breturns\b", "never returns"), (r"\buses\b", "does not use"),
    (r"必须", "禁止"), (r"不能", "必须"), (r"是", "不是"), (r"会", "不会"),
]
_SUPPORT = {"support": {"type": "noul", "instructions": (
    "Do the evidence quotes support the claim? Answer yes only if the claim is consistent with what the quotes say.")}}
_CONTRADICT = {"contradict": {"type": "noul", "instructions": (
    "Does the claim say the opposite of, or conflict with, what the evidence quotes say?")}}


def _takeaway(content: str) -> str | None:
    match = re.search(r"## Takeaway\s*\n+(.+?)(?:\n## |\Z)", content, re.S)
    return match.group(1).strip() if match else None


def _quotes(content: str) -> list[str]:
    head = re.search(r"^## Evidence\s*$", content, re.M)
    return [m.group(1) for m in re.finditer(r'^- T\d+: "(.*)"\s*$', content[head.end():], re.M)] if head else []


def _negate(text: str) -> str | None:
    for pattern, replacement in _NEGATIONS:
        out, count = re.subn(pattern, replacement, text, count=1)
        if count:
            return out
    return None


def _noul(response: dict | None, key: str) -> float | None:
    answer = (response or {}).get("answers", {}).get(key)
    value = answer.get("noul") if isinstance(answer, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, type=Path, help="snapshot of synapse.db (never the live file)")
    parser.add_argument("--config", type=Path, default=Path.home() / ".synapse/config.local.toml")
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()

    client = DecisionClient(load_config(args.config).decision_model)
    if not client.enabled:
        raise SystemExit("[decision_model] is disabled or has no endpoints")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, content FROM nodes WHERE status='active' AND type='persistent' AND injected_count>=3 "
        "AND content LIKE '%## Evidence%' ORDER BY injected_count DESC LIMIT ?", (args.pairs * 4,)
    ).fetchall()

    results: list[tuple[float, float, float, float]] = []
    for node_id, content in rows:
        claim, quotes = _takeaway(content), _quotes(content)[:4]
        negated = _negate(claim) if claim else None
        if not claim or not quotes or not negated or negated == claim:
            continue
        scores = (
            _noul(client.decide({"claim": claim, "evidence_quotes": quotes}, _SUPPORT, offline=True), "support"),
            _noul(client.decide({"claim": negated, "evidence_quotes": quotes}, _SUPPORT, offline=True), "support"),
            _noul(client.decide({"claim": claim, "evidence_quotes": quotes}, _CONTRADICT, offline=True), "contradict"),
            _noul(client.decide({"claim": negated, "evidence_quotes": quotes}, _CONTRADICT, offline=True), "contradict"),
        )
        if None in scores:
            continue
        results.append(scores)  # type: ignore[arg-type]
        print("support orig/neg %.3f/%.3f  contradict orig/neg %.3f/%.3f  %s" % (*scores, node_id[:60]), flush=True)
        if len(results) >= args.pairs:
            break

    n = len(results)
    if not n:
        raise SystemExit("no usable pairs")
    support_ok = sum(r[1] < r[0] for r in results)
    contradict_ok = sum(r[3] > r[2] for r in results)
    print(f"\npairs: {n}")
    print(f"support:    negated scored lower in {support_ok}/{n} ({support_ok / n:.0%}); median {statistics.median(r[0] for r in results):.3f} -> {statistics.median(r[1] for r in results):.3f}")
    print(f"contradict: negated scored higher in {contradict_ok}/{n} ({contradict_ok / n:.0%}); median {statistics.median(r[2] for r in results):.3f} -> {statistics.median(r[3] for r in results):.3f}")
    print("chance level is 50%; a usable verifier needs >= 90% over >= 30 pairs")


if __name__ == "__main__":
    main()
