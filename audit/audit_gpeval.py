"""Preliminary audit of the public GP-EVAL repository (github.com/jamesl385/GP-EVAL).

Reproduces the three checks cited in the strategy report (run 18 Oct 2026):
  1. Grader-verdict parsing: compares the grader's first word (YES/NO) with the
     stored is_correct flag, because agents.py uses `"YES" in content.upper()`.
  2. Missing grader verdicts: runs that ended without a grader message.
  3. Crude diagnosis-leakage screen: flags cases whose Test_Results text contains
     the key terms of Correct_Diagnosis. This is a heuristic that over-flags;
     every flagged case needs manual clinical review before any claim is made.

Usage (from a folder containing the downloaded repo files):
    python3 -I audit_gpeval.py path/to/medical_cases.db path/to/cases_dir

The repository has no licence file. Do not redistribute its data or code
without the authors' permission; run locally and report aggregate findings.
"""
import collections
import glob
import json
import os
import re
import sqlite3
import sys

STOP = set(
    "with of and the in due to secondary acute chronic left right mild early "
    "likely infection cardiovascular disease syndrome".split()
)


def audit_db(path):
    db = sqlite3.connect(path)
    rows = db.execute(
        "select id, llm, is_correct, conversation_log, correct_diagnosis, doctor_diagnosis "
        "from case_results"
    ).fetchall()
    tally = collections.Counter()
    mismatches = []
    for row_id, llm, ok, log, cd, dd in rows:
        parts = log.split("\n\nGrader: ")
        if len(parts) < 2:
            tally[("no_grader", ok)] += 1
            continue
        first = re.match(r"\W*(YES|NO)\b", parts[-1].upper())
        verdict = first.group(1) if first else "OTHER"
        tally[(verdict, ok)] += 1
        if verdict == "NO" and ok == 1:
            mismatches.append((row_id, llm, cd, dd))
    print(f"rows: {len(rows)}")
    print("(grader verdict, stored is_correct) counts:", dict(tally))
    print(f"grader said NO but stored as correct: {len(mismatches)}")
    for m in mismatches:
        print("  ", m)
    acc = collections.defaultdict(list)
    for _, llm, ok, *_ in rows:
        acc[llm].append(ok)
    print("\nstored accuracy by model:")
    for llm, v in sorted(acc.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        print(f"  {llm:45s} n={len(v):4d} acc={sum(v) / len(v):.3f}")


def leakage_screen(cases_dir):
    n = flagged = 0
    examples = []
    for f in glob.glob(os.path.join(cases_dir, "*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            case = json.loads(line)["OSCE_Examination"]
            n += 1
            dx = re.sub(r"\(.*?\)", "", case["Correct_Diagnosis"].split(":")[-1]).lower()
            toks = [t for t in re.findall(r"[a-z]{5,}", dx) if t not in STOP]
            tests = json.dumps(case["Test_Results"]).lower()
            if toks and all(t in tests for t in toks[:2]):
                flagged += 1
                examples.append(case["Correct_Diagnosis"][:70])
    print(f"\ncases: {n}; flagged by crude leakage screen: {flagged} (needs manual review)")
    for e in examples[:15]:
        print("  ", e)


if __name__ == "__main__":
    audit_db(sys.argv[1])
    if len(sys.argv) > 2:
        leakage_screen(sys.argv[2])
