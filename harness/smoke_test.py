"""Controlled, instrumented single-case runner for GP-EVAL.

Calls the benchmark's own `process_single_case` UNMODIFIED (prompts, speaker
selection, diagnosis extraction and scoring logic are untouched). What this
wrapper adds:

  * explicit per-role model selection (the repo's defaults are retired models);
  * a log of every chat-completion call: role, requested/returned model id,
    host, latency, token usage (no headers, no keys, no message content);
  * a log of every outgoing HTTP request (host + path only) to catch
    unexpected API calls;
  * results written to harness/runs/ (git-ignored), never to GP-EVAL/medical_cases.db
    (whose SHA-256 is checked before and after every run);
  * a --mock mode that blocks all network access and replays scripted replies,
    to test the plumbing at zero cost.

Run with the GP-EVAL virtualenv (set GPEVAL_REPO if the clone is elsewhere):
    GP-EVAL/venv/bin/python harness/smoke_test.py --mock ...
API keys are read from GP-EVAL/.env by the repo's own llm_config.py and are
never printed or written to disk by this script.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Path to a local clone of github.com/jamesl385/GP-EVAL (not included in this repo).
REPO = Path(os.environ.get("GPEVAL_REPO", HERE.parent / "GP-EVAL")).resolve()
ORIGINAL_DB = REPO / "medical_cases.db"
RUNS = HERE / "runs"
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402
import openai  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

from multi_med import get_model_config, process_single_case  # noqa: E402  (loads GP-EVAL/.env)
from multi_med.llm_config import DEEPSEEK_API_KEY  # noqa: E402

# Models in the repo's defaults that providers have retired (sources in ENVIRONMENT.md).
RETIRED = {
    "gemini-2.0-flash": "Google shutdown 1 Jun 2026",
    "gemini-2.5-pro-preview-05-06": "Google shutdown 2 Dec 2025",
    "deepseek-V3": "alias deepseek-chat scheduled retirement 24 Jul 2026",
    "deepseek-R1": "alias deepseek-reasoner scheduled retirement 24 Jul 2026",
    "deepseek-R1-0528": "alias deepseek-reasoner scheduled retirement 24 Jul 2026",
}

# DeepSeek list prices, USD per 1M tokens, from api-docs.deepseek.com/quick_start/pricing
# (accessed 18 Oct 2026). (off-peak, peak). Estimates only; the invoice is authoritative.
PRICES = {
    "deepseek-flash": {"hit": (0.003, 0.006), "miss": (0.15, 0.30), "out": (0.60, 1.20)},
    "deepseek-v4-pro": {"hit": (0.022, 0.044), "miss": (0.66, 1.32), "out": (1.98, 3.96)},
}

ROLE_MARKERS = {
    "Doctor": "You are a doctor named Dr Agent",
    "Patient": "Act as a patient",
    "MeasurementAssistant": "You are a clinical assistant",
    "Grader": "You are evaluating a doctor's diagnosis",
}

CALLS: list[dict] = []
HTTP_LOG: list[dict] = []


# ---------------------------------------------------------------- configs
def deepseek_config(model: str, thinking: str) -> dict:
    entry = {"model": model, "base_url": "https://api.deepseek.com", "api_key": DEEPSEEK_API_KEY}
    if thinking in ("enabled", "disabled"):
        entry["extra_body"] = {"thinking": {"type": thinking}}
    cfg = {"config_list": [entry]}
    if thinking != "enabled":
        cfg["temperature"] = 0.1  # same as the repo's configs; ignored by DeepSeek in thinking mode
    return cfg


def resolve(spec: str, mock: bool) -> dict:
    """spec is either a repo MODEL_CONFIGS name or 'deepseek:<model>:<thinking>'."""
    if spec.startswith("deepseek:"):
        _, model, thinking = (spec.split(":") + ["default"])[:3]
        cfg = deepseek_config(model, thinking)
    else:
        if spec in RETIRED and not mock:
            raise SystemExit(f"Refusing to run: '{spec}' is retired ({RETIRED[spec]}). Choose a replacement explicitly.")
        cfg = get_model_config(spec)
    if mock:  # never let a real key or endpoint near a mock run
        cfg = json.loads(json.dumps(cfg, default=str))
        for e in cfg["config_list"]:
            e.update({"api_key": "mock-key", "base_url": "https://mock.invalid/v1"})
            for k in ("api_type", "project_id", "location", "google_application_credentials"):
                e.pop(k, None)
    return cfg


def redact(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg, default=str))
    for e in out.get("config_list", []):
        if "api_key" in e:
            e["api_key"] = "<set>" if e["api_key"] not in (None, "", "None") else "<MISSING>"
        if "google_application_credentials" in e:
            e["google_application_credentials"] = "<path redacted>"
    return out


# ---------------------------------------------------------------- instrumentation
def role_of(messages) -> str:
    first = messages[0].get("content", "") if messages else ""
    first = first if isinstance(first, str) else json.dumps(first)
    for role, marker in ROLE_MARKERS.items():
        if marker in first:
            return role
    return "UNKNOWN"


MOCK_SCRIPT = {
    "Doctor": [
        "How long has he had the fever?",
        "REQUEST EXAMINATION FINDING: otoscopy of both ears",
        "DIAGNOSIS READY: Acute otitis media (left). Fever, ear pulling and a bulging left tympanic membrane.",
    ],
    "Patient": ["About two days, and he keeps pulling at his left ear."],
    "MeasurementAssistant": ["Left tympanic membrane bulging and erythematous; right normal."],
    "Grader": ["YES\nThe primary concept matches.\nTERMINATE"],
}
_mock_counters: dict[str, int] = {}


def mock_completion(kwargs) -> ChatCompletion:
    role = role_of(kwargs["messages"])
    i = _mock_counters.get(role, 0)
    _mock_counters[role] = i + 1
    replies = MOCK_SCRIPT.get(role, ["(mock: unexpected role)"])
    text = replies[min(i, len(replies) - 1)]
    return ChatCompletion.model_validate({
        "id": f"mock-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()),
        "model": f"mock::{kwargs['model']}",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    })


def install_hooks(mock: bool, max_calls: int) -> None:
    original_create = openai.resources.chat.completions.Completions.create

    def create(self, *args, **kwargs):
        if len(CALLS) >= max_calls:
            raise RuntimeError(f"Call budget exceeded ({max_calls}); aborting run.")
        rec = {
            "role": role_of(kwargs.get("messages", [])),
            "requested_model": kwargs.get("model"),
            "host": getattr(self._client.base_url, "host", None),
            "temperature": kwargs.get("temperature"),
            "extra_body": kwargs.get("extra_body"),
            "n_messages": len(kwargs.get("messages", [])),
        }
        t0 = time.perf_counter()
        try:
            resp = mock_completion(kwargs) if mock else original_create(self, *args, **kwargs)
        except Exception as e:  # recorded, then re-raised unchanged
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", latency_s=round(time.perf_counter() - t0, 3))
            CALLS.append(rec)
            raise
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        rec["returned_model"] = resp.model
        rec["usage"] = resp.usage.model_dump() if resp.usage else None
        rec["finish_reason"] = resp.choices[0].finish_reason if resp.choices else None
        CALLS.append(rec)
        return resp

    openai.resources.chat.completions.Completions.create = create

    original_send = httpx.Client.send

    def send(self, request, *args, **kwargs):
        HTTP_LOG.append({"method": request.method, "host": request.url.host, "path": request.url.path})
        if mock:
            raise RuntimeError(f"Network blocked in mock mode: {request.method} {request.url.host}")
        return original_send(self, request, *args, **kwargs)

    httpx.Client.send = send


# ---------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_peak(t: dt.datetime) -> bool:
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def estimate_cost(calls, started_utc) -> dict:
    total, unpriced = 0.0, set()
    tier = 1 if is_peak(started_utc) else 0
    for c in calls:
        u, model = c.get("usage") or {}, c.get("requested_model")
        if model not in PRICES or not u:
            unpriced.add(model)
            continue
        p = PRICES[model]
        hit = u.get("prompt_cache_hit_tokens", 0) or 0
        miss = u.get("prompt_cache_miss_tokens", (u.get("prompt_tokens", 0) or 0) - hit)
        total += (hit * p["hit"][tier] + miss * p["miss"][tier] + (u.get("completion_tokens", 0) or 0) * p["out"][tier]) / 1e6
    return {"usd_estimate": round(total, 6), "tier": "peak" if tier else "off-peak",
            "unpriced_models": sorted(m for m in unpriced if m)}


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True, help="file name inside GP-EVAL/cases/")
    ap.add_argument("--case-index", type=int, required=True, help="0-based line index")
    ap.add_argument("--doctor", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--measurement", required=True)
    ap.add_argument("--grader", required=True)
    ap.add_argument("--mock", action="store_true", help="no network; scripted replies")
    ap.add_argument("--max-calls", type=int, default=120, help="abort if exceeded (cost guard)")
    a = ap.parse_args()

    os.chdir(HERE)  # keep any stray artefacts (e.g. .cache) out of the repo
    case_path = REPO / "cases" / a.case_file
    lines = [l for l in case_path.read_text().splitlines() if l.strip()]
    test_case = json.loads(lines[a.case_index])

    roles = {r: resolve(s, a.mock) for r, s in
             [("doctor", a.doctor), ("patient", a.patient), ("measurement", a.measurement), ("grader", a.grader)]}
    if not a.mock and any("<MISSING>" in json.dumps(redact(c)) for c in roles.values()):
        raise SystemExit("An API key is missing for at least one role. Add it to GP-EVAL/.env (never paste it here).")

    install_hooks(a.mock, a.max_calls)

    db_hash_before = sha256(ORIGINAL_DB)
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()
    error, result = None, None
    try:
        result = process_single_case(test_case, roles["doctor"], roles["patient"], roles["measurement"], roles["grader"])
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    runtime = round(time.perf_counter() - t0, 2)
    db_hash_after = sha256(ORIGINAL_DB)

    log = (result or {}).get("conversation_log", "")
    grader_text = log.split("\n\nGrader: ")[-1] if "\n\nGrader: " in log else None
    run = {
        "run_id": f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        "mode": "mock" if a.mock else "live",
        "started_utc": started.isoformat(),
        "runtime_s": runtime,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "packages": {m: __import__(m).__version__ for m in ("autogen", "openai", "httpx")},
        "case": {"file": a.case_file, "index": a.case_index,
                 "correct_diagnosis": test_case["OSCE_Examination"]["Correct_Diagnosis"]},
        "role_specs": {"doctor": a.doctor, "patient": a.patient, "measurement": a.measurement, "grader": a.grader},
        "role_configs_redacted": {r: redact(c) for r, c in roles.items()},
        "error": error,
        "result": None if result is None else {
            "doctor_diagnosis": result["doctor_diagnosis"],
            "is_correct_repo_logic": result["is_correct"],
            "grader_first_line": grader_text.strip().splitlines()[0] if grader_text else None,
        },
        "calls": CALLS,
        "calls_by_role": {r: sum(1 for c in CALLS if c["role"] == r) for r in {c["role"] for c in CALLS}},
        "unexpected_roles": sorted({c["role"] for c in CALLS} - set(ROLE_MARKERS)),
        "http_requests": HTTP_LOG,
        "http_hosts": sorted({h["host"] for h in HTTP_LOG}),
        "usage_total": {k: sum((c.get("usage") or {}).get(k, 0) or 0 for c in CALLS)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")},
        "cost": estimate_cost(CALLS, started),
        "original_db_sha256": {"before": db_hash_before, "after": db_hash_after,
                               "unchanged": db_hash_before == db_hash_after},
        "conversation_log": log,
    }

    out_dir = RUNS / run["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (out_dir / "conversation.txt").write_text(log)

    if result is not None:
        db = sqlite3.connect(RUNS / "smoke_results.db")
        db.execute("""CREATE TABLE IF NOT EXISTS case_results (
            run_id TEXT, mode TEXT, llm TEXT, presentation TEXT, case_index INTEGER, correct_diagnosis TEXT,
            doctor_diagnosis TEXT, is_correct INTEGER, tag TEXT, conversation_log TEXT)""")
        db.execute("INSERT INTO case_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (run["run_id"], run["mode"], a.doctor, a.case_file, a.case_index, result["correct_diagnosis"],
                    result["doctor_diagnosis"], 1 if result["is_correct"] else 0, test_case.get("tag", ""), log))
        db.commit()
        db.close()

    summary = {k: run[k] for k in ("run_id", "mode", "runtime_s", "error", "result", "calls_by_role",
                                     "unexpected_roles", "http_hosts", "usage_total", "cost")}
    summary["original_db_unchanged"] = run["original_db_sha256"]["unchanged"]
    summary["returned_models"] = sorted({str(c.get("returned_model")) for c in CALLS})
    print("\n=== RUN SUMMARY ===\n" + json.dumps(summary, indent=2, default=str))
    print(f"Full record: {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
