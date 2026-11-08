# Methods and environment

## Provenance
| Item | Value |
|---|---|
| Upstream | https://github.com/jamesl385/GP-EVAL, commit `b67c37e` ("Initial commit: Medical AI diagnosis system (clean)", 29 Jun 2025, single commit). No licence file. |
| Working tree | Unmodified. |
| Results DB | `medical_cases.db`, SHA-256 `254d7b74c03219c606db7c6933d06d368b6227ae687ebcd99c745223b3f47902`. Opened read-only; the harness checks the hash before and after every run. |
| Python | 3.12.11 |
| Pins used | `ag2==0.9.4` (last release before the upstream commit), `openai==1.93.0`, `httpx==0.28.1`, `python-dotenv==1.2.4`. Full lock: [`harness/requirements-lock.txt`](harness/requirements-lock.txt). |
| Unpinned upstream | `requirements.txt` lists `ag2[openai]`, `ag2[gemini]`, `python-dotenv` with no versions, so the versions behind the published results are unknown. |

## How a case runs (from reading `multi_med/agents.py`)
`process_single_case()` builds a four-agent AutoGen group chat:

| Role | Upstream default model | Sees | Speaks |
|---|---|---|---|
| Doctor | model under test | its own instructions | after Patient or Measurement |
| Patient | `gemini-2.0-flash` | the `Patient_Actor` block | after Doctor, unless the Doctor requests an exam or tests |
| MeasurementAssistant | `gemini-2.0-flash` | `Physical_Examination_Findings` and `Test_Results` | when the Doctor writes `REQUEST EXAMINATION FINDING:` or `REQUEST TESTS:` |
| Grader | `gemini-2.5-pro-preview-05-06` | `Correct_Diagnosis` | when the Doctor writes `DIAGNOSIS READY:`; the chat then ends |

- **Speaker selection** is a Python function, so the GroupChatManager makes no LLM calls (confirmed in mock mode).
- **Scoring:** `is_correct = "YES" in grader_text.upper() and doctor_diagnosis is not None`. The diagnosis is extracted up to the first full stop or newline.
- **Limits:** `max_round=100`. Upstream has no turn or cost cap, so the harness adds a 120-call guard. Temperature is 0.1.
- **Expected cost:** about 2 calls per doctor turn plus 1 grader call. Stored logs have a median of 11 doctor turns (max 51), so roughly 20–30 calls per case, with about 100 in the worst case. Each call resends the whole conversation.

## Audit checks ([`audit/audit_gpeval.py`](audit/audit_gpeval.py))
1. **Verdict parsing:** extract the grader's first word (`YES`/`NO`) from the last grader message in each stored conversation and compare it with the stored `is_correct`.
2. **Missing verdicts:** count runs whose conversation contains no grader message.
3. **Leakage screen:** for each case, take the expected diagnosis, drop parenthetical text and a stop list, keep words of 5 or more letters, and flag the case if its first two key terms both appear in the serialised `Test_Results`. This is deliberately crude: it over-flags, and it misses diagnostic findings that are described without naming the disease. It is a triage list for clinical review, not a measurement.

## Configuration issues found before any live call
1. **All default models are retired.** `gemini-2.0-flash` (shut down 1 Jun 2026) and `gemini-2.5-pro-preview-05-06` (shut down 2 Dec 2025). The grader's prompt and scoring code can be kept; its model cannot.
2. **The repo's DeepSeek names are retired.** `deepseek-chat` and `deepseek-reasoner` were retired in Jul 2026; current aliases point to newer models. A DeepSeek rerun therefore tests different models from the stored `deepseek-V3` and `deepseek-R1-0528` rows.
3. **DeepSeek thinking mode is on by default** and ignores temperature. The harness sets it explicitly (`extra_body.thinking.type`).
4. **Vertex-based roles need Google Cloud credentials.**
5. **The upstream `main.py` and `main_parallel.py` write into `medical_cases.db`.** The harness never uses them; it calls `process_single_case` directly and writes to `harness/runs/`.

## Smoke-test configuration (18 Oct 2026)
A documented deviation from the upstream defaults:
- Doctor, Patient, Measurement: `deepseek:deepseek-flash:disabled`
- Grader: `deepseek:deepseek-v4-pro:disabled`
- Prompts, speaker logic, diagnosis extraction and `is_correct` are unchanged.
- Model aliases are unversioned, so record the date with every run.

Per-run records (calls by role, tokens, cost estimate, HTTP hosts, DB hash; no message content) are in [`runs/`](runs/).
