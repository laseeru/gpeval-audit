# GP-EVAL audit: grading, leakage and reproducibility checks on a multi-agent diagnostic LLM benchmark

An independent, exploratory audit of **GP-EVAL / Multi-Med** ([github.com/jamesl385/GP-EVAL](https://github.com/jamesl385/GP-EVAL), commit `b67c37e`). This is a public benchmark in which an LLM "doctor" interviews an LLM "patient", requests examination findings and tests from a "measurement assistant", and is scored by an LLM "grader". Its published results database holds **4,056 runs** (13 models × 312 cases).

I am a medical doctor (MD, Havana 2026). I did this because I did not want to trust a leaderboard number until I had checked how it was produced. The question behind it: **when a benchmark says a model "diagnosed" a case, what exactly was measured?**

> **What this repository is not.** It contains none of GP-EVAL's code or data; that repository has no licence. To reproduce anything, clone GP-EVAL yourself (see [Reproducing](#reproducing)). Findings below are aggregate counts plus run identifiers.

## Findings (re-verified 8 Nov 2026 against the published DB, SHA-256 `254d7b74…3f47902`)

### 1. A grader-parsing defect marks some "NO" verdicts as correct
`is_correct` is set when the string `"YES"` appears **anywhere** in the grader's reply (`"YES" in content.upper()`), not when the verdict is YES. Comparing the grader's *first word* with the stored flag:

| Grader's first word | Stored `is_correct` | Runs |
|---|---|---|
| YES | 1 | 2,707 |
| NO | 0 | 1,300 |
| **NO** | **1** | **3** |
| (no grader message) | 0 | 46 |

- **3 runs** (DB ids 902, 2570, 2800; models `deepseek-V3`, `gpt-4.1-nano`, `gemini-2.5-flash-preview-05-20`) are stored as correct although the grader's verdict was NO.
- **46 runs** (1.1%) ended without any grader message and were scored as incorrect, so a pipeline failure is indistinguishable from a wrong diagnosis.
- **Magnitude:** small. At most 1 run per model, so it does **not** change the published model ranking. The point is that the scoring rule is fragile; it would matter more with a different grader or prompt.

### 2. Test results can give the diagnosis away
In this benchmark the "doctor" can request any test, and the measurement assistant returns the case's stored result. Some stored results (imaging reports, biopsy findings) are effectively diagnostic, so a model can reach the right label by ordering the right test rather than by clinical reasoning.

A crude keyword screen ([`audit/audit_gpeval.py`](audit/audit_gpeval.py)) flags **100 of 312 cases** whose test results contain the key terms of the expected diagnosis. **This is not a leakage rate.** The screen both over-flags (keyword overlap without real leakage) and under-flags. For example, a CT report describing a 7 mm thickened appendix with periappendiceal fat stranding never names "appendicitis" and was missed. Every flagged case needs clinical review before any claim is made.

### 3. Outcome-only grading cannot see reasoning errors
The grader compares the final label with the expected label. A correct label reached with flawed reasoning, or citing evidence the doctor never obtained, still scores as correct. All three smoke-test cases below were graded correct, yet my review found problems the grade cannot see. The clearest is the IgA nephropathy case, where the "doctor" used test results it had never requested to rule out the main alternative.

### 4. The published results cannot be reproduced exactly
- `requirements.txt` lists `ag2[openai]`, `ag2[gemini]` and `python-dotenv` **without versions**, so the library versions behind the published numbers are unknown.
- **Every default model is now retired** (`gemini-2.0-flash` patient and measurement assistant; `gemini-2.5-pro-preview-05-06` grader). The repo's DeepSeek model names have also been retired and remapped. Any rerun today is a documented deviation, not a reproduction.

Details: [METHODS.md](METHODS.md).

## Smoke test (18 Oct 2026, n = 3)
To check that the pipeline still runs, I wrote an instrumented single-case runner ([`harness/smoke_test.py`](harness/smoke_test.py)). It calls the benchmark's own `process_single_case` **unmodified** and adds:
- explicit per-role model selection;
- a log of every model call and outgoing HTTP request (no keys or message content);
- a call cap;
- a SHA-256 check that the original results DB is unchanged;
- a zero-cost `--mock` mode.

| Case | Expected diagnosis | Result |
|---|---|---|
| The febrile child #24 | Acute otitis media (left) | correct, 13 calls |
| Acute abdominal pain #1 | Acute appendicitis | correct, 11 calls |
| Haematuria #23 | IgA nephropathy | correct, 21 calls |

**Configuration (a deviation from the paper's setup, because the originals are retired):** all roles used DeepSeek (`deepseek-flash` for doctor, patient and measurement; `deepseek-v4-pro` for grader), with thinking off and temperature 0.1. Totals: 45 calls, about US$0.006. Sanitised run records (no conversations) are in [`runs/`](runs/).

**Limitations:** three hand-picked cases, one vendor for every role (self-preference bias is possible), and model aliases that change over time. This shows the pipeline runs; it is not an evaluation of any model.

### Clinical review of the three conversations
I reviewed each transcript against the case data as a physician (8 Nov 2026). Initial notes were drafted with AI assistance; I confirmed, corrected or dropped each point, and one drafted finding turned out to be wrong and was removed.

**Case 1: febrile 2-year-old, acute otitis media (correct)**
- *Doctor:* reached the diagnosis efficiently (otoscopy with pneumatic insufflation, meningism check), but never asked about the two loose stools in the history. That gap could miss other pathology. It also did not examine the other ear for comparison, and did not consider a urinary tract infection in a febrile toddler.
- *Patient agent:* volunteered most key features in its first reply, making the history easier than a real consultation, and added a mild inference ("no ear infections before") the case does not state.
- *Measurement assistant:* when asked for findings the case does not contain (tragal and mastoid tenderness), it said nothing instead of stating that the information was unavailable.

**Case 2: 28-year-old man, acute appendicitis (correct)**
- *Doctor:* diagnosed from history, abdominal exam and bloods, then recommended imaging and surgical review. That is reasonable for a classic presentation. But it never asked about dysuria, bowel habit or a food exposure the case contains, never requested vital signs, and dismissed a urinary cause on the urinalysis alone.
- *Leakage:* the stored CT report describes a 7 mm thickened appendix with periappendiceal fat stranding and no perforation. It never names appendicitis, yet it is diagnostic and even gives away the "non-perforated" qualifier. The doctor did not order it here, but a model that orders a CT gets the answer. The keyword screen missed this case.

**Case 3: 34-year-old man, IgA nephropathy (correct label, flawed reasoning)**
- *Doctor:* argued against post-infectious glomerulonephritis using a "normal complement-related picture". **Complement was never requested or shown.** It also called haematuria after a sore throat "a few months" earlier the classic IgA picture, although classic IgA haematuria follows infection within days. It also overstated that a negative drug history "rules out" NSAID-related causes.
- *Measurement assistant:* returned an albumin-to-creatinine ratio when a protein-to-creatinine ratio was requested, without flagging the substitution. In real practice these are different tests.
- *Grader:* correct, but outcome-only grading cannot detect any of the reasoning errors above.

**Takeaway:** in 3 of 3 correctly graded runs there were history gaps, simulator inconsistencies, or reasoning a clinician would not accept. Accuracy on this kind of benchmark measures whether the final label matches, not whether the consultation was safe or sound.

## Reproducing
```bash
git clone https://github.com/jamesl385/GP-EVAL.git      # inside this repo; git-ignored
python3 audit/audit_gpeval.py GP-EVAL/medical_cases.db GP-EVAL/cases

# Smoke test: create a venv inside GP-EVAL, install harness/requirements-lock.txt,
# put DEEPSEEK_API_KEY in GP-EVAL/.env, then:
GP-EVAL/venv/bin/python harness/smoke_test.py --mock \
  --case-file "The febrile child_all_cases.jsonl" --case-index 24 \
  --doctor deepseek:deepseek-flash:disabled --patient gemini-2.0-flash \
  --measurement gemini-2.0-flash --grader gemini-2.5-pro-preview-05-06
```
Use `--mock` first (no network, no cost). Live mode refuses retired model names.

## Status
- **Clinical review of the three smoke-test conversations:** completed 8 Nov 2026 (above).
- **GP-EVAL authors:** I discussed these findings with a member of the Cao Lab before publishing. The project is archived, and they were happy for the results to be shared.
- This is unpublished, exploratory work. Corrections are welcome; please open an issue.

## How this was built
I wrote this with substantial help from an AI assistant (Claude) for code, documentation and first-draft case notes. I directed the work and ran it. I reviewed every clinical observation against the case data myself, and I am responsible for every claim here.

## Credit
GP-EVAL / Multi-Med was built by its authors (repository by `jamesl385`, associated with the Cao Lab, Monash University). This audit is meant to help make benchmarks like it more trustworthy, not to diminish the work.

Abdul-Rashid Dasana Abdulai, MD · [github.com/laseeru](https://github.com/laseeru) · ORCID [0009-0003-9109-7388](https://orcid.org/0009-0003-9109-7388)
