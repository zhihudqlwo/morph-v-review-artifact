# Review guide

## Scope

This is a minimal implementation and a frozen-record reproduction package, not a claim of end-to-end reproduction of every experiment. MORPH-V uses frozen models; there is no model training or parameter-update stage.

Included:

- The actual expanded typed DSL implementation: 81 compatible signatures, generic compilation/execution, invariant checks, structural ranking, and persistent program-library updates.
- The 107 outcome-hidden structural contexts and 18 recorded Qwen3.6 proposal responses from the reported **open-composition** run.
- The original saved `L0`–`L3` program libraries, which can be reloaded and executed.
- All 12 frozen checkpoint prediction banks and separately stored labels for the fixed interval (335 questions), slot-set (60), and event-time (240) populations.
- The original checkpoint evaluator, including video-cluster bootstrap intervals and Holm-adjusted adjacent tests, and focused unit tests.

Not included:

- Raw videos, subtitles, benchmark question/option text, pretrained weights, proprietary endpoints, credentials, or model-response transport receipts.
- A fresh execution of the proposer, SigLIP/ASR feature extraction, or answer-model inference. The model replies and derived contexts are frozen inputs to the offline path.
- The main deployment experiments in Tables 1–2 and 4, the distinct three-proposal held-out recovery run, or the post-hoc event-safety variant. The included open-composition run has **zero** exact reference-signature matches, not three.

## Environment and quick start

Python 3.9 or newer. A CPU is sufficient for the supplied offline path; no API key or GPU is required. The core modules and the offline reproduction use the Python standard library. PyYAML is needed for the optional original acquisition runner; pytest is used for tests.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python scripts/reproduce.py
python -m pytest -q
```

Alternatively, the reproduction itself needs no installation or network:

```bash
PYTHONPATH=src python3 scripts/reproduce.py
```

Generated files are written only under `outputs/reproduction/`, which is excluded from the repository. Original released records under `data/` are never changed.

Release validation used a clean Python 3.9.6 environment with the pinned dependencies: **27 tests passed**, the acquisition replay selected the original three graph IDs, and every checkpoint metric (including confidence intervals) matched the frozen report. File integrity can be checked with `shasum -a 256 -c MANIFEST.sha256`.

## Commands and correspondence to the paper

| Command | What it reproduces | What it does not establish |
| --- | --- | --- |
| `python scripts/reproduce.py --stage acquisition` | Recompiles and structurally evaluates all 18 frozen proposal responses: 10 valid candidates, 3 stored graphs, 0 exact reference matches; verifies selected graph IDs. | It does not obtain fresh model responses or remeasure the original model latency. |
| `python scripts/reproduce.py --stage checkpoints` | Loads the original `L0`–`L3` libraries and executes their stored graphs through the generic registry on the released structural contexts. | It does not compute new answer predictions. |
| `python scripts/reproduce.py --stage growth` | Recomputes Table 3 / Appendix Table 7 from all recorded predictions, with the original sample denominators and 10,000 bootstrap draws. | It is a metric reproduction, not an independent rerun of video QA. |
| `python -m pytest -q` | Exercises all 81 signatures, rejection of incompatible graphs/outcome fields, persistence and reload, acquisition controls, and label-late evaluation. | Small synthetic unit-test inputs are not empirical paper results. |

Expected growth point estimates:

| State | Interval | Slot-set | Event-time | Macro | Gain vs L0 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 63.88 | 55.00 | 37.08 | 51.99 | 0.00 |
| L1 | 70.15 | 55.00 | 37.08 | 54.08 | 2.09 |
| L2 | 70.15 | 70.00 | 37.08 | 59.08 | 7.09 |
| L3 | 70.15 | 70.00 | 39.17 | 59.77 | 7.78 |

The original population is reused across checkpoints. The event-time cohort here differs from the fresh ASII four-arm cohort used for the deployment comparison. Only the first adjacent checkpoint gain passes the Holm-adjusted 0.05 threshold; monotonic point estimates do not imply that every increment is significant.

## Implementation and proposal interface

- `src/evogroundtest/morph_v/expanded_dsl.py`: registry, 81 signatures, typed graphs, compiler, generic executor, binders, transformations, objectives, invariants, and persisted libraries.
- `src/evogroundtest/morph_v/expanded_acquisition.py`: outcome-hidden proposal packets, bounded attempts, structural ranking, coverage feedback, and acquisition controls.
- `src/evogroundtest/morph_v/autonomous_discovery.py`: original boundary checks and legacy helper types used by the expanded implementation; its older candidate inventory is not the expanded search space.
- `src/evogroundtest/morph_v/checkpoint_evaluation.py`: separate authentication of prediction banks followed by label joining and statistics.
- `scripts/morph_v/run_expanded_acquisition_v1.py`: original model/reply-ledger adapter, including the system prompt and structured JSON schema.
- `configs/morph_v/expanded_acquisition_v1.yaml`: paper-run limits, seed, proposer settings, and frozen carrier order. The private endpoint has been removed; credentials are read only from environment variables.

To use the original acquisition adapter for the same offline replay:

```bash
PYTHONPATH=src python scripts/morph_v/run_expanded_acquisition_v1.py \
  --config configs/morph_v/expanded_acquisition_v1.yaml \
  --procedure morph_v \
  --response-ledger data/acquisition/responses.jsonl \
  --ledger-model Qwen3.6-27B \
  --execute
```

Without `--execute`, this adapter performs a no-network preflight. Without a response ledger, `--execute` requires a compatible endpoint configured by `MORPH_QWEN_BASE_URL` and `MORPH_QWEN_API_KEY` and may incur model-serving costs. No endpoint is bundled. New model responses need not reproduce the frozen graph choices exactly.

## Frozen inputs and anonymization

`data/acquisition/contexts.jsonl` contains derived numeric structural evidence, not benchmark media or question text. The slot-set contexts use ordinal scores reconstructed from frozen SigLIP retrieval traces, as indicated by their `context_provenance`; they are not regenerated raw similarity matrices. Model replies are replayed verbatim. Local media paths were removed; question/video identifiers in the evaluation records were replaced with order-preserving anonymous IDs so bootstrap sampling retains the original grouping and order. No predictions, labels, scores, graph definitions, or sample memberships were changed. Prediction-file hashes were recalculated for this anonymous serialization.

Labels under `data/growth/` are used only by the evaluation stage; acquisition loads only `data/acquisition/`. The released original program libraries retain their registry and admission hashes. The reproduction checks its numerical outputs against `data/growth/expected_metrics.json`; this check is not a new scientific measurement.

## Data and third-party resources

Raw datasets, model weights, and third-party code are not redistributed. End-to-end video experiments require the authors' original preprocessing/model-serving pipeline and separately obtained Video-MME-v2, LongVideoBench, and HERBench resources under their respective terms. This minimal release does not claim to package that full pipeline. Benchmark-derived labels and model predictions are supplied only as anonymized numerical evaluation records. Third-party resource licenses are not changed by this artifact.
