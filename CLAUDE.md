# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                                    # Run ranker tests
uv run pytest tests/test_core_pairwise.py -k "test_name"  # Run a single test
uv run learnedcache --help                       # Ranker CLI overview
uv run learnedcache train-ranker --help          # Help for a specific subcommand
uv run learnedcache train-from-binary --help     # Help for the binary-log training command

# Eviction-time binary classifier (standalone module — NOT part of the learnedcache CLI)
.venv/bin/python -m evict_classifier train --help
.venv/bin/python -m evict_classifier train \
  --data-dir data/tracer-bundle-may-28/cache_ext_logs \
  --output-dir <out> --horizon 10 --residency-cap 30 --target-rows 5000000
.venv/bin/python -m pytest evict_classifier/tests/      # Classifier tests

# Trace-driven hit-rate simulation (FIFO / LRU / Belady MIN / protect policy)
.venv/bin/python -m evict_classifier simulate \
  --data-dir data/tracer-bundle-may-28/cache_ext_logs \
  --model-root notebooks-v2/protect-test --output-dir <out> \
  --capacities 262144,524288,1048576
```

## Architecture

This repo holds **two model pipelines**:

1. The **`learnedcache` package** (Gen 5): a linear pairwise-diff ranker. The model compares two cached items at eviction time and predicts which should be evicted first (lower reuse time = evict sooner). Deployed via the `cache_ext_fifo_ml` policy.
2. The **`evict_classifier` module** (Gen 6, current focus): a pointwise binary classifier predicting `P(page reused within horizon H)` at eviction time. Deployed via the `cache_ext_fifo_ml_protect` skip-in-place policy. Standalone by design — it *copies* the loading/preprocess code it needs from `learnedcache` so the ranker stays untouched; run via `python -m evict_classifier`, never wired into the `learnedcache` typer app.

### Model evolution

- **Gen 1-3** (pointwise classifiers): Predict binned time-until-next-access directly. Used custom Keras activations (`Squaremax`, `TaylorSoftmax`) for multi-class classification. Visualized in `visualizations/` as `gen1_output.png` through `gen3_output.png` and classifier architecture diagrams (`pointwisebinaryclassifier.png`, `squaremax5classifier.png`, `taylormax5classifier.png`).
- **Gen 5** (pairwise Bradley-Terry ranker): Instead of predicting absolute reuse time, compares pairs of items and predicts which is reused sooner. Single `Dense(1, sigmoid, use_bias=False)` layer on the feature difference vector. Produces interpretable per-bin weights that map directly to BPF maps.
- **Gen 6** (eviction-time binary reuse classifier, `evict_classifier/`): back to pointwise, but with a binary "reused within H" label, the same 9 eviction-time features, and `Dense(1, sigmoid, use_bias=True)` — the bias becomes the in-kernel decision offset. An insertion-time admission model was explored first and abandoned: at `folio_added` a page has no history (page-level features are all sentinels), and analysis showed reuse is inode-homogeneous with only coarse inode signal available at insertion (oracle AUC 0.92 / available-features 0.845). Eviction time has the rich page-specific signal.

### Pipeline stages

There are two input paths into the system, both using **streaming training**:

**CSV path (text logs):**
1. **Log transform** (`transform-logs`): Raw key=value log files → CSV. Each line is whitespace-separated `key=value` pairs.
2. **Train ranker** (`train-ranker`): Access+eviction CSV pairs → streaming supervised generator → feature discretization (KBinsDiscretizer, quantile) → one-hot encoding → pairwise-diff sampling within eviction events → linear model training via Python generators. The full dataset is **never simultaneously in RAM**.
3. **Export model** (`export-model`): Trained Keras model + discretizer → BPF-compatible JSON with quantized integer weights.
4. **Full pipeline** (`full-pipeline`): All three stages in one command.
5. **Train and export** (`train-and-export`): `train-ranker` + `export-model` in one command (skips log transform).

**Binary path (kernel trace logs):**
1. **Train from binary** (`train-from-binary`): Scans a data directory for workload subdirectories containing `iter_*/` with binary access (`mglru_lc_access_*.bin`) and eviction (`mglru_lc_eviction_*.bin`) files → trains one pairwise ranker per workload via the streaming pipeline → exports BPF weights per workload. Binary records are memory-mapped via numpy structured dtypes (88-byte access records, 8-byte eviction records) — no CSV conversion needed.

### Key architectural decisions

- **Pairwise Bradley-Terry formulation**: The model takes the *difference* of two item feature vectors and predicts P(A reused sooner than B). Labels are derived by comparing `time_until_next_reuse_from_eviction` values. More robust than direct regression because it's invariant to absolute time scales.
- **Streaming training pipeline**: `run_train_ranker` uses a three-phase streaming pipeline:
  1. **Stats pass**: One pass through all trials collects a discretizer subsample (up to 200K rows from raw access arrays), tracks the global target maximum, and counts eviction events for `steps_per_epoch` estimation. Approximates the derived-feature distribution from inter-access gaps — never builds the full supervised DataFrame.
  2. **Fit discretizer**: `fit_discretizer_from_sample` on the subsampled feature matrix.
  3. **Train via generators**: `StreamingSupervisedGenerator` produces per-chunk eviction-event batches with bounded forward lookahead. `_make_pair_generator` wraps this infinitely for `model.fit()` with `steps_per_epoch`. Only one trial's data is in memory at a time.
- **Lazy loading everywhere**: `read_access_eviction_trial_pairs` is a generator (yields one trial at a time). `build_pairs_from_binary` memory-maps binary files and yields trials one at a time. `StreamingSupervisedGenerator` processes eviction events in configurable chunks with bounded page-state. End-to-end: only one trial's data is in memory at any point.
- **Access data never copied**: The streaming generator uses a ts-sort index into the original (often memmap) access array. Access records are loaded on demand into per-page timestamp/feature lists as the eviction window advances. The sort index is the only per-record allocation (~N × 8 bytes).
- **Lookahead auto-scaling**: The access window lookahead defaults to 30% of the eviction time span (minimum 1,000,000 µs), computed once per trial at negligible cost.
- **Within-event sampling**: Pairwise samples are drawn only within the same eviction event (same eviction chunk index), ensuring both items were candidates at the same eviction moment. Uses vectorized uniform-offset pair generation (one-pass unique pairs, no rejection sampling). Presorted fast-path avoids O(N log N) argsort when event_ids are already sorted.
- **No-reuse handling**: Items never reused get a surrogate label (`max_finite + 1.0`), ensuring they rank worse than any actually-reused item.
- **Trial-based holdout**: The last trial is held out as the test set (not random split), simulating generalization to unseen access patterns. Holdout evaluation also uses the streaming generator — the test trial is never fully materialised.
- **Linear model with no bias**: Produces directly interpretable per-bin weights quantizable to integers for BPF.
- **Memory-optimized preprocess pipeline**: Discretizer returns dense `int8` arrays directly from CSR data (skipping float64 intermediate). One-hot encoding uses a single pre-allocated `int8` buffer with in-place column writes. int8 is safe because one-hot values are {0,1} and pairwise differences stay in {-1,0,1}. For 12M rows × 9 features, total one-hot memory drops from ~864 MB (float64 DataFrame) to ~108 MB (int8).
- **In-memory artifact handoff**: `run_train_ranker` returns `model` and `discretizer` in its result dict. `run_export_model` accepts these as optional arguments, avoiding a redundant disk round-trip (save → load).
- **Eager memory reclamation**: `_make_pair_generator` calls `del stream; gc.collect()` after each trial to reclaim sort-index and page-state memory before the next trial starts. `save_evaluation_outputs` deletes `x_eval_full` immediately after the dot product, then frees intermediates as they're consumed.

### Module layout

| Module | Role |
|---|---|
| `learnedcache/__init__.py` | Re-exports `app` from `cli` |
| `learnedcache/__main__.py` | Entry point for `python -m learnedcache` |
| `learnedcache/cli.py` | Typer CLI with 6 commands (`transform-logs`, `train-ranker`, `export-model`, `train-and-export`, `full-pipeline`, `train-from-binary`) and all option defaults. Module-level constants: `DEFAULT_DISCRETIZE_COLS`, `DERIVED_FEATURE_COL`, `DEFAULT_EXPORT_FEATURE_NAMES` |
| `learnedcache/core.py` | Orchestration hub: streaming generator (`StreamingSupervisedGenerator`), three-phase streaming training pipeline (`_run_train_ranker_streaming`), infinite generator wrapper (`_make_pair_generator`), pairwise-diff sampling (`_sample_pairwise_diffs_by_event`), export (`run_export_model`), batch workload training (`run_train_from_binary`). Module-level constants: `PAGE_KEY_COLS`, `TS_COL`, `DERIVED_FEATURE_COL`, `NO_REUSE_LABEL_OFFSET` |
| `learnedcache/binary_loading.py` | Binary log I/O: memory-mapped structured numpy dtypes (`_ACCESS_DTYPE` 88-byte, `_EVICTION_DTYPE` 8-byte); `discover_workloads_and_iters`, `build_pairs_from_binary` (generator), `read_binary_access_log`, `read_binary_eviction_log`, `_mmap_or_warn` |
| `learnedcache/models.py` | Linear pairwise-diff Keras model builder: `build_pairwise_diff_model` → `Input → Dense(1, sigmoid, use_bias=False)`. `build_model` is a backward-compatible alias |
| `learnedcache/loading.py` | Text log parsing (`parse_log_to_csv`, `transform_logs_to_csvs`), CSV I/O (`read_csvs_to_dataframe`), lazy access/eviction file pairing by token (`read_access_eviction_trial_pairs` — generator), token counting (`count_access_eviction_trial_pairs`) |
| `learnedcache/preprocess.py` | `fit_discretizer_from_sample` (for pre-subsampled data), `transform_discretizer_batch` (apply fitted discretizer, returns int8), `one_hot_encode_features` (pre-allocated int8 buffer, in-place column writes) |
| `learnedcache/helpers.py` | Consolidated `save_evaluation_outputs`: 6 training-curve PNGs + eval report (txt) in one pass with single score computation. Module-level matplotlib rcParams configuration |

**Dependency graph:** `cli.py` → `core.py` → {`binary_loading`, `helpers`, `loading`, `models`, `preprocess`}. The bottom 5 modules are leaves with zero internal dependencies. No circular dependencies.

### Data flow

**Training data flow (both CSV and binary):**

```
binary .bin files ──mmap──→ numpy structured arrays ──┐
                                                       ├──→ StreamingSupervisedGenerator
CSV files ──parse──→ pd.DataFrame ────────────────────┘        │
                                                                │  per (page_group × eviction_chunk):
                                                                │  1. build candidate rows (derived + discretize cols)
                                                                │  2. discretize (int8 via CSR.data)
                                                                │  3. one-hot encode (int8, pre-allocated buffer)
                                                                │  4. sample pairwise diffs (vectorized, within-event)
                                                                │  5. yield mini-batches
                                                                ▼
                                              model.fit(generator, steps_per_epoch=...)
                                                                │
                                                                ▼
                                              export → BPF JSON (quantized int weights)
```

**Text log path (CSV):** Raw logs (`*.log`) contain cache access traces with page keys (`dm`, `dn`, `in`, `of`) and features (`pd`, `sz`, `fq`, `sd`, `p2`, `id`, `i2`, `ie`). After transform, access CSVs have these columns plus timestamps. Eviction CSVs only need timestamps. Access/eviction files are paired by filename token (e.g., `trial1_access.csv` ↔ `trial1_eviction.csv`), and tokens must match exactly across patterns.

**Binary log path:** Binary access records are 88-byte structs with fields matching the CSV columns above (plus `_pad` alignment). Eviction records are 8-byte `ts` values. Files live in `data/<workload>/iter_*/mglru_lc_{access,eviction}_*.bin`. The `train-from-binary` CLI command discovers workloads and iter dirs automatically. Binary fields are already typed as unsigned integers — no `pd.to_numeric` needed.

## evict_classifier module (Gen 6)

Standalone package at `evict_classifier/` — separate CLI (`python -m evict_classifier`), separate tests, zero imports from `learnedcache` (`loading.py`/`preprocess.py` are intentional copies of `binary_loading.py`/`preprocess.py`).

### How it samples (the key design)

Every candidate row is a *(prior access aᵢ of page p, eviction event E)* pair where E falls in the interval between aᵢ and p's next access; `label = 1 iff next_access − E < horizon`. Instead of streaming this join through a Python loop per epoch (the ranker's bottleneck), `sampling.py` computes per-access event-index bands with one lexsort + a few `searchsorted` calls and draws a bounded, class-balanced sample entirely in numpy — the join is paid **once** (~seconds for 12M accesses), then training runs in-memory. Full ycsb_b train ≈ 25 s; all three YCSB workloads ≈ 90 s at 5M rows.

Two corrections applied at sampling time:

- **Right-censoring**: eviction events later than `last_access_ts − horizon` are dropped (their reuse window is unobservable; without this the holdout tail degenerates to all-negative and AUC is nan).
- **Residency cap** (`--residency-cap`, default 30 s): the eviction log has no page identity, so a page would otherwise count as a candidate long after it was realistically evicted. The cap excludes candidates idle beyond the cache-turnover estimate (`cache_pages / insertion_rate`), removing "phantom" easy negatives. Capping made AUC honest on ycsb_b/c (0.89→0.87) and *raised* ycsb_e (0.84→0.89, recall 0.48→0.74).

### Module layout

| Module | Role |
|---|---|
| `sampling.py` | Vectorized one-pass candidate sampling: `_interval_bounds` (event bands per access, with horizon + residency cap), `_draw` (weighted record/event draw), `sample_trial`, `collect_workload_sample`. Verified against an O(n²) brute force in tests. `_InsertionAnchor` consumes the insertion log (auto-discovered) to fix the TSA-anchor skew: candidates whose page was re-inserted get the kernel's fresh-entry view (TSA from insertion, `pd`/`p2` sentinels, `fq`=0) |
| `models.py` | `build_binary_classifier`: `Dense(1, sigmoid, use_bias=True)`, layer named `ranking_weight` (exporter contract) |
| `train.py` | Per-workload orchestration: sample → fit discretizer → one-hot once → in-memory `model.fit` → holdout eval → artifacts (`model.keras`, `discretizer.pkl`, `model_weights.json`, `eval_report.txt`, `metrics.json`, `feature_importance.png`) |
| `export.py` | Ranker-compatible BPF JSON **plus** top-level `bias`/`bias_int`/`threshold`/`threshold_int`; bin edges clamped to u64 (sentinel float-rounding can hit 2⁶⁴) |
| `plots.py` | `feature_importance.png` per-bin weight chart (green=protect, red=evict) |
| `simulate.py` | Trace-driven hit-rate simulator: FIFO, LRU, Belady MIN (bypass-allowed, verified vs exhaustive search), and the protect policy with kernel-faithful min-of-group/rotation semantics. Reports full + tail-20% hit rates and the compulsory-miss ceiling |
| `loading.py`, `preprocess.py` | Verbatim copies from `learnedcache` (keep in sync manually if the source changes) |
| `cli.py` / `__main__.py` | Typer app; `--horizon` and `--residency-cap` are in **seconds**, converted to ns internally |
| `KNOWN_ISSUES.md` | Documented train/serve skews and sampling limitations — **read before debugging policy behavior or trusting eval numbers** |

### Key facts

- Feature order is the BPF contract: `["pd","sz","fq","sd","p2","id","i2","ie", time_since_last_access_at_eviction]` — must match the enum in `cache_ext_fifo_ml_protect.bpf.c`.
- In-kernel decision: `sum(weights_int[bin]) + bias_int > threshold_int` ⇒ protect. `threshold` is in logit units (0 ⇒ P>0.5); balanced training makes P>0.5 protect-heavy under the natural ~20% positive rate — tune per workload.
- Reference results (5M rows, H=10s, cap=30s, may-28 traces): ycsb_b 0.874 / ycsb_c 0.887 / ycsb_e 0.890 holdout AUC. Artifacts in `notebooks-v2/protect-test/`.

## Repository layout

### `data/` — Input traces

Contains the **tracer-bundle-may-28** with YCSB workloads (B, C, E) captured from a kernel tracer run. Each workload has one iter directory with three binary files:

```
data/tracer-bundle-may-28/
  ├── DEVLOG.md
  ├── tracer_run.log
  └── cache_ext_logs/
      ├── ycsb_b/iter_1/
      │   ├── mglru_lc_access_1780028116.bin      # 88-byte structured records
      │   ├── mglru_lc_eviction_1780028116.bin     # 8-byte ts records
      │   └── mglru_lc_insertion_1780028116.bin    # insertion trace
      ├── ycsb_c/iter_1/   (same structure)
      └── ycsb_e/iter_1/   (same structure)
```

Earlier workloads (fileserver, mongo, webproxy, varmail, etc.) were run externally; their trained artifacts remain in `outputs-final/`.

### `output-streaming/` — Streaming-trained model artifacts

Three YCSB workload directories (ycsb_b, ycsb_c, ycsb_e), each containing the artifact set from streaming `train-from-binary`:

- `model.keras`, `discretizer.pkl`, `model_weights.json`, `eval_report.txt`
- `accuracy.png`, `loss.png`, `live_auc.png`, `feature_importance.png`

### `tests/` — Test suite (ranker)

3 test files plus `conftest.py`:

- `test_core_pairwise.py` — Pairwise sampling correctness
- `test_export_contract.py` — BPF export format contract
- `test_loading_pairs.py` — Access/eviction file pairing logic

The classifier's tests live separately in `evict_classifier/tests/` (run with
`.venv/bin/python -m pytest evict_classifier/tests/`): interval-band brute-force
equivalence, weighted-draw bounds, label/censoring/residency-cap behavior,
export contract (bias/threshold), and an end-to-end train smoke test on
synthetic `.bin` files.

### `notebooks-v2/` — Trace analysis + classifier artifacts

- `binary_log_analysis.ipynb` — event distributions, reaccess probability (by insertion/first-access position), insertion→access deltas, re-insertion timing for the may-28 binary traces. The findings here motivated the Gen 6 label design (time-based fill exclusion, horizon) and killed the insertion-time model.
- `protect-test/{ycsb_b,ycsb_c,ycsb_e}/` — trained Gen 6 artifacts (weights JSON, keras model, discretizer, eval report, weight plot).

### `notebooks/` — Analysis notebooks

Two-phase analytical pipeline:

**Phase 1 — Model prototyping:**
- `initial_parsing.ipynb` — Earliest exploration: log parsing, feature distribution plots, KBinsDiscretizer pipeline, pointwise classifier training with custom activations on the fileserver workload.
- `pairwise_ranker_vis_mongo.ipynb`, `pairwise_ranker_vis_randomread.ipynb`, `pairwise_ranker_vis_varmail.ipynb` — Gen 5 pairwise ranker training and weight visualization for three workloads. Same template notebook applied to different data.

**Phase 2 — Evaluation:**
- `model_metrics_comparison.ipynb` — Cross-workload summary comparing precision/recall/F1/AUC across all 7 workloads from `outputs-final/` eval reports.
- `latency_analysis.ipynb` — Eviction latency comparison (model vs normal/LRU policy) using `logoutputmodel.log` and `logoutputnormal.log`. Model adds ~3.5 us overhead (~13%).
- `{copyfiles,mongo,openfiles,randomread,varmail,webproxy,webserver}_eval_analysis.ipynb` — Per-workload statistical evaluation of model vs. baseline insertion rates. Each loads 50 paired CSV runs, runs paired t-tests with Cohen's d and power analysis, and produces box plots, CDFs, Q-Q plots, and dashboards.

### `visualizations/` — Architecture diagrams

Graphviz `.gv` files (`BT_ranker.gv`, `ebpf.gv`, `model_loading.gv`) and rendered PNGs showing model architecture evolution across generations. Also contains `guard.c` (not a visualization).

## Dependencies

Python 3.12+, managed with `uv`. Core: TensorFlow/Keras, scikit-learn, pandas, matplotlib, seaborn, typer. Dev: pytest, ipykernel.
