# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                                    # Run all tests
uv run pytest tests/test_core_pairwise.py -k "test_name"  # Run a single test
uv run learnedcache --help                       # CLI overview
uv run learnedcache train-ranker --help          # Help for a specific subcommand
uv run learnedcache train-from-binary --help     # Help for the binary-log training command
```

## Architecture

LearnedCache trains a **linear pairwise-diff ranker** to make cache eviction decisions. The model compares two cached items at eviction time and predicts which should be evicted first (lower reuse time = evict sooner). The trained model is exported as quantized integer weights for deployment in BPF (in-kernel eviction).

### Model evolution

The project went through several generations before landing on the current approach:

- **Gen 1-3** (pointwise classifiers): Predict binned time-until-next-access directly. Used custom Keras activations (`Squaremax`, `TaylorSoftmax`) for multi-class classification. Visualized in `visualizations/` as `gen1_output.png` through `gen3_output.png` and classifier architecture diagrams (`pointwisebinaryclassifier.png`, `squaremax5classifier.png`, `taylormax5classifier.png`).
- **Gen 5** (current — pairwise Bradley-Terry ranker): Instead of predicting absolute reuse time, compares pairs of items and predicts which is reused sooner. Single `Dense(1, sigmoid, use_bias=False)` layer on the feature difference vector. Produces interpretable per-bin weights that map directly to BPF maps.

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

### `tests/` — Test suite

3 test files plus `conftest.py`:

- `test_core_pairwise.py` — Pairwise sampling correctness
- `test_export_contract.py` — BPF export format contract
- `test_loading_pairs.py` — Access/eviction file pairing logic

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
