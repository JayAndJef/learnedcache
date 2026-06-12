# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The Python side of LearnedCache: the **`evict_classifier`** module trains an eviction-time
binary reuse classifier — `P(page reused within horizon H)` from 9 page/inode features —
and exports quantized integer weights for the `cache_ext_fifo_ml_protect` BPF policy
(in `../cache_ext_lc`). (A trace-driven hit-rate simulator existed through Gen 6
validation and has been removed; evaluation now happens on the VM benchmark.)

The earlier **Gen-5 pairwise ranker** (`learnedcache` package, `cache_ext_fifo_ml`
policy) has been **removed**; `notebooks/` and `visualizations/` remain as historical
archive of Gen 1–5 and may reference deleted code.

## Commands

```bash
# All commands run from this directory with the project venv.
.venv/bin/python -m pytest evict_classifier/tests/     # test suite

# Train + export one model per workload (horizon auto-derived from measured
# cache turnover; pass --horizon <s> / --residency-cap <s> to override,
# --capacity <pages> to condition the turnover on a different cache size)
.venv/bin/python -m evict_classifier train \
  --data-dir data/tracer-bundle-may-28/cache_ext_logs \
  --output-dir <out> --target-rows 5000000
# -> <out>/<workload>/{model_weights.json, model.keras, discretizer.pkl,
#                      eval_report.txt, metrics.json, feature_importance.png}

```

`--horizon` and `--residency-cap` are in **seconds** (trace timestamps are ns; the CLI
converts). When omitted, both default to the **measured cache turnover** — the
fill/rotation time `capacity / insertion_rate` (`loading.estimate_turnover`), which is
the window one CLOCK rotation actually buys a protected page. The estimate uses only
the insertion + eviction logs because they are *complete* event streams (the access log
is a kernel subsample): rate is averaged over active insertion periods (idle gaps > 1 s
excluded), and capacity = insertions before the first eviction (cold-start fill),
overridable via `--capacity`. Caveat: the rate reflects miss pressure at the
*collection* cgroup size — conditioning on a much smaller deployment capacity via
`--capacity` over-states the turnover to first order. Chosen values + provenance are
recorded in each model's `metrics.json`. Dependencies are managed with `uv` (Python
≥3.12; keras/tensorflow, scikit-learn, numpy, typer), but invoke via
`.venv/bin/python -m ...` — the uv script integration is not used.

## Architecture (Gen 6 — eviction-time binary reuse classifier)

At eviction time the kernel policy scans the FIFO list head-first: predicted-reused
folios are *protected* (rotate to the list tail), the rest are evicted oldest-first
until the requested victim count is met (no oversampling). The model
is a linear classifier on quantile-binned one-hot features — `Dense(1, sigmoid,
use_bias=True)` — so in-kernel scoring is an integer sum:
`sum(weights_int[bin(feature)]) + bias_int > threshold_int ⇒ protect`.

History: Gens 1–3 were pointwise reuse-time classifiers; Gen 5 was a pairwise
Bradley-Terry ranker (removed). An insertion-time admission model was explored and
abandoned — at `folio_added` a page has no history, and reuse is inode-homogeneous with
only coarse signal available at insertion (oracle AUC 0.92 vs 0.845 from available
features). Eviction time has the rich page-specific signal.

### Insertion/eviction-independent semantics (critical invariant)

Kernel feature state (`per_folio_map`, `per_file_map`) is written **only at
`folio_accessed`** — insertions log an event but mutate no state; evictions delete
nothing (history persists; map LRU overflow provides forgetting). Features are therefore
pure functions of the access stream, and training reads them directly from the tracer's
logged access records with exact train/serve parity:

- candidate features at eviction moment E = the page's most recent prior access record;
- `TSA = E − prior_access_ts` (the 9th, derived feature);
- never-accessed pages have no state → the policy evicts them first by rule; the model
  never scores them, and training never sees them — consistent by construction.

**Trace provenance:** traces collected before this purge (including
`data/tracer-bundle-may-28`) carry feature fields computed under the old
insertion-coupled semantics. Training on them is self-consistent (fine for local
iteration/simulation) but deployment models should be trained on traces re-collected
with the purged tracer (see `../CLOUD_AGENT.md`).

### How sampling works (`sampling.py`)

Every candidate row is a *(prior access aᵢ of page p, eviction event E)* pair with E in
the interval between aᵢ and p's next access; `label = 1 iff next_access − E < horizon`.
One `lexsort` + a few `searchsorted` calls compute per-access event-index bands
(`_interval_bounds`), then `_draw` samples a bounded, class-balanced set entirely in
numpy — the join costs seconds once, and training runs in-memory (~25 s/workload at 5M
rows). Two corrections:

- **Right-censoring**: events later than `last_access_ts − horizon` are dropped (their
  reuse window is unobservable).
- **Residency cap** (`--residency-cap`): the eviction log has no page identity, so
  without a cap a page counts as a candidate long after it was realistically evicted;
  the cap (≈ cache turnover, `cache_pages / insertion_rate`) removes these phantom easy
  negatives.

Holdout = last 20% of eviction events (temporal split, natural class ratio); threshold
is a logit-space knob exported as `threshold_int` (0 ⇒ P>0.5; balanced training makes
that protect-heavy at the natural ~20% positive rate — tune per workload).

### Module layout (`evict_classifier/`)

| Module | Role |
|---|---|
| `loading.py` | Memory-mapped binary log readers (88-byte access, 8-byte eviction, 32-byte insertion records); workload/iter discovery; `build_pairs_from_binary` yields `(trial_id, access, eviction)` — the insertion log is analysis-only |
| `sampling.py` | Vectorized candidate sampling (bands, weighted draw, censoring, residency cap); verified against an O(n²) brute force in tests |
| `preprocess.py` | `KBinsDiscretizer` (quantile, ordinal) fit/transform + int8 one-hot |
| `models.py` | `build_binary_classifier`: `Dense(1, sigmoid, use_bias=True)`, layer named `ranking_weight` (exporter contract) |
| `train.py` | Per-workload orchestration: sample → fit discretizer → one-hot once → in-memory fit → holdout eval → artifacts |
| `export.py` | BPF JSON: per-feature `bin_edges` (u64-clamped) + `weights_int`, plus top-level `bias_int`/`threshold_int`/`weight_scale` |
| `plots.py` | `feature_importance.png` per-bin weight chart (green=protect, red=evict) |
| `cli.py` / `__main__.py` | Typer app: `train` |
| `tests/` | Brute-force band equivalence, draw bounds, label/censoring/cap, export contract, turnover estimator, end-to-end train smoke |

Feature order is the BPF contract: `["pd","sz","fq","sd","p2","id","i2","ie",
time_since_last_access_at_eviction]` — must match the enum in
`cache_ext_fifo_ml_protect.bpf.c`.

### Reference results (jun-11 traces, purged tracer, auto horizon)

5M rows, auto H (~53-108 s turnover at the 10 G collection cgroup): holdout AUC
ycsb_a 0.903 / b 0.900 / c 0.924 / d 0.989 / e 0.911 / f 0.903. Models:
`output-jun-11/`. ycsb_d has no eviction-policy headroom (compulsory-bound) —
exclude it from hit-rate claims.

## Repository layout

- `data/tracer-bundle-may-28/cache_ext_logs/{ycsb_b,c,e}/iter_1/` — binary traces
  (access/eviction/insertion), **pre-purge provenance**.
- `evict_classifier/` — the module (see table above).
- `notebooks-v2/` — `binary_log_analysis.ipynb` (trace analysis that motivated Gen 6)
  and `protect-test/` (trained model artifacts).
- `notebooks/`, `visualizations/` — **historical archive** (Gens 1–5); reference the
  removed ranker package and old artifact dirs, not expected to run.

## Roadmap

See `../IMPROVEMENTS.md` (Belady-imitation labels are the headline next step) and
`evict_classifier/KNOWN_ISSUES.md` for the current limitation inventory.
