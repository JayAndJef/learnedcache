# New model specification: eviction-time pairwise-diff Bradley-Terry ranker

## 1) Objective and model contract

This project trains a **pairwise ranking model** for page-cache eviction.

Active objective is:

- **Pairwise-diff ranking over eviction events** (default and only production objective).
- At each eviction event, compare page candidates from the **same event**.
- Learn \(P(A \text{ reused sooner than } B)\) from one-hot feature difference:
  \[
  x_{\text{diff}} = x_A - x_B,\quad
  y = \mathbb{1}[t_A < t_B]
  \]
  where \(t_A, t_B\) are time-until-next-reuse labels measured from eviction time.

Model form:

- Linear Bradley-Terry style ranker:
  - Input: `feature_diff`
  - Output: sigmoid probability
  - Layer: `Dense(1, activation="sigmoid", use_bias=False, name="ranking_weight")`
- Loss: binary cross-entropy
- Metric: pairwise accuracy

## 2) Data contract and pairing rules

### Input streams

Each trial consists of:

- `<token>_access.csv`
- `<token>_eviction.csv`

### Required columns

Access CSV must contain:

- page key columns: `dm`, `dn`, `in`, `of`
- timestamp: `ts`
- configured discretized feature columns

Eviction CSV must contain:

- timestamp: `ts`

### Pairing and ordering

- Access/eviction files pair by shared token.
- Pairing order is deterministic (token-sorted).
- Token sets must match exactly (strict 1:1 trial pairing).
- Hard-fail on:
  - no access files
  - no eviction files
  - no common tokens
  - any access-only or eviction-only token

No cross-token/trial boundary mixing is allowed.

## 3) Eviction-time supervised base table semantics

For each trial:

1. Sort access by `ts`.
2. Sort eviction by `ts`.
3. Track latest state per page key.
4. At each eviction timestamp:
   - consume accesses with `access_ts <= eviction_ts`
   - emit one row per tracked page with:
     - `trial_id`
     - `eviction_ts`
     - `time_since_last_access_at_eviction`
     - `time_until_next_reuse_from_eviction`
     - page key columns
     - configured feature columns

No-reuse handling:

- If no future reuse exists after eviction for a page, assign
  \[
  \text{no\_reuse\_label} = \max(\text{finite targets}) + 1
  \]
- This makes no-reuse pages naturally worse-ranked than finite-reuse pages.

## 4) Pairwise sample generation policy

Pairwise samples are generated **only within the same** `(trial_id, eviction_ts)` event.

Rules:

- Sample bounded number of pairs per event (`pairs_per_event`) for computational feasibility.
- Drop self-pairs.
- Drop ties where `target_a == target_b`.
- Keep no-reuse pages; they are worse rank due to larger target.
- Labels:
  - `1` if `target_a < target_b` (A reused sooner)
  - `0` otherwise
- Features:
  - `X_diff = X_a - X_b`

Optional global safety cap:

- `max_pairs_total` may subsample final pair set.

## 5) Feature pipeline contract

- Fit `KBinsDiscretizer` on train rows only.
- Transform train/test with same discretizer.
- One-hot encode discretized features into contiguous numeric vectors.
- Build pairwise diffs from encoded row vectors.

Guarantees:

- deterministic feature layout
- bin counts tracked (`n_bins_list`)
- safe bin-index validation

## 6) Train/test split policy

- Split by `trial_id`:
  - train = all but last trial
  - test = last trial
- Hard-fail if fewer than 2 trials or empty train/test split.

## 7) CLI and artifact contract

CLI train commands must expose pairwise sampling controls:

- `--pairs-per-event`
- `--max-pairs-total`
- `--pair-random-state`

Artifacts:

- `model.keras`
- `discretizer.pkl`
- `training_curves.png`
- `feature_importance.png`
- `eval_report.txt`

## 8) Reporting contract (pairwise)

Reports/plots must describe pairwise semantics:

- train/val accuracy
- train/val BCE loss
- confusion matrix on test pairs
- pair counts and tie-drop stats
- label balance

No pointwise-regression wording in active paths.

## 9) Critical test requirements

Minimum required tests include:

- strict access/eviction token-set equality and deterministic pairing
- multi-trial supervised-row isolation (no cross-boundary reuse leakage)
- pairwise sampler tests:
  - same-event-only generation behavior
  - tie-drop behavior
  - deterministic sampling by seed
  - bounded sampling (`pairs_per_event`, optional `max_pairs_total`)
  - no-reuse worst-rank behavior via labels
- export feature/bin shape contract checks

## 10) Validation gate

Before completion, all changes must pass:

1. syntax check
2. pytest suite (including pairwise sampler/core/loading/export tests)
3. artifact/report contract checks for a training run

No implementation is complete unless all gates pass.