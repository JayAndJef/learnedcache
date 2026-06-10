"""Eviction-time binary reuse classifier.

A standalone sibling of the ``learnedcache`` pairwise ranker. Instead of
ranking eviction candidates, it trains a *pointwise* binary classifier that
predicts ``P(page reused within horizon H)`` from the same eviction-time
features, and exports BPF-compatible weights (per-bin weights + bias +
threshold) for the ``cache_ext_fifo_ml_protect`` skip-in-place policy.

The module deliberately *copies* (rather than imports) the pieces of the
feature pipeline it needs from ``learnedcache`` so the ranker stays untouched.
Run via ``python -m evict_classifier``.
"""
