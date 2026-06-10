"""Keras model for the eviction-time binary reuse classifier."""

from __future__ import annotations

import keras
from keras import layers

# Shared with the ranker's exporter contract: the weight-bearing Dense layer is
# named "ranking_weight" so weight extraction is identical. Unlike the ranker,
# this layer HAS a bias -- it becomes the in-kernel decision-threshold offset.
WEIGHT_LAYER_NAME = "ranking_weight"


def build_binary_classifier(n_encoded_features: int) -> keras.Model:
    """Build a linear pointwise binary reuse classifier.

    ``Input(one-hot features) -> Dense(1, sigmoid, use_bias=True)`` predicting
    ``P(page reused within horizon H)``. Linear-on-one-hot keeps the model a
    sum of per-bin weights plus a bias, which maps directly onto the BPF policy's
    integer score: ``sum(weight[bin(feature)]) + bias > threshold``.
    """
    inputs = layers.Input(shape=(n_encoded_features,), name="features")
    outputs = layers.Dense(
        1, activation="sigmoid", use_bias=True, name=WEIGHT_LAYER_NAME
    )(inputs)
    return keras.Model(inputs=inputs, outputs=outputs, name="BinaryReuseClassifier")
