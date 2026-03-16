import keras
from keras import layers

def build_pairwise_diff_model(n_encoded_features: int) -> keras.Model:
    """Build a linear Bradley-Terry style pairwise-diff ranker."""
    input_diff = layers.Input(shape=(n_encoded_features,), name="feature_diff")
    output = layers.Dense(1, activation="sigmoid", use_bias=False, name="ranking_weight")(input_diff)
    model = keras.Model(inputs=input_diff, outputs=output, name="LinearPairwiseRanker")
    return model

def build_model(n_encoded_features: int) -> keras.Model:
    """Backward-compatible alias for pairwise-diff ranker builder."""
    return build_pairwise_diff_model(n_encoded_features)