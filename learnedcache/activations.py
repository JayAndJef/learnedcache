"""
Custom activation functions optimized for quantization to int on embedded systems.

These alternatives to softmax are designed to be more hardware-friendly:
- squaremax: Uses square operations instead of exp
- taylor_softmax: Uses Taylor series approximation of exp
"""

import tensorflow as tf
import keras


@keras.utils.register_keras_serializable(package="learnedcache")
def squaremax(x, axis=-1):
    """
    Squaremax activation function - a quantization-friendly alternative to softmax.
    
    Instead of using exp(x), uses x^2 which is easier to implement in integer arithmetic.
    Formula: squaremax(x) = (x - max(x))^2 / sum((x - max(x))^2)
    
    This avoids exponentials which are expensive on embedded systems and can overflow
    with integer representations.
    
    Parameters
    ----------
    x : tensor
        Input tensor (logits)
    axis : int, default=-1
        Axis along which to compute squaremax
        
    Returns
    -------
    tensor
        Squaremax activations (all non-negative, sum to 1 along axis)
    """
    x_shifted = x - tf.reduce_max(x, axis=axis, keepdims=True)
    
    x_squared = tf.square(x_shifted)
    
    sum_squared = tf.reduce_sum(x_squared, axis=axis, keepdims=True)
    
    return x_squared / (sum_squared + 1e-8)


@keras.utils.register_keras_serializable(package="learnedcache")
def taylor_softmax(x, axis=-1, order=3):
    """
    Taylor series approximation of softmax - quantization-friendly.
    
    Uses Taylor series expansion: exp(x) ≈ 1 + x + x^2/2! + x^3/3! + ...
    This replaces the exponential with polynomial operations that are easier
    to implement in fixed-point integer arithmetic on embedded systems.
    
    Parameters
    ----------
    x : tensor
        Input tensor (logits)
    axis : int, default=-1
        Axis along which to compute softmax
    order : int, default=3
        Order of Taylor series (higher = more accurate but more computation)
        
    Returns
    -------
    tensor
        Approximate softmax activations (sum to ~1 along axis)
    """
    x_shifted = x - tf.reduce_max(x, axis=axis, keepdims=True)
    
    exp_approx = tf.ones_like(x_shifted) 
    
    x_power = x_shifted
    factorial = 1.0
    
    for n in range(1, order + 1):
        factorial *= n
        exp_approx += x_power / factorial
        x_power *= x_shifted
    
    exp_approx = tf.maximum(exp_approx, 1e-8)
    
    sum_exp = tf.reduce_sum(exp_approx, axis=axis, keepdims=True)
    
    return exp_approx / (sum_exp + 1e-8)


@keras.utils.register_keras_serializable(package="learnedcache")
class Squaremax(keras.layers.Layer):
    """Squaremax activation layer."""
    
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return squaremax(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable(package="learnedcache")
class TaylorSoftmax(keras.layers.Layer):
    """Taylor softmax activation layer."""
    
    def __init__(self, axis=-1, order=3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.order = order
        
    def call(self, inputs):
        return taylor_softmax(inputs, axis=self.axis, order=self.order)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "order": self.order})
        return config