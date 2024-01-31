# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TF implementation of Token Learner(Ryoo et al 2021)."""

import functools
from typing import Optional, Sequence, Union
import numpy as np
import tensorflow as tf


def gelu(x: float) -> float:
  return 0.5 * x * (1 +
                    tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def _maybe_dropout(rate: float = 0.0, name: str = "dropout"):
  """Helper function to return dropout layer if rate is non zero."""
  if rate:
    return tf.keras.layers.Dropout(rate, name=name)
  return lambda x, *args: x  # Does nothing to x.


class MlpBlock(tf.keras.layers.Layer):
  """Transformer MLP / feed-forward block."""

  def __init__(self,
               *,
               mlp_dim: int,
               out_dim: Optional[int] = None,
               kernel_init: Optional[tf.keras.initializers.Initializer] = tf
               .keras.initializers.glorot_uniform(),
               bias_init: Optional[tf.keras.initializers.Initializer] = tf.keras
               .initializers.RandomNormal(stddev=1e-6),
               dropout_rate: float = 0.1,
               **kwargs):
    """Initializer for the MLP Block.

    This computes outer_dense(gelu(hidden_dense(input))), with dropout
    applied as necessary.

    Note: Especially outside a keras workflow, make sure to call layer.build

    Args:
      mlp_dim: The dimension of the inner representation (output of hidden
        layer). Usually larger than the input/output dim.
      out_dim: The output dimension of the block. If None, the model output dim
        is equal to the input dim (usually desired)
      kernel_init: Initializer for dense kernels, used for both dense layers.
      bias_init: Initializer for dense biases, used for both dense layers.
      dropout_rate: Dropout rate to be applied after dense ( & activation)
      **kwargs: Other keyword args passed to the tf.keras.layers.Layer
        constructor e.g. the name
    """
    super().__init__(**kwargs)
    self._out_dim = out_dim
    self._hidden_dropout = _maybe_dropout(dropout_rate)
    self._output_dropout = _maybe_dropout(dropout_rate)
    self._hidden_layer = tf.keras.layers.Dense(
        mlp_dim,
        activation=gelu,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name="hidden_dense")

    # If out_dim is None, infer out_dim = input_dim at self.build()
    self._output_layer = functools.partial(
        tf.keras.layers.Dense,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name="final_dense")

  def build(self, input_shape: Sequence[int]):
    out_dim = self._out_dim or input_shape[-1]
    self._output_layer = self._output_layer(units=out_dim)
    super().build(input_shape)

  def call(self,
           inputs: tf.Tensor,
           *,
           is_training: Union[bool, tf.Tensor] = False) -> tf.Tensor:
    """Applies Transformer MlpBlock module."""
    x = self._hidden_layer(inputs)
    x = self._hidden_dropout(x, is_training)
    x = self._output_layer(x)
    x = self._output_dropout(x, is_training)
    return x


class TokenLearnerModule(tf.keras.layers.Layer):
  """TokenLearner module V1.1 (https://arxiv.org/abs/2106.11297)."""

  def __init__(self,
               num_tokens: int,
               bottleneck_dim: int = 64,
               dropout_rate: float = 0.):
    super().__init__()

    self.mlp = MlpBlock(
        mlp_dim=bottleneck_dim, out_dim=num_tokens, dropout_rate=dropout_rate)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    if len(inputs.shape) == 4:
      bs, h, w, c = inputs.shape
      inputs = tf.reshape(inputs, [bs, h * w, c])

    selected = self.layernorm(inputs)

    selected = self.mlp(
        selected, is_training=training)  # Shape: [bs, h*w, n_token].

    selected = tf.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
    selected = tf.nn.softmax(selected, axis=-1)

    feat = tf.einsum("...si,...id->...sd", selected, inputs)

    return feat  # Shape: [bs, n_token, c]
