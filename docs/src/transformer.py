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
"""RT1 decoder transformer.

Copied from:
https://www.tensorflow.org/text/tutorials/transformer#decoder
"""
from typing import Tuple, Union

import tensorflow as tf


class _TransformerLayer(tf.keras.layers.Layer):
  """A single transformer block."""

  def __init__(self,
               layer_size: int = 4096,
               num_heads: int = 8,
               feed_forward_size: int = 512,
               dropout_rate: float = 0.1,
               return_attention_scores: bool = False):
    """Creates a Transformer layer.

    Args:
      layer_size: Size of the multiple head attention layer.
      num_heads: Number of heads for the multiple head attention layer.
      feed_forward_size: Dimensionality of the feed_forward layer.
      dropout_rate: Dropout rate.
      return_attention_scores: Return attention scores.
    """
    super(_TransformerLayer, self).__init__()

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.mha1 = tf.keras.layers.MultiHeadAttention(
        key_dim=layer_size, num_heads=num_heads, dropout=dropout_rate)
    self.ff = tf.keras.layers.Dense(feed_forward_size)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout_ff = tf.keras.layers.Dropout(dropout_rate)
    self._return_attention_scores = return_attention_scores

  def call(self, x: tf.Tensor, attention_mask: tf.Tensor,
           training: bool) -> Tuple[tf.Tensor, Union[tf.Tensor, None]]:
    """Calls the layer.

    Args:
      x: Input Tensor of shape `(B, T, dim)`.
      attention_mask: a boolean mask of shape `(B, T, T)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).

    Returns:
      y: Output Tensor of shape `(B, T, dim)`. Also return the attention scores
      of shape `(B, T, dim)` or None.
    """
    x1 = self.layernorm1(x)
    mha_results = self.mha1(
        query=x1,
        key=x1,
        value=x1,
        attention_mask=attention_mask,
        return_attention_scores=self._return_attention_scores,
        training=training)
    if self._return_attention_scores:
      x1, score = mha_results
    else:
      x1, score = mha_results, None

    x = x + x1

    y = self.layernorm2(x)
    ff_y = self.ff(y)
    ff_y = self.dropout_ff(ff_y, training=training)
    x = x + ff_y
    return x, score


class Transformer(tf.keras.layers.Layer):
  """A decoder only transformer."""

  def __init__(self,
               num_layers: int = 1,
               layer_size: int = 4096,
               num_heads: int = 8,
               feed_forward_size: int = 512,
               dropout_rate: float = 0.1,
               vocab_size: int = 256,
               return_attention_scores: bool = False):
    """Creates a transformer.

    Args:
      num_layers: Number of transformer layers.
      layer_size: Size of the multiple head attention layer.
      num_heads: Number of heads for the multiple head attention layer.
      feed_forward_size: Dimensionality of the feed_forward layer.
      dropout_rate: Dropout rate.
      vocab_size: Dimensionality of tokens from the output layer.
      return_attention_scores: Return attention scores.
    """
    super(Transformer, self).__init__()

    self._layers = [
        _TransformerLayer(  # pylint: disable=g-complex-comprehension
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            return_attention_scores=return_attention_scores)
        for _ in range(num_layers)
    ]
    self._token_emb = tf.keras.layers.Dense(feed_forward_size)
    self._position_emb = tf.keras.layers.Dense(feed_forward_size)
    self._output_tokens = tf.keras.layers.Dense(vocab_size)

  def call(
      self,
      x: tf.Tensor,
      training: bool,
      attention_mask: tf.Tensor,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, list[tf.Tensor]]]:
    """Calls the layer.

    Args:
      x: Input Tensor of shape `(B, T, dim)`.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
      attention_mask: a boolean mask of shape `(B, T, T)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.

    Returns:
      x: Output Tensor of shape `(B, T, vocab_size)`. If
      `return_attention_scores`, also return attention scores of
      a list of `layer` of elements with shape `(B, T, dim)`.
    """

    seq_len = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]

    positions = tf.one_hot(
        tf.tile(tf.expand_dims(tf.range(0, seq_len, 1), 0), [batch_size, 1]),
        seq_len)

    x = self._token_emb(x)
    x += self._position_emb(positions)
    scores = []

    for layer in self._layers:
      x, score = layer(x, attention_mask=attention_mask, training=training)
      if score is not None:
        scores.append(score)
    x = self._output_tokens(x)
    return x, scores
