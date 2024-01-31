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
"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""
from typing import Optional
from robotics_transformer.film_efficientnet import pretrained_efficientnet_encoder
from robotics_transformer.tokenizers import token_learner
import tensorflow as tf


class RT1ImageTokenizer(tf.keras.layers.Layer):
  """Tokenizes based on vocab size."""

  def __init__(self,
               embedding_output_dim: int,
               use_token_learner: bool = False,
               num_tokens: int = 8,
               **kwargs):
    """Instantiates a RT1ImageTokenizer.

    Args:
      embedding_output_dim: The output size of the tokens.
      use_token_learner: Whether to use token learner. See
        https://arxiv.org/abs/2106.11297
      num_tokens: Relevant only for token learner - the number of learned
        tokens.
      **kwargs: Keyword arguments to base class.
    """
    super().__init__(**kwargs)
    self._embedding_output_dim = embedding_output_dim

    self._tokenizer = pretrained_efficientnet_encoder.EfficientNetEncoder(
        pooling=False, early_film=True)

    self._use_token_learner = use_token_learner
    if self._use_token_learner:
      self._num_tokens = num_tokens
      self._token_learner = token_learner.TokenLearnerModule(
          num_tokens=self._num_tokens)

  @property
  def tokens_per_context_image(self) -> int:
    if self._use_token_learner:
      num_tokens = self._num_tokens
    else:
      num_tokens = 81
    return num_tokens

  def __call__(self,
               image: tf.Tensor,
               context: Optional[tf.Tensor] = None,
               training: bool = False) -> tf.Tensor:
    """Gets image tokens.

    Args:
      image: Images of shape (b, t, h, w, 3) to tokenize.
      context: An optional context vector (e.g., a natural language embedding).
        Expected to have shape (b, t, embedding_dim).
      training: Whether or not we are in training mode.

    Returns:
      tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
    """
    image_shape = tf.shape(image)
    b = image_shape[0]
    t = image_shape[1]
    h = image_shape[2]
    w = image_shape[3]
    c = image_shape[4]

    # Fold the time axis into the batch axis.
    image = tf.reshape(image, [b * t, h, w, c])
    if context is not None:
      context_rank = tf.rank(context)
      assertion = tf.Assert(context_rank == 3, data=[context_rank])
      with tf.control_dependencies([assertion]):
        context = tf.reshape(context, [b * t, tf.shape(context)[-1]])
    tokens = self.get_image_embeddings(image, context, training)
    if self._use_token_learner:
      tokens = self._token_learner(tokens, training)
    # Unflatten the time axis, which was previously flattened into the batch.
    tokens = tf.reshape(tokens, [b, t, tf.shape(tokens)[1], -1])
    return tokens

  def get_image_embeddings(self,
                           image: tf.Tensor,
                           context: Optional[tf.Tensor],
                           training: bool = False) -> tf.Tensor:
    """Gets embeddings from image.

    Args:
      image: Expected to be float32 in range [0, 1] with shape (b, h, w, 3).
      context: Expected to be float32 with shape (b, embedding_dim)
      training: Whether or not we are in training mode.

    Returns:
      tokens of shape (b, num_tokens, emedding_dim)
    """
    image_tokens = self._tokenizer(image, context=context, training=training)
    image_tokens = tf.reshape(image_tokens, [-1, 81, 512])
    return image_tokens
