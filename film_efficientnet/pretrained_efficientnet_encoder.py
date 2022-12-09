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
"""Encoder based on Efficientnet."""

from typing import Optional

import gin
from robotics_transformer.film_efficientnet import film_conditioning_layer
from robotics_transformer.film_efficientnet import film_efficientnet_encoder
import tensorflow as tf

_MODELS = {
    'b3': film_efficientnet_encoder.EfficientNetB3,
}

_SIZES = {
    'b3': 300,
}


@gin.configurable
class EfficientNetEncoder(tf.keras.layers.Layer):
  """Applies a pretrained Efficientnet based encoder."""

  def __init__(self,
               model_variant: str = 'b3',
               freeze: bool = False,
               early_film: bool = True,
               weights: Optional[str] = 'imagenet',
               include_top: bool = False,
               pooling: bool = True,
               **kwargs):
    """Initialize the model.

    Args:
      model_variant: One of 'b0-b7' of the efficient encoders. See
        https://arxiv.org/abs/1905.11946 to understand the variants.
      freeze: Whether or not to freeze the pretrained weights (seems to not work
        well).
      early_film: Whether to inject film layers into the efficientnet encoder
        (seems to be essential to getting strong performance).
      weights: Which pretrained weights to use. Either 'imagenet', a path to the
        pretrained weights, or None for from scratch.
      include_top: Whether to add the top fully connected layer. If True, this
        will cause encoding to fail and is used only for unit testing purposes.
      pooling: If false, returns feature map before global average pooling
      **kwargs: Keras specific layer kwargs.
    """
    super(EfficientNetEncoder, self).__init__(**kwargs)
    if model_variant not in _MODELS:
      raise ValueError(f'Unknown variant {model_variant}')
    self.model_variant = model_variant
    self.early_film = early_film
    self.freeze = freeze
    self.conv1x1 = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling())
    self.net = _MODELS[model_variant](
        include_top=include_top,
        weights=weights,
        include_film=early_film,
    )
    self.film_layer = film_conditioning_layer.FilmConditioning(num_channels=512)
    self._pooling = pooling

  def _prepare_image(self, image: tf.Tensor) -> tf.Tensor:
    """Resize the input image and check that the range is correct."""
    if len(image.shape) != 4 or image.shape[-1] != 3:
      raise ValueError('Provided image should have shape (b, h, w, 3).')
    size = _SIZES[self.model_variant]
    if image.shape[1] < size / 4 or image.shape[2] < size / 4:
      raise ValueError('Provided image is too small.')
    if image.shape[1] > size * 4 or image.shape[2] > size * 4:
      raise ValueError('Provided image is too large.')
    image = tf.image.resize(image, (size, size))
    c1 = tf.Assert(tf.reduce_max(image) <= 1, data=[tf.reduce_max(image)])
    c2 = tf.Assert(tf.reduce_min(image) >= 0, data=[tf.reduce_min(image)])
    with tf.control_dependencies([c1, c2]):
      image *= 255  # The image is expected to be in range(0, 255).
      image = film_efficientnet_encoder.preprocess_input(image)
      return image

  def _encode(self, image: tf.Tensor, context: tf.Tensor,
              training: bool) -> tf.Tensor:
    """Run the image through the efficientnet encoder."""
    image = self._prepare_image(image)
    if self.early_film:
      return self.net((image, context), training=training)
    return self.net(image, training=training)

  def call(self,
           image: tf.Tensor,
           context: Optional[tf.Tensor] = None,
           training: bool = True) -> tf.Tensor:
    if self.freeze:
      features = tf.stop_gradient(self._encode(image, context, training))
    else:
      features = self._encode(image, context, training)
    if context is not None:
      features = self.conv1x1(features)
      features = self.film_layer(features, context)

    if not self._pooling:
      return features

    # Global average pool.
    return tf.reduce_mean(features, [1, 2])
