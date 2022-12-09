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
"""Tests that film_efficientnet can detect an image of a cat."""

from absl.testing import parameterized
import numpy as np
from robotics_transformer.film_efficientnet import film_efficientnet_encoder
from skimage import data
import tensorflow as tf


class FilmEfficientnetTest(tf.test.TestCase, parameterized.TestCase):

  def _helper(self, include_film, model_variant):
    if model_variant == 'b0':
      size = 224
      fe = film_efficientnet_encoder.EfficientNetB0
    elif model_variant == 'b1':
      size = 240
      fe = film_efficientnet_encoder.EfficientNetB1
    elif model_variant == 'b2':
      size = 260
      fe = film_efficientnet_encoder.EfficientNetB2
    elif model_variant == 'b3':
      size = 300
      fe = film_efficientnet_encoder.EfficientNetB3
    elif model_variant == 'b4':
      size = 380
      fe = film_efficientnet_encoder.EfficientNetB4
    elif model_variant == 'b5':
      size = 456
      fe = film_efficientnet_encoder.EfficientNetB5
    elif model_variant == 'b6':
      size = 528
      fe = film_efficientnet_encoder.EfficientNetB6
    elif model_variant == 'b7':
      size = 600
      fe = film_efficientnet_encoder.EfficientNetB7
    else:
      raise ValueError(f'Unknown variant: {model_variant}')
    fe = fe(include_top=True, weights='imagenet', include_film=include_film)
    image = np.expand_dims(data.chelsea(), axis=0)
    image = tf.image.resize(image, (size, size))
    context = np.random.randn(1, 512)
    if include_film:
      eff_output = fe(
          (film_efficientnet_encoder.preprocess_input(image), context),
          training=False)
    else:
      eff_output = fe(
          film_efficientnet_encoder.preprocess_input(image), training=False)
    film_preds = film_efficientnet_encoder.decode_predictions(
        eff_output.numpy(), top=10)
    self.assertIn('tabby', [f[1] for f in film_preds[0]])

  @parameterized.parameters([True, False])
  def test_keras_equivalence_b3(self, include_film):
    self._helper(include_film, 'b3')


if __name__ == '__main__':
  tf.test.main()
