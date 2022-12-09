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
"""Tests for pretrained_efficientnet_encoder."""

import numpy as np
from robotics_transformer.film_efficientnet import film_efficientnet_encoder
from robotics_transformer.film_efficientnet import pretrained_efficientnet_encoder as eff
from skimage import data
import tensorflow as tf


class PretrainedEfficientnetEncoderTest(tf.test.TestCase):

  def test_encoding(self):
    """Test that we get a correctly shaped decoding."""
    state = np.random.RandomState(0)
    context = state.uniform(-1, 1, (10, 512))
    model = eff.EfficientNetEncoder()
    image = np.expand_dims(data.chelsea(), axis=0) / 255
    preds = model(image, context, training=False).numpy()
    self.assertEqual(preds.shape, (10, 512))

  def test_imagenet_classification(self):
    """Test that we can correctly classify an image of a cat."""
    state = np.random.RandomState(0)
    context = state.uniform(-1, 1, (10, 512))
    model = eff.EfficientNetEncoder(include_top=True)
    image = np.expand_dims(data.chelsea(), axis=0) / 255
    preds = model._encode(image, context, training=False).numpy()
    predicted_names = [
        n[1]
        for n in film_efficientnet_encoder.decode_predictions(preds, top=3)[0]
    ]
    self.assertIn('tabby', predicted_names)


if __name__ == '__main__':
  tf.test.main()
