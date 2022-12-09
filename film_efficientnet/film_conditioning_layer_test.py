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
"""Tests for film_conditioning_layer."""
from absl.testing import parameterized
import numpy as np
from robotics_transformer.film_efficientnet import film_conditioning_layer
import tensorflow as tf


class FilmConditioningLayerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([2, 4])
  def test_film_conditioning_rank_two_and_four(self, conv_rank):
    batch = 2
    num_channels = 3
    if conv_rank == 2:
      conv_layer = np.random.randn(batch, num_channels)
    elif conv_rank == 4:
      conv_layer = np.random.randn(batch, 1, 1, num_channels)
    else:
      raise ValueError(f'Unexpected conv rank: {conv_rank}')
    context = np.random.rand(batch, num_channels)
    film_layer = film_conditioning_layer.FilmConditioning(num_channels)
    out = film_layer(conv_layer, context)
    tf.debugging.assert_rank(out, conv_rank)


if __name__ == '__main__':
  tf.test.main()
