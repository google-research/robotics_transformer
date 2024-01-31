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
"""Tests for transformer."""
from absl.testing import parameterized
from robotics_transformer import transformer
import tensorflow as tf


class TransformerTest(parameterized.TestCase):

  def setUp(self):
    self._vocab_size = 10
    batch_size = 8
    sequence_len = 12
    self._tokens = tf.random.uniform(
        [batch_size, sequence_len, self._vocab_size],
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
    )
    super(TransformerTest, self).setUp()

  @parameterized.parameters(True, False)
  def test_transformer_forwardpass(self, return_attention_scores):
    network = transformer.Transformer(
        num_layers=2,
        layer_size=512,
        num_heads=4,
        feed_forward_size=256,
        dropout_rate=0.1,
        vocab_size=self._vocab_size,
        return_attention_scores=return_attention_scores)

    output_tokens, attention_scores = network(self._tokens, attention_mask=None)
    self.assertSequenceEqual(self._tokens.shape.as_list(),
                             output_tokens.shape.as_list())
    if return_attention_scores:
      self.assertNotEmpty(attention_scores)
    else:
      self.assertEmpty(attention_scores)


if __name__ == '__main__':
  tf.test.main()
