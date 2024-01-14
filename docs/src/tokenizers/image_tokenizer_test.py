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
"""Tests for image_tokenizer."""
from absl.testing import parameterized
from robotics_transformer.tokenizers import image_tokenizer
import tensorflow as tf


class ImageTokenizerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('sample_image', 512, 224, False, 8),
      ('sample_image_token_learner', 512, 224, True, 8))
  def testTokenize(self, output_dim, image_resolution, use_token_learner,
                   num_tokens):
    batch = 1
    seq = 2
    tokenizer = image_tokenizer.RT1ImageTokenizer(
        embedding_output_dim=output_dim,
        use_token_learner=use_token_learner,
        num_tokens=num_tokens)

    image = tf.random.normal(
        shape=(batch, seq, image_resolution, image_resolution, 3))
    image = tf.clip_by_value(image, 0.0, 1.0)
    context_vector = tf.random.uniform((batch, seq, 512))
    image_tokens = tokenizer(image, context_vector)
    if use_token_learner:
      self.assertEqual(image_tokens.shape, [batch, seq, num_tokens, 512])
    else:
      self.assertEqual(image_tokens.shape, [batch, seq, 81, 512])


if __name__ == '__main__':
  tf.test.main()
