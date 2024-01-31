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
"""Tests for token_learner."""
from absl.testing import parameterized
from robotics_transformer.tokenizers import token_learner
import tensorflow as tf


class TokenLearnerTest(parameterized.TestCase):

  @parameterized.named_parameters(('sample_input', 512, 8))
  def testTokenLearner(self, embedding_dim, num_tokens):
    batch = 1
    seq = 2
    token_learner_layer = token_learner.TokenLearnerModule(
        num_tokens=num_tokens)

    inputvec = tf.random.normal(shape=(batch * seq, 81, embedding_dim))

    learnedtokens = token_learner_layer(inputvec)
    self.assertEqual(learnedtokens.shape,
                     [batch * seq, num_tokens, embedding_dim])


if __name__ == '__main__':
  tf.test.main()
