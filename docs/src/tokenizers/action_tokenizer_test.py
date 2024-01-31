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
"""Tests for action_tokenizer."""
import numpy as np
from robotics_transformer.tokenizers import action_tokenizer
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from tf_agents.specs import tensor_spec


class ActionTokenizerTest(tf.test.TestCase):

  def testTokenize_int32(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')
    tokenizer = action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)
    self.assertEqual(1, tokenizer.tokens_per_action)
    action = tensorspec_utils.TensorSpecStruct(terminate_episode=[0, 1])
    action_tokens = tokenizer.tokenize(action)
    self.assertEqual([1], action_tokens.numpy())

  def testTokenize_int32_not_one_hot(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')
    tokenizer = action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)
    self.assertEqual(1, tokenizer.tokens_per_action)
    action = tensorspec_utils.TensorSpecStruct(terminate_episode=[1, 8])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      tokenizer.tokenize(action)

  def testDetokenize_int32(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')
    tokenizer = action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)
    # 0 token should become a one hot: [1, 0]
    action = tokenizer.detokenize(tf.constant([0], dtype=tf.int32))
    self.assertSequenceEqual([1, 0], list(action['terminate_episode'].numpy()))
    # 1 token should become a one hot: [0, 1]
    action = tokenizer.detokenize(tf.constant([1], dtype=tf.int32))
    self.assertSequenceEqual([0, 1], list(action['terminate_episode'].numpy()))
    # OOV 3 token should become a default one hot: [1, 0]
    action = tokenizer.detokenize(tf.constant([3], dtype=tf.int32))
    self.assertSequenceEqual([1, 0], list(action['terminate_episode'].numpy()))

  def testTokenize_float(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')
    tokenizer = action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)
    self.assertEqual(3, tokenizer.tokens_per_action)
    action = tensorspec_utils.TensorSpecStruct(world_vector=[0.1, 0.5, -0.8])
    action_tokens = tokenizer.tokenize(action)
    self.assertSequenceEqual([4, 6, 0], list(action_tokens.numpy()))

  def testTokenize_float_with_time_dimension(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')
    tokenizer = action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)
    self.assertEqual(3, tokenizer.tokens_per_action)
    batch_size = 2
    time_dimension = 3
    action = tensorspec_utils.TensorSpecStruct(
        world_vector=tf.constant(
            [[0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8],
             [0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8]],
            shape=[batch_size, time_dimension, tokenizer.tokens_per_action]))
    action_tokens = tokenizer.tokenize(action)
    self.assertSequenceEqual(
        [batch_size, time_dimension, tokenizer.tokens_per_action],
        action_tokens.shape.as_list())

  def testTokenize_float_at_limits(self):
    minimum = -1.
    maximum = 1.
    vocab_size = 10
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (2,),
        dtype=tf.float32,
        minimum=minimum,
        maximum=maximum,
        name='world_vector')
    tokenizer = action_tokenizer.RT1ActionTokenizer(
        action_spec, vocab_size=vocab_size)
    self.assertEqual(2, tokenizer.tokens_per_action)
    action = tensorspec_utils.TensorSpecStruct(world_vector=[minimum, maximum])
    action_tokens = tokenizer.tokenize(action)
    # Minimum value will go to 0
    # Maximum value witll go to vocab_size-1
    self.assertSequenceEqual([0, vocab_size - 1], list(action_tokens.numpy()))

  def testTokenize_invalid_action_spec_shape(self):
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (2, 2), dtype=tf.float32, minimum=1, maximum=-1, name='world_vector')
    with self.assertRaises(ValueError):
      action_tokenizer.RT1ActionTokenizer(action_spec, vocab_size=10)

  def testTokenizeAndDetokenizeIsEqual(self):
    action_spec = tensorspec_utils.TensorSpecStruct()

    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')

    action_spec.rotation_delta = tensor_spec.BoundedTensorSpec(
        (3,),
        dtype=tf.float32,
        minimum=-np.pi / 2.,
        maximum=np.pi / 2.,
        name='rotation_delta')

    action_spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
        (1,),
        dtype=tf.float32,
        minimum=-1.,
        maximum=1.,
        name='gripper_closedness_action')

    num_sub_action_space = 2
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (num_sub_action_space,),
        dtype=tf.int32,
        minimum=0,
        maximum=1,
        name='terminate_episode')

    tokenizer = action_tokenizer.RT1ActionTokenizer(
        action_spec,
        vocab_size=1024,
        action_order=[
            'terminate_episode', 'world_vector', 'rotation_delta',
            'gripper_closedness_action'
        ])
    self.assertEqual(8, tokenizer.tokens_per_action)

    # Repeat the following test N times with fuzzy inputs.
    n_repeat = 10
    for _ in range(n_repeat):
      action = tensorspec_utils.TensorSpecStruct(
          world_vector=np.random.uniform(low=-1., high=1.0, size=3),
          rotation_delta=np.random.uniform(
              low=-np.pi / 2., high=np.pi / 2., size=3),
          gripper_closedness_action=np.random.uniform(low=0., high=1.0, size=1),
          terminate_episode=[0, 1])
      action_tokens = tokenizer.tokenize(action)
      policy_action = tokenizer.detokenize(action_tokens)

      for k in action:
        self.assertSequenceAlmostEqual(
            action[k], policy_action[k].numpy(), places=2)

      # Repeat the test with batched actions
      batched_action = tensorspec_utils.TensorSpecStruct(
          world_vector=[
              np.random.uniform(low=-1., high=1.0, size=3),
              np.random.uniform(low=-1., high=1.0, size=3)
          ],
          rotation_delta=[
              np.random.uniform(low=-np.pi / 2., high=np.pi / 2., size=3),
              np.random.uniform(low=-np.pi / 2., high=np.pi / 2., size=3)
          ],
          gripper_closedness_action=[
              np.random.uniform(low=0., high=1.0, size=1),
              np.random.uniform(low=0., high=1.0, size=1)
          ],
          terminate_episode=[[0, 1], [1, 0]])
      action_tokens = tokenizer.tokenize(batched_action)
      policy_action = tokenizer.detokenize(action_tokens)

      for k in batched_action:
        for a, policy_a in zip(batched_action[k], policy_action[k].numpy()):
          self.assertSequenceAlmostEqual(a, policy_a, places=2)


if __name__ == '__main__':
  tf.test.main()
