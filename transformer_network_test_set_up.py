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
"""Tests for networks."""

import copy
from typing import Optional, Tuple, Union

from absl.testing import parameterized
import numpy as np
from robotics_transformer import sequence_agent
from robotics_transformer import transformer_network
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

BATCH_SIZE = 2
TIME_SEQUENCE_LENGTH = 3
HEIGHT = 256
WIDTH = 320
NUM_IMAGE_TOKENS = 2


def spec_names_list() -> list[str]:
  """Lists the different types of specs accepted by the transformer."""
  return ['default']


def state_spec_list() -> list[tensorspec_utils.TensorSpecStruct]:
  """Lists the different types of state spec accepted by the transformer."""
  state_spec = tensorspec_utils.TensorSpecStruct()
  state_spec.image = tensor_spec.BoundedTensorSpec([HEIGHT, WIDTH, 3],
                                                   dtype=tf.float32,
                                                   name='image',
                                                   minimum=0.,
                                                   maximum=1.)
  state_spec.natural_language_embedding = tensor_spec.TensorSpec(
      shape=[512], dtype=tf.float32, name='natural_language_embedding')

  state_spec_mask = copy.deepcopy(state_spec)
  state_spec_mask.initial_binary_mask = tensor_spec.BoundedTensorSpec(
      [HEIGHT, WIDTH, 1],
      dtype=tf.int32,
      name='initial_binary_mask',
      minimum=0,
      maximum=255)

  state_spec_tcl = copy.deepcopy(state_spec)
  state_spec_tcl.original_image = tensor_spec.BoundedTensorSpec(
      [HEIGHT, WIDTH, 3],
      dtype=tf.float32,
      name='original_image',
      minimum=0.,
      maximum=1.)

  return [
      state_spec,
      state_spec_mask,
      state_spec_tcl,
  ]


def observations_list(training: bool = True) -> list[dict[str, tf.Tensor]]:
  """Lists the different types of observations accepted by the transformer."""
  if training:
    image_shape = [BATCH_SIZE, TIME_SEQUENCE_LENGTH, HEIGHT, WIDTH, 3]
    emb_shape = [BATCH_SIZE, TIME_SEQUENCE_LENGTH, 512]
    mask_shape = [BATCH_SIZE, TIME_SEQUENCE_LENGTH, HEIGHT, WIDTH, 1]
  else:
    # inference currently only support batch size of 1
    image_shape = [1, HEIGHT, WIDTH, 3]
    emb_shape = [1, 512]
    mask_shape = [1, HEIGHT, WIDTH, 1]
  return [
      {
          'image': tf.constant(0.5, shape=image_shape),
          'natural_language_embedding': tf.constant(1., shape=emb_shape),
      },
      {
          'image': tf.constant(0.5, shape=image_shape),
          'natural_language_embedding': tf.constant(1., shape=emb_shape),
          'initial_binary_mask': tf.constant(192, shape=mask_shape),
      },
      {  # This is used for TCL.
          'image': tf.constant(0.5, shape=image_shape),
          'original_image': tf.constant(0.4, shape=image_shape),
          'natural_language_embedding': tf.constant(1., shape=emb_shape),
      },
  ]


NAME_TO_STATE_SPECS = dict(zip(spec_names_list(), state_spec_list()))
NAME_TO_OBSERVATIONS = dict(zip(spec_names_list(), observations_list()))
NAME_TO_INF_OBSERVATIONS = dict(
    zip(spec_names_list(), observations_list(False)))


class FakeImageTokenizer(tf.keras.layers.Layer):
  """Fake Image Tokenizer for testing Transformer."""

  def __init__(self,
               encoder: ...,
               position_embedding: ...,
               embedding_output_dim: int,
               patch_size: int,
               use_token_learner: bool = False,
               num_tokens: int = NUM_IMAGE_TOKENS,
               use_initial_binary_mask: bool = False,
               **kwargs):
    del encoder, position_embedding, patch_size, use_token_learner
    super().__init__(**kwargs)
    self.tokens_per_context_image = num_tokens
    if use_initial_binary_mask:
      self.tokens_per_context_image += 1
    self.embedding_output_dim = embedding_output_dim
    self.use_initial_binary_mask = use_initial_binary_mask

  def __call__(self,
               image: tf.Tensor,
               context: Optional[tf.Tensor] = None,
               initial_binary_mask: Optional[tf.Tensor] = None,
               training: bool = False) -> tf.Tensor:
    if self.use_initial_binary_mask:
      assert initial_binary_mask is not None
    image_shape = tf.shape(image)
    seq_size = image_shape[1]
    batch_size = image_shape[0]
    all_tokens = []
    num_tokens = self.tokens_per_context_image
    for t in range(seq_size):
      tokens = tf.ones([batch_size, 1, num_tokens, self.embedding_output_dim
                       ]) * image[0][t][0][0]
      all_tokens.append(tokens)
    return tf.concat(all_tokens, axis=1)


class TransformerNetworkTestUtils(tf.test.TestCase, parameterized.TestCase):
  """Defines specs, SequenceAgent, and various other testing utilities."""

  def _define_specs(self,
                    train_batch_size=BATCH_SIZE,
                    inference_batch_size=1,
                    time_sequence_length=TIME_SEQUENCE_LENGTH,
                    inference_sequence_length=TIME_SEQUENCE_LENGTH,
                    token_embedding_size=512,
                    image_width=WIDTH,
                    image_height=HEIGHT):
    """Defines specs and observations (both training and inference)."""
    self.train_batch_size = train_batch_size
    self.inference_batch_size = inference_batch_size
    self.time_sequence_length = time_sequence_length
    self.inference_sequence_length = inference_sequence_length
    self.token_embedding_size = token_embedding_size
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')

    action_spec.rotation_delta = tensor_spec.BoundedTensorSpec(
        (3,),
        dtype=tf.float32,
        minimum=-np.pi / 2,
        maximum=np.pi / 2,
        name='rotation_delta')

    action_spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
        (1,),
        dtype=tf.float32,
        minimum=-1.,
        maximum=1.,
        name='gripper_closedness_action')
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')

    state_spec = tensorspec_utils.TensorSpecStruct()
    state_spec.image = tensor_spec.BoundedTensorSpec(
        [image_height, image_width, 3],
        dtype=tf.float32,
        name='image',
        minimum=0.,
        maximum=1.)
    state_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[self.token_embedding_size],
        dtype=tf.float32,
        name='natural_language_embedding')
    self._policy_info_spec = {
        'return':
            tensor_spec.BoundedTensorSpec((),
                                          dtype=tf.float32,
                                          minimum=0.0,
                                          maximum=1.0,
                                          name='return'),
        'discounted_return':
            tensor_spec.BoundedTensorSpec((),
                                          dtype=tf.float32,
                                          minimum=0.0,
                                          maximum=1.0,
                                          name='discounted_return'),
    }

    self._state_spec = state_spec
    self._action_spec = action_spec

    self._inference_observation = {
        'image':
            tf.constant(
                1,
                shape=[self.inference_batch_size, image_height, image_width, 3],
                dtype=tf.dtypes.float32),
        'natural_language_embedding':
            tf.constant(
                1.,
                shape=[self.inference_batch_size, self.token_embedding_size],
                dtype=tf.dtypes.float32),
    }
    self._train_observation = {
        'image':
            tf.constant(
                0.5,
                shape=[
                    self.train_batch_size, self.time_sequence_length,
                    image_height, image_width, 3
                ]),
        'natural_language_embedding':
            tf.constant(
                1.,
                shape=[
                    self.train_batch_size, self.time_sequence_length,
                    self.token_embedding_size
                ]),
    }
    self._inference_action = {
        'world_vector':
            tf.constant(0.5, shape=[self.inference_batch_size, 3]),
        'rotation_delta':
            tf.constant(0.5, shape=[self.inference_batch_size, 3]),
        'terminate_episode':
            tf.constant(
                [0, 1] * self.inference_batch_size,
                shape=[self.inference_batch_size, 2]),
        'gripper_closedness_action':
            tf.constant(0.5, shape=[self.inference_batch_size, 1]),
    }
    self._train_action = {
        'world_vector':
            tf.constant(
                0.5,
                shape=[self.train_batch_size, self.time_sequence_length, 3]),
        'rotation_delta':
            tf.constant(
                0.5,
                shape=[self.train_batch_size, self.time_sequence_length, 3]),
        'terminate_episode':
            tf.constant(
                [0, 1] * self.train_batch_size * self.time_sequence_length,
                shape=[self.train_batch_size, self.time_sequence_length, 2]),
        'gripper_closedness_action':
            tf.constant(
                0.5,
                shape=[self.train_batch_size, self.time_sequence_length, 1]),
    }

  def _create_agent(self, actor_network=None):
    """Creates SequenceAgent using custom actor_network."""
    time_step_spec = ts.time_step_spec(observation_spec=self._state_spec)
    if actor_network is None:
      actor_network = transformer_network.TransformerNetwork

    self._agent = sequence_agent.SequenceAgent(
        time_step_spec=time_step_spec,
        action_spec=self._action_spec,
        actor_network=actor_network,
        actor_optimizer=tf.keras.optimizers.Adam(),
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
        time_sequence_length=TIME_SEQUENCE_LENGTH)
    self._num_action_tokens = (
        # pylint:disable=protected-access
        self._agent._actor_network._action_tokenizer._tokens_per_action)
    # pylint:enable=protected-access

  def setUp(self):
    self._define_specs()
    super().setUp()

  def get_image_value(self, step_idx: int) -> float:
    return float(step_idx) / self.time_sequence_length

  def get_action_logits(self, batch_size: int, value: int,
                        vocab_size: int) -> tf.Tensor:
    return tf.broadcast_to(
        tf.one_hot(value % vocab_size, vocab_size)[tf.newaxis, tf.newaxis, :],
        [batch_size, 1, vocab_size])

  def create_obs(self, value) -> dict[str, tf.Tensor]:
    observations = {}
    observations['image'] = value * self._inference_observation['image']
    observations[
        'natural_language_embedding'] = value * self._inference_observation[
            'natural_language_embedding']
    return observations

  def fake_action_token_emb(self, action_tokens) -> tf.Tensor:
    """Just pad with zeros."""
    shape = action_tokens.shape
    assert self.vocab_size > self.token_embedding_size
    assert len(shape) == 4
    return action_tokens[:, :, :, :self.token_embedding_size]

  def fake_transformer(
      self, all_tokens, training,
      attention_mask) -> Union[tf.Tensor, Tuple[tf.Tensor, list[tf.Tensor]]]:
    """Fakes the call to TransformerNetwork._transformer."""
    del training
    del attention_mask
    # We expect ST00 ST01 A00 A01...
    # Where:
    # * ST01 is token 1 of state 0.
    # * A01 is token 1 of action 0.
    shape = all_tokens.shape.as_list()
    batch_size = shape[0]
    self.assertEqual(batch_size, 1)
    emb_size = self.token_embedding_size

    # transform to [batch_size, num_tokens, token_size]
    all_tokens = tf.reshape(all_tokens, [batch_size, -1, emb_size])
    # Pads tokens to be of vocab_size.
    self.assertGreater(self.vocab_size, self.token_embedding_size)
    all_shape = all_tokens.shape
    self.assertLen(all_shape.as_list(), 3)
    output_tokens = tf.concat([
        all_tokens,
        tf.zeros([
            all_shape[0], all_shape[1],
            self.vocab_size - self.token_embedding_size
        ])
    ],
                              axis=-1)
    num_tokens_per_step = NUM_IMAGE_TOKENS + self._num_action_tokens
    # Check state/action alignment.
    window_range = min(self._step_idx + 1, self.time_sequence_length)
    for j in range(window_range):
      # The index step that is stored in j = 0.
      first_step_idx = max(0, self._step_idx + 1 - self.time_sequence_length)
      image_idx = j * num_tokens_per_step
      action_start_index = image_idx + NUM_IMAGE_TOKENS
      for t in range(NUM_IMAGE_TOKENS):
        self.assertAllEqual(
            self.get_image_value(first_step_idx + j) *
            tf.ones_like(all_tokens[0][image_idx][:self.token_embedding_size]),
            all_tokens[0][image_idx + t][:self.token_embedding_size])
      # if j is not the current step in the window, all action dimensions
      # from previous steps are already infered and thus can be checked.
      action_dims_range = self.action_inf_idx if j == window_range - 1 else self._num_action_tokens
      for t in range(action_dims_range):
        token_idx = action_start_index + t
        action_value = (first_step_idx + j) * self._num_action_tokens + t
        self.assertAllEqual(
            self.get_action_logits(
                batch_size=batch_size,
                value=action_value,
                vocab_size=self.vocab_size)[0][0][:self.token_embedding_size],
            all_tokens[0][token_idx][:self.token_embedding_size])
    # Output the right action dimension value.
    image_token_index = (
        min(self._step_idx, self.time_sequence_length - 1) *
        num_tokens_per_step)
    transformer_shift = -1
    action_index = (
        image_token_index + NUM_IMAGE_TOKENS + self.action_inf_idx +
        transformer_shift)
    action_value = self._step_idx * self._num_action_tokens + self.action_inf_idx
    action_logits = self.get_action_logits(
        batch_size=batch_size, value=action_value, vocab_size=self.vocab_size)
    output_tokens = tf.concat([
        output_tokens[:, :action_index, :], action_logits[:, :, :],
        output_tokens[:, action_index + 1:, :]
    ],
                              axis=1)
    self.action_inf_idx = (self.action_inf_idx + 1) % self._num_action_tokens
    attention_scores = []
    return output_tokens, attention_scores
