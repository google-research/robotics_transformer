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

from absl.testing import parameterized

from robotics_transformer import transformer_network
from robotics_transformer.transformer_network_test_set_up import BATCH_SIZE
from robotics_transformer.transformer_network_test_set_up import NAME_TO_INF_OBSERVATIONS
from robotics_transformer.transformer_network_test_set_up import NAME_TO_STATE_SPECS
from robotics_transformer.transformer_network_test_set_up import observations_list
from robotics_transformer.transformer_network_test_set_up import spec_names_list
from robotics_transformer.transformer_network_test_set_up import state_spec_list
from robotics_transformer.transformer_network_test_set_up import TIME_SEQUENCE_LENGTH
from robotics_transformer.transformer_network_test_set_up import TransformerNetworkTestUtils

import tensorflow as tf
from tf_agents.specs import tensor_spec


class TransformerNetworkTest(TransformerNetworkTestUtils):

  # pylint:disable=g-complex-comprehension
  @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'state_spec': spec,
      'train_observation': obs,
  } for (name, spec,
         obs) in zip(spec_names_list(), state_spec_list(), observations_list())]
                                 )
  # pylint:enable=g-complex-comprehension
  def testTransformerTrainLossCall(self, state_spec, train_observation):
    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=self._action_spec,
        time_sequence_length=TIME_SEQUENCE_LENGTH)

    network.create_variables()
    self.assertNotEmpty(network.variables)

    network.set_actions(self._train_action)
    network_state = tensor_spec.sample_spec_nest(
        network.state_spec, outer_dims=[BATCH_SIZE])
    output_actions, network_state = network(
        train_observation, step_type=None, network_state=network_state)
    expected_shape = [2, 3]
    self.assertEqual(network.get_actor_loss().shape,
                     tf.TensorShape(expected_shape))
    self.assertCountEqual(self._train_action.keys(), output_actions.keys())

  # pylint:disable=g-complex-comprehension
  @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'spec_name': name,
  } for name in spec_names_list()])
  # pylint:enable=g-complex-comprehension
  def testTransformerInferenceLossCall(self, spec_name):
    state_spec = NAME_TO_STATE_SPECS[spec_name]
    observation = NAME_TO_INF_OBSERVATIONS[spec_name]

    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=self._action_spec,
        time_sequence_length=TIME_SEQUENCE_LENGTH,
        action_order=[
            'terminate_episode', 'world_vector', 'rotation_delta',
            'gripper_closedness_action'
        ])
    network.create_variables()
    self.assertNotEmpty(network.variables)

    network.set_actions(self._inference_action)
    # inference currently only support batch size of 1
    network_state = tensor_spec.sample_spec_nest(
        network.state_spec, outer_dims=[1])

    output_actions, network_state = network(
        observation, step_type=None, network_state=network_state)

    tf.debugging.assert_equal(network.get_actor_loss(), 0.0)
    self.assertCountEqual(self._inference_action.keys(), output_actions.keys())

  # pylint:disable=g-complex-comprehension
  @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'state_spec': spec,
      'train_observation': obs,
  } for name, spec, obs in zip(spec_names_list(), state_spec_list(),
                               observations_list())])
  # pylint:enable=g-complex-comprehension
  def testTransformerLogging(self, state_spec, train_observation):
    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=self._action_spec,
        time_sequence_length=TIME_SEQUENCE_LENGTH,
        action_order=[
            'terminate_episode', 'world_vector', 'rotation_delta',
            'gripper_closedness_action'
        ])

    network.create_variables()
    self.assertNotEmpty(network.variables)

    network.set_actions(self._train_action)
    network_state = tensor_spec.sample_spec_nest(
        network.state_spec, outer_dims=[BATCH_SIZE])
    _ = network(train_observation, step_type=None, network_state=network_state)
    network.add_summaries(
        train_observation,
        network.get_aux_info(),
        debug_summaries=True,
        training=True)

  # pylint:disable=g-complex-comprehension
  @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'state_spec': spec,
  } for name, spec in zip(spec_names_list(), state_spec_list())])
  # pylint:enable=g-complex-comprehension
  def testTransformerCausality(self, state_spec):
    """Tests the causality for the transformer.

    Args:
      state_spec: Which state spec to test the transformer with
    """
    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=self._action_spec,
        time_sequence_length=TIME_SEQUENCE_LENGTH)
    network.create_variables()
    self.assertNotEmpty(network.variables)

    time_sequence_length = network._time_sequence_length
    tokens_per_image = network._tokens_per_context_image
    tokens_per_action = network._tokens_per_action

    def _split_image_and_action_tokens(all_tokens):
      image_start_indices = [(tokens_per_image + tokens_per_action) * k
                             for k in range(time_sequence_length)]
      image_tokens = tf.stack(
          [all_tokens[i:i + tokens_per_image] for i in image_start_indices],
          axis=0)
      action_start_indices = [i + tokens_per_image for i in image_start_indices]
      action_tokens = [
          tf.stack([
              all_tokens[i:i + tokens_per_action] for i in action_start_indices
          ], 0)
      ]
      image_tokens = tf.one_hot(image_tokens, network._token_embedding_size)
      # Remove extra dimension before the end once b/254902773 is fixed.
      shape = image_tokens.shape
      # Add batch dimension.
      image_tokens = tf.reshape(image_tokens,
                                [1] + shape[:-1] + [1] + shape[-1:])
      return image_tokens, action_tokens

    # Generate some random tokens for image and actions.
    all_tokens = tf.random.uniform(
        shape=[time_sequence_length * (tokens_per_image + tokens_per_action)],
        dtype=tf.int32,
        maxval=10,
        minval=0)
    context_image_tokens, action_tokens = _split_image_and_action_tokens(
        all_tokens)
    # Get the output tokens without any zeroed out input tokens.
    output_tokens = network._transformer_call(
        context_image_tokens=context_image_tokens,
        action_tokens=action_tokens,
        attention_mask=network._default_attention_mask,
        batch_size=1,
        training=False)[0]

    for t in range(time_sequence_length *
                   (tokens_per_image + tokens_per_action)):
      # Zero out future input tokens.
      all_tokens_at_t = tf.concat(
          [all_tokens[:t + 1],
           tf.zeros_like(all_tokens[t + 1:])], 0)
      context_image_tokens, action_tokens = _split_image_and_action_tokens(
          all_tokens_at_t)
      # Get the output tokens with zeroed out input tokens after t.
      output_tokens_at_t = network._transformer_call(
          context_image_tokens=context_image_tokens,
          action_tokens=action_tokens,
          attention_mask=network._default_attention_mask,
          batch_size=1,
          training=False)[0]
      # The output token is unchanged if future input tokens are zeroed out.
      self.assertAllEqual(output_tokens[:t + 1], output_tokens_at_t[:t + 1])

  def testLossMasks(self):
    self._define_specs()
    self._create_agent()
    image_tokens = 3
    action_tokens = 2
    self._agent._actor_network._time_sequence_length = 2
    self._agent._actor_network._tokens_per_context_image = image_tokens
    self._agent._actor_network._tokens_per_action = action_tokens
    self._agent._actor_network._generate_masks()
    self.assertAllEqual(
        self._agent._actor_network._action_tokens_mask,
        tf.constant([
            image_tokens, image_tokens + 1, 2 * image_tokens + action_tokens,
            2 * image_tokens + action_tokens + 1
        ], tf.int32))
    self._agent._actor_network._generate_masks()
    self.assertAllEqual(
        self._agent._actor_network._action_tokens_mask,
        tf.constant([
            image_tokens, image_tokens + 1, 2 * (image_tokens) + action_tokens,
            2 * (image_tokens) + action_tokens + 1
        ], tf.int32))


if __name__ == '__main__':
  # Useful to enable if running with ipdb.
  tf.config.run_functions_eagerly(True)
  tf.test.main()
