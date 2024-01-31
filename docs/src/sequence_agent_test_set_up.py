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
"""Tests for sequence_agent."""
from typing import Type

import numpy as np
from robotics_transformer import sequence_agent
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class DummyActorNet(network.Network):
  """Used for testing SequenceAgent and its subclass."""

  def __init__(self,
               output_tensor_spec=None,
               train_step_counter=None,
               policy_info_spec=None,
               time_sequence_length=1,
               use_tcl=False,
               **kwargs):
    super().__init__(**kwargs)

  @property
  def tokens_per_action(self):
    return 8

  def set_actions(self, actions):
    self._actions = actions

  def get_actor_loss(self):
    return self._actor_loss

  def call(self,
           observations,
           step_type,
           network_state,
           actions=None,
           training=False):
    del step_type
    image = observations['image']
    tf.expand_dims(tf.reduce_mean(image, axis=-1), -1)
    actions = tensorspec_utils.TensorSpecStruct(
        world_vector=tf.constant(1., shape=[1, 3]),
        rotation_delta=tf.constant(1., shape=[1, 3]),
        terminate_episode=tf.constant(1, shape=[1, 2]),
        gripper_closedness_action=tf.constant(1., shape=[1, 1]),
    )
    return actions, network_state

  @property
  def trainable_weights(self):
    return [tf.Variable(1.0)]


class SequenceAgentTestSetUp(tf.test.TestCase):
  """Defines spec for testing SequenceAgent and its subclass, tests create."""

  def setUp(self):
    super().setUp()
    self._action_spec = tensorspec_utils.TensorSpecStruct()
    self._action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')

    self._action_spec.rotation_delta = tensor_spec.BoundedTensorSpec(
        (3,),
        dtype=tf.float32,
        minimum=-np.pi / 2,
        maximum=np.pi / 2,
        name='rotation_delta')

    self._action_spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
        (1,),
        dtype=tf.float32,
        minimum=-1.,
        maximum=1.,
        name='gripper_closedness_action')
    self._action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')

    state_spec = tensorspec_utils.TensorSpecStruct()
    state_spec.image = tensor_spec.BoundedTensorSpec([256, 320, 3],
                                                     dtype=tf.float32,
                                                     name='image',
                                                     minimum=0.,
                                                     maximum=1.)
    state_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512], dtype=tf.float32, name='natural_language_embedding')
    self._time_step_spec = ts.time_step_spec(observation_spec=state_spec)

    self.sequence_agent_cls = sequence_agent.SequenceAgent

  def create_agent_and_initialize(self,
                                  actor_network: Type[
                                      network.Network] = DummyActorNet,
                                  **kwargs):
    """Creates the agent and initialize it."""
    agent = self.sequence_agent_cls(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        actor_network=actor_network,
        actor_optimizer=tf.keras.optimizers.Adam(),
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
        **kwargs)
    agent.initialize()
    return agent

  def testCreateAgent(self):
    """Creates the Agent and save the agent.policy."""
    agent = self.create_agent_and_initialize()
    self.assertIsNotNone(agent.policy)

    policy_model_saver = policy_saver.PolicySaver(
        agent.policy,
        train_step=tf.compat.v2.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=()),
        input_fn_and_spec=None)
    save_options = tf.saved_model.SaveOptions(
        experimental_io_device='/job:localhost',
        experimental_custom_gradients=False)
    policy_model_saver.save('/tmp/unittest/policy/0', options=save_options)


if __name__ == '__main__':
  tf.test.main()
