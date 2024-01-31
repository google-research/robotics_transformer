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
"""Sequence policy and agent that directly output actions via actor network.

These classes are not intended to change as they are generic enough for any
all-neural actor based agent+policy. All new features are intended to be
implemented in `actor_network` and `loss_fn`.
"""
from typing import Optional, Type

from absl import logging
import tensorflow as tf
from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class SequencePolicy(actor_policy.ActorPolicy):
  """A policy that directly outputs actions via an actor network."""

  def __init__(self, **kwargs):
    self._actions = None
    super().__init__(**kwargs)

  def set_actions(self, actions):
    self._actor_network.set_actions(actions)

  def get_actor_loss(self):
    return self._actor_network.get_actor_loss()

  def get_aux_info(self):
    return self._actor_network.get_aux_info()

  def set_training(self, training):
    self._training = training

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    del seed
    action, policy_state = self._apply_actor_network(
        time_step.observation,
        step_type=time_step.step_type,
        policy_state=policy_state)
    info = ()
    return policy_step.PolicyStep(action, policy_state, info)

  def _distribution(self, time_step, policy_state):
    current_step = super()._distribution(time_step, policy_state)
    return current_step


class SequenceAgent(tf_agent.TFAgent):
  """A sequence agent that directly outputs actions via an actor network."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               actor_network: Type[network.Network],
               actor_optimizer: tf.keras.optimizers.Optimizer,
               policy_cls: Type[actor_policy.ActorPolicy] = SequencePolicy,
               time_sequence_length: int = 6,
               debug_summaries: bool = False,
               **kwargs):
    self._info_spec = ()
    self._actor_network = actor_network(  # pytype: disable=missing-parameter  # dynamic-method-lookup
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        policy_info_spec=self._info_spec,
        train_step_counter=kwargs['train_step_counter'],
        time_sequence_length=time_sequence_length)

    self._actor_optimizer = actor_optimizer
    # Train policy is only used for loss and never exported as saved_model.
    self._train_policy = policy_cls(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=self._info_spec,
        actor_network=self._actor_network,
        training=True)
    collect_policy = policy_cls(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=self._info_spec,
        actor_network=self._actor_network,
        training=False)
    super(SequenceAgent, self).__init__(
        time_step_spec,
        action_spec,
        collect_policy,  # We use the collect_policy as the eval policy.
        collect_policy,
        train_sequence_length=time_sequence_length,
        **kwargs)
    self._data_context = data_converter.DataContext(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=collect_policy.info_spec,
        use_half_transition=True)
    self.as_transition = data_converter.AsHalfTransition(
        self._data_context, squeeze_time_dim=False)
    self._debug_summaries = debug_summaries

    num_params = 0
    for weight in self._actor_network.trainable_weights:
      weight_params = 1
      for dim in weight.shape:
        weight_params *= dim
      logging.info('%s has %s params.', weight.name, weight_params)
      num_params += weight_params
    logging.info('Actor network has %sM params.', round(num_params / 1000000.,
                                                        2))

  def _train(self, experience: types.NestedTensor,
             weights: types.Tensor) -> tf_agent.LossInfo:
    self.train_step_counter.assign_add(1)
    loss_info = self._loss(experience, weights, training=True)
    self._apply_gradients(loss_info.loss)
    return loss_info

  def _apply_gradients(self, loss: types.Tensor):
    variables = self._actor_network.trainable_weights
    gradients = tf.gradients(loss, variables)
    # Skip nan and inf gradients.
    new_gradients = []
    for g in gradients:
      if g is not None:
        new_g = tf.where(
            tf.math.logical_or(tf.math.is_inf(g), tf.math.is_nan(g)),
            tf.zeros_like(g), g)
        new_gradients.append(new_g)
      else:
        new_gradients.append(g)
    grads_and_vars = list(zip(new_gradients, variables))
    self._actor_optimizer.apply_gradients(grads_and_vars)

  def _loss(self, experience: types.NestedTensor, weights: types.Tensor,
            training: bool) -> tf_agent.LossInfo:
    transition = self.as_transition(experience)
    time_steps, policy_steps, _ = transition
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy = self._train_policy
    policy.set_actions(policy_steps.action)
    policy.set_training(training=training)
    with tf.name_scope('actor_loss'):
      policy_state = policy.get_initial_state(batch_size)
      policy.action(time_steps, policy_state=policy_state)
      valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
      loss = valid_mask * policy.get_actor_loss()
      loss = tf.reduce_mean(loss)
      policy.set_actions(None)
      self._actor_network.add_summaries(time_steps.observation,
                                        policy.get_aux_info(),
                                        self._debug_summaries, training)
      return tf_agent.LossInfo(loss=loss, extra=loss)
