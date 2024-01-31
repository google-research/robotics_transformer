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
"""Tensorflow based methods for sequence agents."""
from typing import Optional, Tuple, Union, Any

from absl import logging
import numpy as np

from robotics_transformer import transformer
from robotics_transformer.film_efficientnet import preprocessors
from robotics_transformer.tokenizers import action_tokenizer
from robotics_transformer.tokenizers import image_tokenizer

from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


class TransformerNetwork(network.Network):
  """A transformer based actor network."""

  def __init__(
      self,
      input_tensor_spec: tensorspec_utils.TensorSpecStruct,
      output_tensor_spec: tensorspec_utils.TensorSpecStruct,
      train_step_counter: int = 0,
      vocab_size: int = 256,
      token_embedding_size: int = 512,
      num_layers: int = 1,
      layer_size: int = 4096,
      num_heads: int = 8,
      feed_forward_size: int = 512,
      dropout_rate: float = 0.1,
      time_sequence_length: int = 1,
      crop_size: int = 236,
      policy_info_spec: Optional[dict[Any,
                                      tensor_spec.BoundedTensorSpec]] = None,
      action_order: Optional[list[str]] = None,
      use_token_learner: Optional[bool] = True,
      return_attention_scores: bool = False,
      **kwargs):
    """Creates a transformer network.

    Args:
      input_tensor_spec: Nested list/tuple/dict of TensorSpecs, describing the
        shape of input tensor.
      output_tensor_spec: Nested list/tuple/dict of TensorSpecs, describing the
        shape of output tensor.
      train_step_counter: Counter for number of steps.
      vocab_size: Dimensionality of tokens from the output layer.
      token_embedding_size: Dimensionality of tokens from the embedding layer.
      num_layers: Number of transformer layers.
      layer_size: Size of the multiple head attention layer.
      num_heads: Number of heads for the multiple head attention layer.
      feed_forward_size: Dimensionality of the feed_forward layer.
      dropout_rate: Dropout rate.
      time_sequence_length: Length of the time sequence.
      crop_size: Height and width of the square crop, where original image will
        be padded to allow full field of view to be extracted.
      policy_info_spec: Spec on return value given return type of the return
        tokenizer.
      action_order: Order of actions for the action tokenizer.
      use_token_learner: Whether to use token learner. See
        https://arxiv.org/abs/2106.11297
      return_attention_scores: show attention scores in tensorboard.
      **kwargs: Keyword parameter arguments.
    """
    self._input_tensor_spec = input_tensor_spec
    self._output_tensor_spec = output_tensor_spec
    self._train_step_counter = train_step_counter
    self._actions = None
    self._returns = None
    self._vocab_size = vocab_size
    self._token_embedding_size = token_embedding_size
    self._time_sequence_length = time_sequence_length
    self._crop_size = crop_size

    self._transformer = transformer.Transformer(
        num_layers=num_layers,
        layer_size=layer_size,
        num_heads=num_heads,
        feed_forward_size=feed_forward_size,
        dropout_rate=dropout_rate,
        vocab_size=self._vocab_size,
        return_attention_scores=return_attention_scores)

    # create tokenizers
    self._image_tokenizer = image_tokenizer.RT1ImageTokenizer(
        embedding_output_dim=self._token_embedding_size,
        use_token_learner=use_token_learner)
    self._action_tokenizer = action_tokenizer.RT1ActionTokenizer(
        output_tensor_spec,
        vocab_size=self._vocab_size,
        action_order=action_order)

    self._tokens_per_action = self._action_tokenizer.tokens_per_action
    self._tokens_per_context_image = self._image_tokenizer.tokens_per_context_image
    # generate loss and attention masks
    self._generate_masks()

    # define mappings to token embedding size
    self._action_token_emb = tf.keras.layers.Dense(self._token_embedding_size)

    # define loss function
    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self._attention_scores = []
    self._use_token_learner = use_token_learner

    super(TransformerNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, **kwargs)
    self._state_spec = {
        # Force this to be 4 dimension due to b/254902773.
        # Otherwise can be dimension 3.
        'context_image_tokens':
            tensor_spec.TensorSpec(
                shape=(time_sequence_length, self._tokens_per_context_image, 1,
                       token_embedding_size),
                dtype=tf.float32,
                name='context_image_tokens'),
        'action_tokens':
            tensor_spec.TensorSpec(
                shape=(time_sequence_length, self._tokens_per_action, 1, 1),
                dtype=tf.int32,
                name='action_tokens'),
        # Stores where in the window we are.
        # This value is within range [0, time_sequence_length + 1].
        # When seq_idx == time_sequence_length, context_image_tokens and
        # action_tokens need to be shifted to the left.
        'seq_idx':
            tensor_spec.TensorSpec(
                shape=(1, 1, 1, 1), dtype=tf.int32, name='seq_idx')
    }

  @property
  def attention_scores(self) -> list[tf.Tensor]:
    """Return attention score. This is for debugging/visualization purpose."""
    return self._attention_scores

  def _get_action_index_for_token(self, k):
    """Returns action associated with the token at given position `k`.

    If k is not an action token then it returns -1.
    If k is part of the first action in the sequence then returns 0 etc.

    Args:
        k: an int that represents the position in the sequence.

    Returns:
        The index of the action that this position belongs to, or if this
        position is part of an image token then returns -1.
    """
    if (k < 0 or k >= self._all_num_tokens):
      return -1

    n = k
    if n % self._single_time_step_num_tokens < self._tokens_per_context_image:
      return -1
    return int(n / self._single_time_step_num_tokens)

  def _generate_masks(self):
    """Generate mask for action prediction loss and attention visualization."""
    # each time step = [image, action]
    self._single_time_step_num_tokens = (
        self._tokens_per_action + self._tokens_per_context_image)

    # full sequence = [prefix context + N x timestep + postfix context]
    self._all_num_tokens = (
        self._time_sequence_length * self._single_time_step_num_tokens)

    # create mask for action predition loss
    self._action_tokens_mask = []
    for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
      for x in range(0, self._tokens_per_action, 1):
        self._action_tokens_mask.append(x + n + self._tokens_per_context_image)
    self._action_tokens_mask = tf.constant(
        self._action_tokens_mask, dtype=tf.int32)

    # The look ahead mask ensures causality.
    self._default_attention_mask = tf.linalg.band_part(
        tf.ones((self._all_num_tokens, self._all_num_tokens)), -1, 0)

    action_mask = np.ndarray(
        shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)
    for i in range(self._all_num_tokens):
      for j in range(self._all_num_tokens):
        action_i = self._get_action_index_for_token(i)
        action_j = self._get_action_index_for_token(j)
        mask = 0
        if action_i != -1 and action_j != -1:
          # Ignore actions of previous steps.
          if action_j < action_i:
            mask = 1
          # If we're not auto-regression, ignore action dimensions of current
          # step.
          if (action_j == action_i and j <= i):
            mask = 1
        action_mask[i, j] = mask
    self._default_attention_mask -= action_mask

  def _transformer_call(
      self,
      context_image_tokens: tf.Tensor,
      action_tokens: tf.Tensor,
      batch_size: int,
      training: bool,
      attention_mask: tf.Tensor,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Calls the transformer.

    Args:
      context_image_tokens: Tokenized context and image in Tensor of shape `(B,
        T, num token, -1)`.
      action_tokens: Discrete action token sequence of size [8, 256].
      batch_size: Batch size as when reshaping all tokens.
      training: Whether to run the transformer in training mode.
      attention_mask: Optional bool tensor for masking transformer's attention.

    Returns:
      Output tokens in Tensor of shape `(B, T, dim)`. If
      return_attention_scores, also return the attention scores of
      shape `(B, T, dim)`.
    """
    input_token_sequence = self._assemble_input_token_sequence(
        context_image_tokens, action_tokens, batch_size)

    # run transformer
    output_tokens, self._attention_scores = self._transformer(
        input_token_sequence, training, attention_mask)
    return output_tokens

  def _get_tokens_and_mask(self,
                           observations: dict[str, tf.Tensor],
                           network_state: dict[str, tf.Tensor],
                           training: bool = False):
    # tokenize all inputs
    context_image_tokens, network_state = self._tokenize_images(
        observations, network_state, training)
    action_tokens = self._tokenize_actions(observations, network_state)

    # generate transformer attention mask
    attention_mask = self._default_attention_mask

    return (context_image_tokens, action_tokens, attention_mask)

  def _transformer_call_and_slice(self,
                                  *args,
                                  slice_start: int = 0,
                                  slice_length: int = 1,
                                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    output_tokens = self._transformer_call(*args, **kwargs)

    slice_end = slice_start + slice_length
    token_logits = output_tokens[:, slice_start:slice_end, :]
    token = tf.argmax(token_logits, axis=-1, output_type=tf.int32)

    return token, token_logits

  def call(self,
           observations: dict[str, tf.Tensor],
           network_state: dict[str, tf.Tensor],
           training: bool = False):
    """Calls the transformer network.

    Args:
      observations: Observation data including image and natural language
        embedding in dict of Tensors.
      network_state: Network state data including time step, image, action
        tokens, step number in dict of Tensors.
      training: Whether to call transformer network in training mode.

    Returns:
      A tuple `(Detokenized output actions, network state)`.
    """
    # used to determine training vs inference call
    # outer_rank will be 2 -> [b, t] during training and
    # outer_rank will be 1 -> [b] during inference
    outer_rank = self._get_outer_rank(observations)
    assert outer_rank in (1, 2)

    b, t = self._get_batch_size_and_seq_len(network_state)

    context_image_tokens, action_tokens, attention_mask = self._get_tokens_and_mask(
        observations, network_state, training)

    self._aux_info = {'action_labels': action_tokens}

    if outer_rank == 1:  # This is an inference call
      # run transformer in loop to produce action tokens one-by-one
      seq_idx = tf.reshape(network_state['seq_idx'], [1])[0]
      action_t = tf.minimum(seq_idx, self._time_sequence_length - 1)
      # Transformer shifts all to the left by one step by default (it's usually
      # predicting the next token as default training task...).
      transformer_shift = -1
      # We only want to get the action predicted at time_step.
      start_index = (
          transformer_shift + self._tokens_per_context_image + action_t *
          (self._single_time_step_num_tokens))
      current_action_tokens = []
      action_predictions_logits = []
      for k in range(self._tokens_per_action):
        action_index = start_index + k
        token, token_logits = self._transformer_call_and_slice(
            context_image_tokens,
            action_tokens,
            attention_mask=attention_mask,
            batch_size=b,
            training=training,
            slice_start=action_index  # slicing single action dimension
        )
        action_predictions_logits.append(token_logits)
        current_action_tokens.append(token)
        # action_tokens is [b, t * self._tokens_per_action]
        action_tokens = tf.reshape(action_tokens, [b, -1])
        action_start_index = (action_t * self._tokens_per_action) + k
        action_tokens = tf.concat([
            action_tokens[:, :action_start_index], token,
            action_tokens[:, action_start_index + 1:]
        ],
                                  axis=1)
        # action_tokens is [b, t, self._tokens_per_action]
        action_tokens = tf.reshape(action_tokens,
                                   [b, t, self._tokens_per_action])
      self._aux_info.update({
          # action_predictions_logits is
          # [b, self._tokens_per_action, self._vocab_size]
          'action_predictions_logits': tf.concat(action_predictions_logits, 1)
      })
      # predicted_tokens_for_output is [b, self._tokens_per_action]
      predicted_tokens_for_output = tf.concat(current_action_tokens, 1)
      # state_action_tokens is [b, 1, self._tokens_per_action, 1, 1]
      one_state_action_tokens = predicted_tokens_for_output[:, tf.newaxis, :,
                                                            tf.newaxis,
                                                            tf.newaxis]

      state_action_tokens = network_state['action_tokens']
      network_state['action_tokens'] = tf.concat([
          state_action_tokens[:, :action_t, ...], one_state_action_tokens,
          state_action_tokens[:, action_t + 1:, ...]
      ],
                                                 axis=1)
      # Increment the time_step for the next inference call.
      network_state['seq_idx'] = tf.reshape(
          tf.minimum(seq_idx + 1, self._time_sequence_length), [-1, 1, 1, 1, 1])

      self._loss = tf.constant(0.0)
    else:
      # training call --> simply run one transformer forward pass
      output_tokens = self._transformer_call(
          context_image_tokens,
          action_tokens,
          attention_mask=attention_mask,
          batch_size=b,
          training=training)

      # Gather all predicted actions for the action loss.
      action_logits = tf.gather(
          output_tokens, self._action_tokens_mask - 1, axis=1)
      action_logits_for_training = tf.reshape(
          action_logits, [b, t, self._tokens_per_action, -1])

      # Only take the last action as the action.
      # action_logits_for_output is [b, self._tokens_per_action, emb]
      action_logits_for_output = action_logits_for_training[:, -1]

      # predicted_tokens_for_output is [b, self._tokens_per_action]
      predicted_tokens_for_output = tf.argmax(
          action_logits_for_output, axis=-1, output_type=tf.int32)

      num_items = (
          tf.cast(b * t, tf.float32) * self._single_time_step_num_tokens)
      action_loss = tf.reduce_mean(
          self._loss_object(action_tokens, action_logits_for_training) /
          num_items,
          axis=-1)

      self._loss = action_loss

      # store action labels and predictions for visualization
      self._aux_info.update({
          'action_predictions':
              tf.argmax(
                  action_logits_for_training, axis=-1, output_type=tf.int32),
          'action_loss':
              action_loss,
          'actor_loss_mask':
              tf.ones([b], dtype=tf.float32)
      })

    output_actions = self._action_tokenizer.detokenize(
        predicted_tokens_for_output)
    return output_actions, network_state

  def add_summaries(self, observations: dict[str, tf.Tensor],
                    logging_info: dict[str, tf.Tensor], debug_summaries: bool,
                    training: bool) -> None:
    """Adds summaries.

    Args:
      observations: Observation data including image and natural language
        instruction in dict of Tensors.
      logging_info: Dict with all data stored for logging during training pass.
      debug_summaries: Whether to include debug summaries.
      training: Whether this function is called during training or inference.
    """
    num_params = 0
    for weight in self.trainable_weights:
      weight_params = 1
      for dim in weight.shape:
        weight_params *= dim
      num_params += weight_params
    tf.compat.v2.summary.scalar(name='num_params', data=num_params)
    # debug_summaries are for the non-tpu worker, train_summary.
    if debug_summaries:
      image = observations['image']  # [b, t, h, w, c]
      image_h = image.shape[2]
      image_w = image.shape[3]
      batch_size = image.shape[0]
      num_ts = image.shape[1]
      logging.info('image shape %s', image.shape)
      # Concat images for different timesteps across width.
      image = tf.concat(tf.unstack(image, axis=1), 2)
      # Concat images for different batches (up to 8) across height.
      image = tf.expand_dims(tf.concat(tf.unstack(image, axis=0)[0:8], 0), 0)
      tf.summary.image(
          'observations/image',
          image,
          step=self._train_step_counter,
          # Single output since we have concatenated images along batch.
          max_outputs=1)

      # [b, t], strings
      if 'natural_language_instruction' in observations:
        task = observations['natural_language_instruction'][:, 0]
        tf.summary.text(
            'natural_language_instruction', task, step=self._train_step_counter)
      if self.attention_scores and not self._use_token_learner:
        for l_idx, layer_attention_score in enumerate(self.attention_scores):
          logging.info('Attention score shape: %s, %s', l_idx,
                       layer_attention_score.shape)
          for head_idx in range(layer_attention_score.shape[1]):
            pairwise_attention = tf.expand_dims(
                layer_attention_score[:, head_idx], -1)
            # pairwise attention shape (16, 552, 552, 1)
            # make attention from different time steps comparable
            pairwise_attention = pairwise_attention * np.arange(
                1, pairwise_attention.shape[1] + 1)[None, :, None, None]

            # visualize spatial attention, note this only supports
            # mk1_500tasks_transformer pipeline with no token learner
            img_tf_ts = tf.reshape(
                tf.transpose(
                    tf.reshape(
                        tf.reduce_sum(pairwise_attention, axis=1) / np.arange(
                            pairwise_attention.shape[1], 0, -1)[None, :, None],
                        [batch_size, num_ts, -1]),
                    [0, 2, 1])[:, :-self._tokens_per_action, :],
                [-1, 9, 9, num_ts])

            img_tf_ts = tf.image.resize(
                img_tf_ts, [image_h, image_w],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img_tf_ts_concat = tf.concat(tf.unstack(img_tf_ts, axis=3), 2)
            img_tf_ts_concat_min = tf.reduce_min(
                img_tf_ts_concat, axis=[1, 2], keepdims=True)
            img_tf_ts_concat = (img_tf_ts_concat - img_tf_ts_concat_min) / (
                tf.reduce_max(img_tf_ts_concat, axis=[1, 2], keepdims=True) -
                img_tf_ts_concat_min)
            img_tf_ts_concat = tf.concat(
                tf.unstack(img_tf_ts_concat, axis=0)[:8], 0)
            img_tf_ts_concat = tf.expand_dims(
                tf.expand_dims(img_tf_ts_concat, 0), -1)
            tf.summary.image(
                'attention/layer_{}/head_{}'.format(l_idx, head_idx),
                img_tf_ts_concat,
                step=self._train_step_counter,
                # Single output since we have concatenated images along batch.
                max_outputs=1)

            if img_tf_ts_concat.shape[1] == image.shape[
                1] and img_tf_ts_concat.shape[2] == image.shape[2]:
              # can overlay
              overlay_viz = tf.cast(
                  (tf.cast(image, tf.float32) * (0.2 + img_tf_ts_concat) / 1.2),
                  tf.uint8)
              tf.summary.image(
                  'overlay_attention/layer_{}/head_{}'.format(l_idx, head_idx),
                  overlay_viz,
                  step=self._train_step_counter,
                  # Single output since we have concatenated images along batch.
                  max_outputs=1)

    # log action info
    action_labels = tf.boolean_mask(logging_info['action_labels'],
                                    logging_info['actor_loss_mask'])
    action_predictions = tf.boolean_mask(logging_info['action_predictions'],
                                         logging_info['actor_loss_mask'])
    with tf.name_scope('ActionTokens'):
      token_accuracy = (
          tf.cast(tf.equal(action_labels, action_predictions), tf.float32))
      accuracy = tf.reduce_mean(token_accuracy)
      tf.compat.v2.summary.scalar(
          name='accuracy', data=accuracy, step=self._train_step_counter)
      # Accuracy across timesteps
      for t in range(self._time_sequence_length):
        tf.compat.v2.summary.scalar(
            name='accuracy/time_step/{}'.format(t),
            data=tf.reduce_mean(token_accuracy[:, t, :]),
            step=self._train_step_counter)
      token_index = 0
      for k in self._action_tokenizer.action_order:
        spec = self._action_tokenizer.action_spec[k]
        if spec.dtype == tf.int32:
          n_tokens = 1
        else:
          n_tokens = spec.shape[0]
        action_token_accuracy = tf.reduce_mean(
            token_accuracy[:, :, token_index:token_index + n_tokens])
        tf.compat.v2.summary.scalar(
            name='accuracy/action_type/{}'.format(k),
            data=action_token_accuracy,
            step=self._train_step_counter)
        for n in range(n_tokens):
          tf.summary.histogram(
              'tokens/{}_{}/labels'.format(k, n + 1),
              action_labels[:, :, token_index],
              step=self._train_step_counter)
          tf.summary.histogram(
              'tokens/{}_{}/predictions'.format(k, n + 1),
              action_predictions[:, :, token_index],
              step=self._train_step_counter)
          token_index += 1

    # log loss components
    with tf.name_scope('TokenLosses'):
      tf.compat.v2.summary.scalar(
          name='action_loss',
          data=tf.reduce_mean(logging_info['action_loss']),
          step=self._train_step_counter)

  def _tokenize_images(self, observations, network_state, training):
    image = observations['image']  # [b, t, h, w, c]
    outer_rank = self._get_outer_rank(observations)
    if outer_rank == 1:  # This is an inference call
      seq_idx = tf.reshape(network_state['seq_idx'], [1])[0]
      time_step = tf.minimum(seq_idx, self._time_sequence_length - 1)
      image = tf.expand_dims(image, 1)

    image_shape = tf.shape(image)
    b = image_shape[0]
    input_t = image_shape[1]
    h = image_shape[2]
    w = image_shape[3]
    c = image_shape[4]

    context = self._extract_context_from_observation(observations, input_t)

    image = tf.reshape(image, [b * input_t, h, w, c])
    seed = tf.random.uniform(shape=(2,), maxval=2**30, dtype=tf.int32)
    image = preprocessors.convert_dtype_and_crop_images(
        image,
        crop_size=self._crop_size,
        training=training,
        pad_then_crop=True,
        convert_dtype=True,
        seed=seed)
    image = tf.reshape(image, [b, input_t, h, w, c])
    context_image_tokens = self._image_tokenizer(
        image, context=context, training=training)
    num_tokens = tf.shape(context_image_tokens)[2]
    context_image_tokens = tf.reshape(context_image_tokens,
                                      [b, input_t, num_tokens, 1, -1])
    if outer_rank == 1:  # This is an inference call
      network_state['context_image_tokens'] = tf.reshape(
          network_state['context_image_tokens'], [
              b, self._time_sequence_length, self._tokens_per_context_image, 1,
              -1
          ])
      state_image_tokens = network_state['context_image_tokens']
      # network_state as input for this call is the output from the last call.
      # Therefore, we need to shift all images to the left by 1 in the time axis
      # to align w/ the time dim in this call.
      state_image_tokens = tf.cond(
          seq_idx == self._time_sequence_length,
          lambda: tf.roll(state_image_tokens, -1, axis=1),
          lambda: state_image_tokens)

      context_image_tokens = tf.concat([
          state_image_tokens[:, :time_step, ...], context_image_tokens,
          state_image_tokens[:, time_step + 1:, ...]
      ],
                                       axis=1)
      network_state['context_image_tokens'] = context_image_tokens

    return context_image_tokens, network_state

  def _tokenize_actions(self, observations, network_state):
    outer_rank = self._get_outer_rank(observations)
    if outer_rank == 1:  # This is an inference call
      action_tokens = tf.squeeze(network_state['action_tokens'], [3, 4])
      seq_idx = tf.reshape(network_state['seq_idx'], [1])[0]
      # network_state as input for this call is the output from the last call.
      # Therefore, we need to shift all actions by 1 to the left.
      action_tokens = tf.cond(seq_idx == self._time_sequence_length,
                              lambda: tf.roll(action_tokens, -1, axis=1),
                              lambda: action_tokens)
    else:
      assert outer_rank == 2
      if self._actions is None:
        b, t = self._get_batch_size_and_seq_len(network_state)
        action_tokens = tf.zeros(
            shape=[b, t, self._tokens_per_action], dtype=tf.int32)
      else:
        action_tokens = self._action_tokenizer.tokenize(self._actions)
    return action_tokens

  def _assemble_input_token_sequence(self, context_image_tokens, action_tokens,
                                     batch_size):
    # embed action tokens
    action_tokens = tf.one_hot(action_tokens, self._vocab_size)
    action_tokens = self._action_token_emb(action_tokens)
    action_tokens = tf.zeros_like(action_tokens)  # b/260260205

    # Because of b/254902773, we need to add 1 extra dimension.
    action_tokens = tf.expand_dims(action_tokens, axis=-2)

    # assemble token sequence
    input_token_sequence = tf.concat([context_image_tokens, action_tokens],
                                     axis=2)

    input_token_sequence = tf.reshape(
        input_token_sequence, [batch_size, -1, self._token_embedding_size])
    return input_token_sequence

  def _extract_context_from_observation(self, observations, seq_len):
    """Extract context from observation."""
    context = None
    if 'natural_language_embedding' in observations:
      outer_rank = self._get_outer_rank(observations)
      context = observations['natural_language_embedding']  # [b, t, emb-size]
      if outer_rank == 1:
        context = tf.tile(context[:, None], [1, seq_len, 1])
    return context

  def set_actions(self, actions: tensorspec_utils.TensorSpecStruct):
    """Sets actions that will be tokenized and used in transformer network.

    Args:
      actions: actions to be tokenized and used in transformer network. example
        actions are terminate = [0, 1] world_vector = [0.9, 0.8, -0.3]
        rotation_delta = [-0.1, 0.2, .6] gripper_closedness = 0.9
    """
    self._actions = actions

  def _get_outer_rank(self, observations):
    # used to determine training vs inference call
    # outer_rank will be 2 -> [b, t] during training and
    # outer_rank will be 1 -> [b] during inference
    return nest_utils.get_outer_rank(observations, self._input_tensor_spec)

  def _get_batch_size_and_seq_len(self, network_state):
    image_shape = tf.shape(network_state['context_image_tokens'])
    b = image_shape[0]
    t = image_shape[1]
    return b, t

  def get_actor_loss(self) -> tf.Tensor:
    return self._loss

  def get_aux_info(self) -> dict[str, Any]:
    return self._aux_info
