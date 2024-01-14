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

# pytype: skip-file
# pylint: skip-file
"""EfficientNet models modified with added film layers.

Mostly copied from third_party/py/keras/applications/efficientnet.py
"""

import copy
import math
import os
import warnings
import json

from absl import logging
import tensorflow.compat.v2 as tf
from tensorflow.keras import layers

from robotics_transformer.film_efficientnet.film_conditioning_layer import FilmConditioning

BASE_WEIGHTS_PATH = 'efficientnet_checkpoints/efficientnet'
IMAGENET_JSON_PATH = 'efficientnet_checkpoints/imagenet_classes.json'
CLASS_INDEX = None

WEIGHTS_PATHS = {
    'efficientnetb3': BASE_WEIGHTS_PATH + 'b3.h5',
    'efficientnetb3_notop': BASE_WEIGHTS_PATH + 'b3_notop.h5',
}

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)

  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For EfficientNet, input preprocessing is included as part of the model
  (as a `Rescaling` layer), and thus
  `tf.keras.applications.efficientnet.preprocess_input` is actually a
  pass-through function. EfficientNet models expect their inputs to be float
  tensors of pixels with values in the [0-255] range.

  Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded. Defaults to 'imagenet'.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to None.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to 'softmax'.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""

IMAGENET_STDDEV_RGB = [0.229, 0.224, 0.225]


def validate_activation(classifier_activation, weights):
  """validates that the classifier is compatible with the weights.

  Args:
    classifier_activation: str or callable activation function
    weights: The pretrained weights to load.

  Raises:
    ValueError: if an activation other than `None` or `softmax` are used with
      pretrained weights.
  """
  if weights is None:
    return

  classifier_activation = tf.keras.activations.get(classifier_activation)
  if classifier_activation not in {
      tf.keras.activations.get('softmax'),
      tf.keras.activations.get(None)
  }:
    raise ValueError('Only `None` and `softmax` activations are allowed '
                     'for the `classifier_activation` argument when using '
                     'pretrained weights, with `include_top=True`; Received: '
                     f'classifier_activation={classifier_activation}')


def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.

  Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

  Returns:
    A tuple.
  """
  img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
  input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1],
                                                 correct[1]))


def obtain_input_shape(input_shape,
                       default_size,
                       min_size,
                       data_format,
                       require_flatten,
                       weights=None):
  """Internal utility to compute/validate a model's input shape.

  Args:
    input_shape: Either None (will return the default network input shape), or a
      user-provided shape to be validated.
    default_size: Default input width/height for the model.
    min_size: Minimum input width/height accepted by the model.
    data_format: Image data format to use.
    require_flatten: Whether the model is expected to be linked to a classifier
      via a Flatten layer.
    weights: One of `None` (random initialization) or 'imagenet' (pre-training
      on ImageNet). If weights='imagenet' input channels must be equal to 3.

  Returns:
    An integer shape tuple (may include None entries).

  Raises:
    ValueError: In case of invalid argument values.
  """
  if weights != 'imagenet' and input_shape and len(input_shape) == 3:
    if data_format == 'channels_first':
      if input_shape[0] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[0]) + ' input channels.',
            stacklevel=2)
      default_shape = (input_shape[0], default_size, default_size)
    else:
      if input_shape[-1] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[-1]) + ' input channels.',
            stacklevel=2)
      default_shape = (default_size, default_size, input_shape[-1])
  else:
    if data_format == 'channels_first':
      default_shape = (3, default_size, default_size)
    else:
      default_shape = (default_size, default_size, 3)
  if weights == 'imagenet' and require_flatten:
    if input_shape is not None:
      if input_shape != default_shape:
        raise ValueError('When setting `include_top=True` '
                         'and loading `imagenet` weights, '
                         f'`input_shape` should be {default_shape}.  '
                         f'Received: input_shape={input_shape}')
    return default_shape
  if input_shape:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[0] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; Received '
                           f'`input_shape={input_shape}`')
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
            (input_shape[2] is not None and input_shape[2] < min_size)):
          raise ValueError(f'Input size must be at least {min_size}'
                           f'x{min_size}; Received: '
                           f'input_shape={input_shape}')
    else:
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError('`input_shape` must be a tuple of three integers.')
        if input_shape[-1] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; Received '
                           f'`input_shape={input_shape}`')
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
            (input_shape[1] is not None and input_shape[1] < min_size)):
          raise ValueError('Input size must be at least '
                           f'{min_size}x{min_size}; Received: '
                           f'input_shape={input_shape}')
  else:
    if require_flatten:
      input_shape = default_shape
    else:
      if data_format == 'channels_first':
        input_shape = (3, None, None)
      else:
        input_shape = (None, None, 3)
  if require_flatten:
    if None in input_shape:
      raise ValueError('If `include_top` is True, '
                       'you should specify a static `input_shape`. '
                       f'Received: input_shape={input_shape}')
  return input_shape


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation='swish',
                 blocks_args='default',
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax',
                 include_film=False):
  """Instantiates the EfficientNet architecture using given scaling coefficients.

  Args:
    width_coefficient: float, scaling coefficient for network width.
    depth_coefficient: float, scaling coefficient for network depth.
    default_size: integer, default input image size.
    dropout_rate: float, dropout rate before final classifier layer.
    drop_connect_rate: float, dropout rate at skip connections.
    depth_divisor: integer, a unit of network width.
    activation: activation function.
    blocks_args: list of dicts, parameters to construct block modules.
    model_name: string, model name.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    weights: one of `None` (random initialization), 'imagenet' (pre-training on
      ImageNet), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
      as image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is
      False. It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction when `include_top` is
      `False`. - `None` means that the output of the model will be the 4D tensor
      output of the last convolutional layer. - `avg` means that global average
      pooling will be applied to the output of the last convolutional layer, and
      thus the output of the model will be a 2D tensor. - `max` means that
      global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True, and if no `weights` argument is
      specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    include_film: bool, whether or not to insert film conditioning layers.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if blocks_args == 'default':
    blocks_args = DEFAULT_BLOCKS_ARGS

  if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=tf.keras.backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if include_film:
    with tf.compat.v1.variable_scope('context_input'):
      context_input = layers.Input(shape=512)
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not tf.keras.backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

  def round_filters(filters, divisor=depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
      new_filters += divisor
    return int(new_filters)

  def round_repeats(repeats):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))

  # Build stem
  x = img_input
  x = layers.Rescaling(1. / 255.)(x)
  x = layers.Normalization(axis=bn_axis)(x)
  # Note that the normaliztion layer uses square value of STDDEV as the
  # variance for the layer: result = (input - mean) / sqrt(var)
  # However, the original implemenetation uses (input - mean) / var to
  # normalize the input, we need to divide another sqrt(var) to match the
  # original implementation.
  # See https://github.com/tensorflow/tensorflow/issues/49930 for more details
  # We always apply this transformation, even when not using imagenet weights,
  # because it needs to be in the graph when grafting weights from imagenet
  # pretrained models.
  x = layers.Rescaling(1. / tf.math.sqrt(IMAGENET_STDDEV_RGB))(x)

  x = layers.ZeroPadding2D(padding=correct_pad(x, 3), name='stem_conv_pad')(x)
  x = layers.Conv2D(
      round_filters(32),
      3,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='stem_conv')(
          x)
  x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
  x = layers.Activation(activation, name='stem_activation')(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)

  b = 0
  blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
  for (i, args) in enumerate(blocks_args):
    assert args['repeats'] > 0
    # Update block input and output filters based on depth multiplier.
    args['filters_in'] = round_filters(args['filters_in'])
    args['filters_out'] = round_filters(args['filters_out'])

    for j in range(round_repeats(args.pop('repeats'))):
      # The first block needs to take care of stride and filter size increase.
      if j > 0:
        args['strides'] = 1
        args['filters_in'] = args['filters_out']
      x = block(
          x,
          activation,
          drop_connect_rate * b / blocks,
          name='block{}{}_'.format(i + 1, chr(j + 97)),
          **args)
      if include_film:
        with tf.compat.v1.variable_scope('film_conditioning'):
          x = FilmConditioning(num_channels=x.shape[-1])(x, context_input)
      b += 1

  # Build top
  x = layers.Conv2D(
      round_filters(1280),
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='top_conv')(
          x)
  x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
  x = layers.Activation(activation, name='top_activation')(x)
  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    validate_activation(classifier_activation, weights)
    x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        name='predictions')(
            x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = tf.keras.utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
    if include_film:
      inputs = (img_input, context_input)

  # Create model.
  model = tf.keras.Model(inputs, x, name=model_name)

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      key = model_name
    else:
      key = model_name + '_notop'
    weights_path = os.path.join(os.path.dirname(__file__), WEIGHTS_PATHS[key])
    model.load_weights(weights_path, skip_mismatch=False, by_name=False)
  elif weights is not None:
    model.load_weights(weights, skip_mismatch=False, by_name=False)
  return model


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
  """An inverted residual block.

  Args:
      inputs: input tensor.
      activation: activation function.
      drop_rate: float between 0 and 1, fraction of the input units to drop.
      name: string, block label.
      filters_in: integer, the number of input filters.
      filters_out: integer, the number of output filters.
      kernel_size: integer, the dimension of the convolution window.
      strides: integer, the stride of the convolution.
      expand_ratio: integer, scaling coefficient for the input filters.
      se_ratio: float between 0 and 1, fraction to squeeze the input filters.
      id_skip: boolean.

  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

  # Expansion phase
  filters = filters_in * expand_ratio
  if expand_ratio != 1:
    x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'expand_conv')(
            inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
    x = layers.Activation(activation, name=name + 'expand_activation')(x)
  else:
    x = inputs

  # Depthwise Convolution
  if strides == 2:
    x = layers.ZeroPadding2D(
        padding=correct_pad(x, kernel_size), name=name + 'dwconv_pad')(
            x)
    conv_pad = 'valid'
  else:
    conv_pad = 'same'
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=strides,
      padding=conv_pad,
      use_bias=False,
      depthwise_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'dwconv')(
          x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
  x = layers.Activation(activation, name=name + 'activation')(x)

  # Squeeze and Excitation phase
  if 0 < se_ratio <= 1:
    filters_se = max(1, int(filters_in * se_ratio))
    se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
    if bn_axis == 1:
      se_shape = (filters, 1, 1)
    else:
      se_shape = (1, 1, filters)
    se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
    se = layers.Conv2D(
        filters_se,
        1,
        padding='same',
        activation=activation,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_reduce')(
            se)
    se = layers.Conv2D(
        filters,
        1,
        padding='same',
        activation='sigmoid',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_expand')(
            se)
    x = layers.multiply([x, se], name=name + 'se_excite')

  # Output phase
  x = layers.Conv2D(
      filters_out,
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'project_conv')(
          x)
  x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
  if id_skip and strides == 1 and filters_in == filters_out:
    if drop_rate > 0:
      x = layers.Dropout(
          drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(
              x)
    x = layers.add([x, inputs], name=name + 'add')
  return x


def maybe_restore_with_film(
    *args,
    weights='imagenet',
    include_film=False,
    **kwargs,
):
  n1 = EfficientNet(*args, weights=weights, include_film=False, **kwargs)
  if not include_film:
    return n1
  # Copy the model weights over to a new model. This is necessary
  # in case we have inserted early film layers. In this case,
  # the pretrained weights will fail to restore properly
  # unless we do this trick.
  n2 = EfficientNet(*args, weights=None, include_film=True, **kwargs)
  # The layers without the film layers.
  l1 = {l.name: l for l in n1.layers}
  # The layers with the film layers.
  l2 = {l.name: l for l in n2.layers}
  for layer_name, layer in l2.items():
    if layer_name in l1:
      layer.set_weights(l1[layer_name].get_weights())
    # Annoyingly, the rescaling and normalization layers get different names
    # in each graph.
    elif 'rescaling' in layer_name:
      _, num = layer_name.split('_')
      l1_layer_name = 'rescaling_' + str(int(num) - 2 or '')
      l1_layer_name = l1_layer_name.rstrip('_')
      layer.set_weights(l1[l1_layer_name].get_weights())
    elif 'normalization' in layer_name:
      _, num = layer_name.split('_')
      l1_layer_name = 'normalization_' + str(int(num) - 1 or '')
      l1_layer_name = l1_layer_name.rstrip('_')
      layer.set_weights(l1[l1_layer_name].get_weights())
  return n2


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   include_film=False,
                   **kwargs):
  return maybe_restore_with_film(
      1.2,
      1.4,
      300,
      0.3,
      model_name='efficientnetb3',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      include_film=include_film,
      **kwargs)


EfficientNetB3.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB3')


def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  """A placeholder method for backward compatibility.

  The preprocessing logic has been included in the efficientnet model
  implementation. Users are no longer required to call this method to normalize
  the input data. This method does nothing and only kept as a placeholder to
  align the API surface between old and new version of model.

  Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting `tf.keras.image_data_format() is
      used (unless you changed it, it defaults to "channels_last").{mode}

  Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
  """
  return x


def decode_predictions(preds, top=5):
  global CLASS_INDEX
  if CLASS_INDEX is None:
    with open(os.path.join(os.path.dirname(__file__), IMAGENET_JSON_PATH)) as f:
      CLASS_INDEX = json.load(f)
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results
