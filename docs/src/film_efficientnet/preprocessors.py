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
"""Preprocessing functions for transforming the image for training."""

from typing import Optional

import gin
import tensorflow.compat.v2 as tf

CROP_SIZE = 472


@gin.configurable(
    denylist=['images', 'crop_size', 'training', 'convert_dtype', 'seed'])
def convert_dtype_and_crop_images(images,
                                  crop_size: int = CROP_SIZE,
                                  training: bool = True,
                                  pad_then_crop: bool = False,
                                  convert_dtype: bool = True,
                                  seed: Optional[tf.Tensor] = None):
  """Convert uint8 [512, 640, 3] images to float32 and square crop.

  Args:
    images: [B, H, W, 3] uint8 tensor of images.
    crop_size: Width of the square crop.
    training: If we are in training (random crop) or not-training (fixed crop).
    pad_then_crop: If True, pads image and then crops the original image size.
      This allows full field of view to be extracted.
    convert_dtype: whether or not to convert the image to float32 in the range
      of (0, 1).
    seed: Optional seed of shape (2,) for giving to tf.random.stateless_uniform

  Returns:
    [B, crop_size, crop_size, 3] images of dtype float32.
  """

  if seed is None:
    seed = tf.random.uniform(shape=(2,), maxval=2**30, dtype=tf.int32)

  seed2 = tf.random.experimental.stateless_split(seed, num=1)[0]

  if convert_dtype:
    images = tf.image.convert_image_dtype(images, tf.float32)
  image_height = images.get_shape().as_list()[-3]
  image_width = images.get_shape().as_list()[-2]

  if pad_then_crop:

    if training:
      if image_height == 512:
        ud_pad = 40
        lr_pad = 100
      elif image_height == 256:
        ud_pad = 20
        lr_pad = 50
      else:
        raise ValueError(
            'convert_dtype_and_crop_images only supports image height 512 or '
            '256.')
      max_y = 2 * ud_pad
      max_x = 2 * lr_pad
      images = tf.image.pad_to_bounding_box(
          images,
          offset_height=ud_pad,
          offset_width=lr_pad,
          target_height=image_height + 2 * ud_pad,
          target_width=image_width + 2 * lr_pad)
      offset_y = tf.random.stateless_uniform((),
                                             maxval=max_y + 1,
                                             dtype=tf.int32,
                                             seed=seed)
      offset_x = tf.random.stateless_uniform((),
                                             maxval=max_x + 1,
                                             dtype=tf.int32,
                                             seed=seed2)
      images = tf.image.crop_to_bounding_box(images, offset_y, offset_x,
                                             image_height, image_width)
  else:
    # Standard cropping.
    max_y = image_height - crop_size
    max_x = image_width - crop_size

    if training:
      offset_y = tf.random.stateless_uniform((),
                                             maxval=max_y + 1,
                                             dtype=tf.int32,
                                             seed=seed)
      offset_x = tf.random.stateless_uniform((),
                                             maxval=max_x + 1,
                                             dtype=tf.int32,
                                             seed=seed2)
      images = tf.image.crop_to_bounding_box(images, offset_y, offset_x,
                                             crop_size, crop_size)
    else:
      images = tf.image.crop_to_bounding_box(images, max_y // 2, max_x // 2,
                                             crop_size, crop_size)
  return images
