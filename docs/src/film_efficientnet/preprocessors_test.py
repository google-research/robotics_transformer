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
"""Tests for preprocessors."""
from absl.testing import parameterized
import numpy as np
from robotics_transformer.film_efficientnet import preprocessors
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v2 as tf


def _random_image(shape):
  images = tf.random.uniform(
      shape, minval=0, maxval=255, dtype=tf.dtypes.int32, seed=42)
  return tf.cast(images, tf.uint8)


def _get_features(
    image_shape=(2, 512, 640, 3), use_task_image=False, use_goal_image=False):
  # Time-dimension stacking occurs during training but not eval.
  state = tensorspec_utils.TensorSpecStruct(image=_random_image(image_shape))
  if use_task_image:
    state.task_image = _random_image(image_shape)
  if use_goal_image:
    state.goal_image = _random_image(image_shape)
  return state


class PreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((True, False, False), (False, True, False),
                            (True, False, True), (False, True, True))
  def testConvertDtypeAndCropImages(self, training, pad_then_crop,
                                    convert_dtype):
    features = _get_features()
    images = preprocessors.convert_dtype_and_crop_images(
        features.image,
        training=training,
        pad_then_crop=pad_then_crop,
        convert_dtype=convert_dtype)
    expected_cropped_shape = ([2, 512, 640, 3]
                              if pad_then_crop else [2, 472, 472, 3])
    tf.ensure_shape(images, expected_cropped_shape)
    if convert_dtype:
      self.assertEqual(images.dtype, tf.float32)
      self.assertLessEqual(images.numpy().max(), 1.)
      self.assertGreaterEqual(images.numpy().min(), 0.)
    else:
      self.assertEqual(images.dtype, tf.uint8)
      self.assertLessEqual(images.numpy().max(), 255)
      self.assertGreaterEqual(images.numpy().min(), 0)
      self.assertGreater(images.numpy().max(), 1)

  def testConvertDtypeAndCropImagesSeeded(self):
    features = _get_features()
    seed = tf.constant([1, 2], tf.int32)
    images1 = preprocessors.convert_dtype_and_crop_images(
        features.image, training=True, pad_then_crop=True, seed=seed)
    images2 = preprocessors.convert_dtype_and_crop_images(
        features.image, training=True, pad_then_crop=True, seed=seed)
    diff = np.sum(np.abs(images1.numpy() - images2.numpy()))
    self.assertAlmostEqual(diff, 0)

  def testConvertDtypeAndCropImagesUnseeded(self):
    features = _get_features()
    seed1 = tf.constant([1, 2], tf.int32)
    images1 = preprocessors.convert_dtype_and_crop_images(
        features.image, training=True, pad_then_crop=True, seed=seed1)
    seed2 = tf.constant([2, 3], tf.int32)
    images2 = preprocessors.convert_dtype_and_crop_images(
        features.image, training=True, pad_then_crop=True, seed=seed2)
    diff = np.sum(np.abs(images1.numpy() - images2.numpy()))
    self.assertNotAlmostEqual(diff, 0)
