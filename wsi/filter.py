# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

import numpy as np
import PIL
import wsi.slide as slide
from wsi.slide import Time
import skimage.filters as sk_filters


def pil_to_np(pil_img):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  t = Time()
  rgb = np.asarray(pil_img)
  np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  return PIL.Image.fromarray(np_img)


def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.

  Shape (h, w, c) to (h, w).

  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  t = Time()
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  np_info(grayscale, "Gray", t.elapsed())
  return grayscale


def filter_complement(np_img, output_type="uint8"):
  """
  Obtain the complement of an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).

  Returns:
    Complement image as Numpy array.
  """
  t = Time()
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  np_info(complement, "Complement", t.elapsed())
  return complement


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """
  np_arr = np.asarray(np_arr)
  max = np_arr.max()
  min = np_arr.min()
  mean = np_arr.mean()
  std = np_arr.std()
  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"
  print("%-20s | Time: %-14s Max: %5.2f  Min: %5.2f  Mean: %7.2f  Std: %7.2f Type: %-6s Shape: %s" % (
    name, str(elapsed), max, min, mean, std, np_arr.dtype, np_arr.shape))


def filter_hysteresis_threshold(np_img, low, high, output_type="uint8"):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    low: Low threshold.
    high: High threshold.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
  """
  t = Time()
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")
  np_info(hyst, "Hysteresis Threshold", t.elapsed())
  return hyst


img_path = slide.get_training_thumb_path(4)
pil_img = PIL.Image.open(img_path)
pil_img.show()
rgb_np_img = pil_to_np(pil_img)
gray_np_img = filter_rgb_to_grayscale(rgb_np_img)
np_to_pil(gray_np_img).show()
complement_np_img = filter_complement(gray_np_img)
np_to_pil(complement_np_img).show()
hyst_np_img = filter_hysteresis_threshold(complement_np_img, 50, 100)
np_to_pil(hyst_np_img).show()
