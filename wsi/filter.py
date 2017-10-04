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
import skimage.morphology as sk_morphology


def pil_to_np_rgb(pil_img):
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
  print("%-20s | Time: %-14s Max: %6.2f  Min: %6.2f  Mean: %6.2f  Std: %6.2f Type: %-6s Shape: %s" % (
    name, str(elapsed), max, min, mean, std, np_arr.dtype, np_arr.shape))


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

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


def filter_otsu_threshold(np_img, output_type="uint8"):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  t = Time()
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu


def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"):
  """
  Filter image based on entropy (complexity).

  Args:
    np_img: Image as a NumPy array.
    neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
    threshold: Threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
  """
  t = Time()
  entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
  if output_type == "bool":
    pass
  elif output_type == "float":
    entr = entr.astype(float)
  else:
    entr = entr.astype("uint8") * 255
  np_info(entr, "Entropy", t.elapsed())
  return entr


def filter_remove_small_objects(np_img, min_size=3000, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
     NumPy array (bool, float, or uint8).
  """
  t = Time()
  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sm.astype(float)
  else:
    rem_sm = rem_sm.astype("uint8") * 255
  np_info(rem_sm, "Remove Small Objs", t.elapsed())
  return rem_sm


img_path = slide.get_training_thumb_path(4)
img = slide.open_image(img_path)
img.show()
rgb = pil_to_np_rgb(img)
gray = filter_rgb_to_grayscale(rgb)
# np_to_pil(gray).show()
complement = filter_complement(gray)
np_to_pil(complement).show()
hyst = filter_hysteresis_threshold(complement)
np_to_pil(hyst).show()
# entr = filter_entropy(complement)
# np_to_pil(entr).show()
# entr = filter_entropy(complement, neighborhood=6, threshold=4)
# np_to_pil(entr).show()
# rem_small = filter_remove_small_objects(hyst)
# np_to_pil(rem_small).show()
otsu = filter_otsu_threshold(complement)
np_to_pil(otsu).show()
