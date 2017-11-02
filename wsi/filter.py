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

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import PIL
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
import wsi.slide as slide
from wsi.slide import Time
from PIL import ImageDraw, ImageFont

# If True, display NumPy array stats for filters (min, max, mean, is_binary).
DISPLAY_FILTER_STATS = False


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
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
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

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if DISPLAY_FILTER_STATS == False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


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


def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8"):
  """
  Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
  local Otsu threshold.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
  """
  t = Time()
  local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
  local_otsu = (np_img <= local_otsu)
  if output_type == "bool":
    pass
  elif output_type == "float":
    local_otsu = local_otsu.astype(float)
  else:
    local_otsu = local_otsu.astype("uint8") * 255
  np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
  return local_otsu


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


def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
  """
  Filter image based on Canny algorithm edges.

  Args:
    np_img: Image as a NumPy array.
    sigma: Width (std dev) of Gaussian.
    low_threshold: Low hysteresis threshold value.
    high_threshold: High hysteresis threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
  """
  t = Time()
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  np_info(can, "Canny Edges", t.elapsed())
  return can


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


def filter_contrast_stretch(np_img, low=40, high=60):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range percentage low value.
    high: Range percentage high value.

  Returns:
    Image with contrast enhanced.
  """
  t = Time()
  low_p, high_p = np.percentile(np_img, (low, high))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  np_info(contrast_stretch, "Contrast Stretch", t.elapsed())
  return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
  """
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
  """
  t = Time()
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  np_info(hist_equ, "Hist Equalization", t.elapsed())
  return hist_equ


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
  """
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
  t = Time()
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  np_info(adapt_equ, "Adapt Equalization", t.elapsed())
  return adapt_equ


def filter_local_equalization(np_img, disk_size=50):
  """
  Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used for the local histograms

  Returns:
    NumPy array with contrast enhanced using local equalization.
  """
  t = Time()
  local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
  np_info(local_equ, "Local Equalization", t.elapsed())
  return local_equ


def filter_rgb_to_hed(np_img, output_type="uint8"):
  """
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.

  Args:
    np_img: RGB image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array (float or uint8) with HED channels.
  """
  t = Time()
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  np_info(hed, "RGB to HED", t.elapsed())
  return hed


def filter_hed_to_hematoxylin(np_img):
  """
  Obtain Hematoxylin channel from HED NumPy array.

  Args:
    np_img: HED image as a NumPy array.

  Returns:
    NumPy array for Hematoxylin channel.
  """
  t = Time()
  hema = np_img[:, :, 0]
  np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema


def filter_hed_to_eosin(np_img):
  """
  Obtain Eosin channel from HED NumPy array.

  Args:
    np_img: HED image as a NumPy array.

  Returns:
    NumPy array for Eosin channel.
  """
  t = Time()
  eosin = np_img[:, :, 1]
  np_info(eosin, "HED to Eosin", t.elapsed())
  return eosin


def filter_binary_fill_holes(np_img, output_type="bool"):
  """
  Fill holes in a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Binary Fill Holes", t.elapsed())
  return result


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Erode a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for erosion.
    iterations: How many times to repeat the erosion.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Binary Erosion", t.elapsed())
  return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Dilate a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Binary Dilation", t.elapsed())
  return result


def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for opening.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Binary Opening", t.elapsed())
  return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Binary Closing", t.elapsed())
  return result


# NOTE: Potentially could feed in RGB image with mask (such as hysteresis threshold) applied. Then, in a following
# step, could filter out segments that aren't pink or purple enough.
def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
  """
  Use Kmeans segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  np_info(result, "Kmeans Segmentation", t.elapsed())
  return result


def filter_rag_threshold(np_img, compactness=10, n_segments=800):
  """
  Use Kmeans segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, 9)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  np_info(result, "RAG Threshold", t.elapsed())
  return result


def filter_threshold(np_img, threshold, output_type="bool"):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.

  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Threshold", t.elapsed())
  return result


def filter_out_green_channel(rgb, green_thresh=200, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value. If green channel value is greater than green_thresh, mask out pixel.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()
  g = rgb[:, :, 1]
  result = (g < green_thresh) & (g > 0)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter out Green", t.elapsed())
  return result


def filter_out_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
                   display_np_info=False):
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    np_info(result, "Filter out Red", t.elapsed())
  return result


def filter_out_red_pen(rgb, output_type="bool"):
  t = Time()
  result = filter_out_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_out_red(rgb, red_lower_thresh=120, green_upper_thresh=10, blue_upper_thresh=10)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter out Red Pen", t.elapsed())
  return result


def filter_out_bluegreen(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                         display_np_info=False):
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    np_info(result, "Filter out BlueGreen", t.elapsed())
  return result


def filter_out_green_pen(rgb, output_type="bool"):
  t = Time()
  result = filter_out_bluegreen(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_out_bluegreen(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_out_bluegreen(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_out_bluegreen(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_out_bluegreen(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_out_bluegreen(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_out_bluegreen(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter out Green Pen", t.elapsed())
  return result


def filter_out_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                    display_np_info=False):
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    np_info(result, "Filter out Blue", t.elapsed())
  return result


def filter_out_blue_pen(rgb, output_type="bool"):
  t = Time()
  result = filter_out_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_out_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_out_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_out_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_out_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_out_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_out_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter out Blue Pen", t.elapsed())
  return result


def filter_out_grays(rgb, tolerance=15, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter out Grays", t.elapsed())
  return result


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  t = Time()
  result = rgb * np.dstack([mask, mask, mask])
  np_info(result, "Mask RGB", t.elapsed())
  return result


def uint8_to_bool(np_img):
  """
  Convert NumPy array of uint8 (255,0) values to bool (True,False) values

  Args:
    np_img: Binary image as NumPy array of uint8 (255,0) values.

  Returns:
    NumPy array of bool (True,False) values.
  """
  result = (np_img / 255).astype(bool)
  return result


def add_text_and_display(np_img, text, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0)):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.

  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  font = ImageFont.truetype(font_path, size)
  draw.text((0, 0), text, color, font=font)
  result.show()


def apply_filters_to_image(slide_num, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.

  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Dictionary of image information (used for HTML page generation).
  """
  t = Time()
  print("Processing slide #%d" % slide_num)

  html_page_info = dict()

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)
  img_path = slide.get_training_thumb_path(slide_num)
  img = slide.open_image(img_path)

  rgb = pil_to_np_rgb(img)
  k_v = save_display(save, display, rgb, slide_num, 1, "S%03d-F%03d Original" % (slide_num, 1), "rgb")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_not_green = filter_out_green_channel(rgb, green_thresh=200)
  rgb_not_green = mask_rgb(rgb, mask_not_green)
  k_v = save_display(save, display, rgb_not_green, slide_num, 2, "S%03d-F%03d Not Green" % (slide_num, 2),
                     "rgb-not-green")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_not_gray = filter_out_grays(rgb)
  rgb_not_gray = mask_rgb(rgb, mask_not_gray)
  k_v = save_display(save, display, rgb_not_gray, slide_num, 3, "S%03d-F%03d Not Gray" % (slide_num, 3), "rgb-not-gray")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_no_red_pen = filter_out_red_pen(rgb)
  rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)
  k_v = save_display(save, display, rgb_no_red_pen, slide_num, 4, "S%03d-F%03d No Red Pen" % (slide_num, 4),
                     "rgb-no-red-pen")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_no_green_pen = filter_out_green_pen(rgb)
  rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)
  k_v = save_display(save, display, rgb_no_green_pen, slide_num, 5, "S%03d-F%03d No Green Pen" % (slide_num, 5),
                     "rgb-no-green-pen")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_no_blue_pen = filter_out_blue_pen(rgb)
  rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)
  k_v = save_display(save, display, rgb_no_blue_pen, slide_num, 6, "S%03d-F%03d No Blue Pen" % (slide_num, 6),
                     "rgb-no-blue-pen")
  if save: html_page_info[k_v[0]] = k_v[1]

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
  k_v = save_display(save, display, rgb_gray_green_pens, slide_num, 7,
                     "S%03d-F%03d Not Gray, Not Green, No Pens" % (slide_num, 7), "rgb-no-gray-no-green-no-pens")
  if save: html_page_info[k_v[0]] = k_v[1]

  # mask_not_gray_and_not_green = mask_not_gray & mask_not_green
  # rgb_not_gray_and_not_green = mask_rgb(rgb, mask_not_gray_and_not_green)
  # k_v = save_display(save, display, rgb_not_gray_and_not_green, slide_num, 4,
  #                    "S%03d-F%03d Not Gray and Not Green" % (slide_num, 4), "rgb-not-green-and-not-gray")
  # if save: html_page_info[k_v[0]] = k_v[1]

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = mask_rgb(rgb, mask_remove_small)
  k_v = save_display(save, display, rgb_remove_small, slide_num, 7,
                     "S%03d-F%03d Not Gray, Not Green, No Pens,\nRemove Small Objects" % (slide_num, 7),
                     "rgb-not-green-not-gray-no-pens-remove-small")
  if save: html_page_info[k_v[0]] = k_v[1]

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return html_page_info


def save_display(save, display, np_img, slide_num, filter_num, display_text, file_text):
  """
  Optionally save an image and/or display the image.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    np_img: Image as a NumPy array.
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.

  Returns:
    Image information as a key/value tuple.
  """
  if display: add_text_and_display(np_img, display_text)
  if save:
    save_filtered_image(np_img, slide_num, filter_num, file_text)
    key = slide_num * 1000 + filter_num
    value = (slide_num, filter_num, display_text, file_text)
    return (key, value)


def image_cell(slide_num, filter_num, display_text, file_text):
  """
  Generate HTML for viewing a processed image.

  Args:
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.

  Returns:
    HTML for viewing a processed image.
  """
  # return "    <td>\n" + \
  #        "      <a href=\"" + slide.get_filter_thumb_path(slide_num, filter_num, file_text) + "\">\n" + \
  #        "        " + display_text + "<br/>\n" + \
  #        "        " + slide.get_filter_thumb_filename(slide_num, filter_num, file_text) + "<br/>\n" + \
  #        "        <img src=\"" + slide.get_filter_thumb_path(slide_num, filter_num, file_text) + "\" />\n" + \
  #        "      </a>\n" + \
  #        "    </td>\n"
  return "    <td>\n" + \
         "      <a href=\"" + slide.get_filter_thumb_path(slide_num, filter_num, file_text) + "\">\n" + \
         "        " + display_text + "<br/>\n" + \
         "        " + slide.get_filter_thumb_filename(slide_num, filter_num, file_text) + "<br/>\n" + \
         "        <img class=\"lazyload\" src=\"data:image/gif;base64,R0lGODdhAQABAPAAAMPDwwAAACwAAAAAAQABAAACAkQBADs=\" data-src=\"" + slide.get_filter_thumb_path(
    slide_num, filter_num, file_text) + "\" />\n" + \
         "      </a>\n" + \
         "    </td>\n"


def html_header():
  """
  Generate an HTML header for previewing filtered images.

  Returns:
    HTML header for viewing processed images.
  """
  html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
         "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
         "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
         "  <head>\n" + \
         "    <title>Image Processing</title>\n" + \
         "    <style type=\"text/css\">\n" + \
         "     img { max-width: 400px; max-height: 400px; border: 2px solid black; }\n" + \
         "     td { border: 2px solid black; }\n" + \
         "    </style>\n" + \
         "  </head>\n" + \
         "  <body>\n" + \
         "  <script src=\"../js/lazyload.js\"></script>\n" + \
         "  <table>\n"
  return html


def html_footer():
  """
  Generate an HTML footer for previewing filtered images.

  Returns:
    HTML footer for viewing processed images.
  """
  html = "</table>\n" + \
         "<script>lazyload();</script>\n" + \
         "</body>\n" + \
         "</html>\n"
  return html


def save_filtered_image(np_img, slide_num, filter_num, filter_text):
  """
  Save a filtered image to the file system.

  Args:
    np_img: Image as a NumPy array.
    slide_num:  The slide number.
    filter_num: The filter number.
    filter_text: Descriptive text to add to the image filename.
  """
  np_to_pil(np_img).save(slide.get_filter_thumb_path(slide_num, filter_num, filter_text))


def generate_filter_html_page(html_page_info):
  """
  Generate an HTML page to view the filtered images.

  Args:
    html_page_info: Dictionary of image information.
  """
  html = ""
  html += html_header()

  row = 0
  for key in sorted(html_page_info):
    value = html_page_info[key]
    current_row = value[0]
    if current_row > row:
      html += "  <tr>\n"
      row = current_row
    html += image_cell(value[0], value[1], value[2], value[3])
    next_key = key + 1
    if next_key not in html_page_info:
      html += "  </tr>\n"

  html += html_footer()
  text_file = open("filters.html", "w")
  text_file.write(html)
  text_file.close()


def apply_filters_to_image_list(image_num_list, save, display):
  """
  Apply filters to a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    The starting index and the ending index of the slides that were converted to thumbnails.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
    result = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(result)
  return html_page_info


def apply_filters_to_image_range(start_ind, end_ind, save, display):
  """
  Apply filters to a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    The starting index and the ending index of the slides that were converted to thumbnails.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind + 1):
    result = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(result)
  return (start_ind, end_ind, html_page_info)


def singleprocess_apply_filters_to_images(save=True, display=False, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Applying filters to images\n")

  if image_num_list is not None:
    html_page_info = apply_filters_to_image_list(image_num_list, save, display)
  else:
    num_training_slides = slide.get_num_training_slides()
    (s, e, html_page_info) = apply_filters_to_image_range(1, num_training_slides, save, display)

  print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

  if save:
    generate_filter_html_page(html_page_info)


def multiprocess_apply_filters_to_images(save=True, display=False):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).
  Each process will process a range of slide numbers.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
  """
  timer = Time()
  print("Applying filters to images (multiprocess)\n")

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  num_train_images = slide.get_num_training_slides()
  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  # each task specifies a range of slides
  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    tasks.append((start_index, end_index, save, display))
    if start_index == end_index:
      print("Task #" + str(num_process) + ": Process slide " + str(start_index))
    else:
      print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    results.append(pool.apply_async(apply_filters_to_image_range, t))

  html_page_info = dict()
  for result in results:
    (start_ind, end_ind, html_page_info_res) = result.get()
    html_page_info.update(html_page_info_res)
    if (start_ind == end_ind):
      print("Done filtering slide %d" % start_ind)
    else:
      print("Done filtering slides %d through %d" % (start_ind, end_ind))

  if save:
    generate_filter_html_page(html_page_info)

  print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))


# apply_filters_to_image(3, display=True, save=False)
# singleprocess_apply_filters_to_images(save=True, display=False)
# multiprocess_apply_filters_to_images(save=True, display=False)

# red_pen_slides = [4, 15, 24, 48, 63, 67, 115, 117, 122, 130, 135, 165, 166, 185, 209, 237, 245, 249, 279, 281, 282, 289,
#                   336, 349, 357, 380, 450, 482]
# singleprocess_apply_filters_to_images(save=True, display=False, image_num_list=red_pen_slides)
green_pen_slides = [51, 74, 84, 86, 125, 180, 200, 337, 359, 360, 375, 382, 431]
singleprocess_apply_filters_to_images(save=True, display=False, image_num_list=green_pen_slides)
# blue_pen_slides = [7, 28, 74, 107, 130, 140, 157, 174, 200, 221, 241, 318, 340, 355, 394, 410, 414, 457, 499]
# singleprocess_apply_filters_to_images(save=True, display=False, image_num_list=blue_pen_slides)

# img_path = slide.get_training_thumb_path(2)
# img = slide.open_image(img_path)
# rgb = pil_to_np_rgb(img)
# add_text_and_display(rgb, "RGB")
# gray = filter_rgb_to_grayscale(rgb)
# add_text_and_display(gray, "Grayscale")
# complement = filter_complement(gray)
# add_text_and_display(complement, "Complement")
# hyst_mask = filter_hysteresis_threshold(complement, output_type="bool")
# entropy_mask = filter_entropy(complement, output_type="bool")
# hyst_and_entropy_mask = hyst_mask & entropy_mask
# add_text_and_display(mask_rgb(rgb, hyst_mask), "RGB with Hysteresis Threshold Mask")
# add_text_and_display(mask_rgb(rgb, entropy_mask), "RGB with Entropy Mask")
# add_text_and_display(mask_rgb(rgb, hyst_and_entropy_mask), "RGB with Hyst and Entropy Masks")
# rem_small_mask = filter_remove_small_objects(hyst_and_entropy_mask, output_type="bool")
# add_text_and_display(mask_rgb(rgb, rem_small_mask), "RGB with Small Objects Removed")
# hed = filter_rgb_to_hed(rgb)
# hema = filter_hed_to_hematoxylin(hed)
# hema_thresh_mask = filter_threshold(hema, 25)
# add_text_and_display(hema_thresh_mask, "Hema Threshold")
# rem_small_and_hema_thresh_mask = rem_small_mask & hema_thresh_mask
# add_text_and_display(mask_rgb(rgb, rem_small_and_hema_thresh_mask), "RGB with Remove Small and Hema Thresh")
# add_text_and_display(mask_rgb(rgb, ~rem_small_and_hema_thresh_mask), "RGB with Remove Small and Hema Thresh, Inverse")
# mask_not_green = filter_out_green(rgb, green_thresh=200)
# add_text_and_display(mask_rgb(rgb, mask_not_green), "RGB Not Green")
# add_text_and_display(mask_rgb(rgb, ~mask_not_green), "RGB Not Green, Inverse")
# mask_not_gray = filter_out_grays(rgb)
# add_text_and_display(mask_rgb(rgb, mask_not_gray), "RGB Not Gray")
# add_text_and_display(mask_rgb(rgb, ~mask_not_gray), "RGB Not Gray, Inverse")
# mask_not_gray_or_green = filter_out_grays(mask_rgb(rgb, mask_not_green))
# add_text_and_display(mask_rgb(rgb, mask_not_gray_or_green), "RGB Not Gray or Green")
# add_text_and_display(mask_rgb(rgb, ~mask_not_gray_or_green), "RGB Not Gray or Green, Inverse")
# mask_not_gray_and_not_green = mask_not_gray & mask_not_green
# add_text_and_display(mask_rgb(rgb, mask_not_gray_and_not_green), "RGB Not Gray and Not Green")
# add_text_and_display(mask_rgb(rgb, ~mask_not_gray_and_not_green), "RGB Not Gray and Not Green, Inverse")
# hist_eq = filter_histogram_equalization(hema)
# add_text_and_display(hist_eq, "Hist Eq")
# hyst = filter_hysteresis_threshold(complement)
# np_to_pil(hyst).show()
# entr = filter_entropy(complement)
# np_to_pil(entr).show()
# entr = filter_entropy(complement, neighborhood=6, threshold=4)
# np_to_pil(entr).show()
# rem_small = filter_remove_small_objects(hyst)
# np_to_pil(rem_small).show()
# otsu = filter_otsu_threshold(complement)
# np_to_pil(otsu).show()
# can = filter_canny(gray, sigma=7) # interesting WRT pen ink edges
# np_to_pil(can).show()
# plt.imshow(can)
# plt.show()
# local_otsu = filter_local_otsu_threshold(complement)
# np_to_pil(local_otsu).show()
# contrast_stretch = filter_contrast_stretch(gray, low=40, high=60)
# np_to_pil(contrast_stretch).show()
# complement = filter_complement(gray)
# np_to_pil(complement).show()
# hyst = filter_hysteresis_threshold(complement)
# np_to_pil(hyst).show()
# hist_equ = filter_histogram_equalization(rgb)
# np_to_pil(hist_equ).show()
# hist_equ = filter_histogram_equalization(gray, nbins=2)
# np_to_pil(hist_equ).show()
# hist_equ = filter_histogram_equalization(gray, nbins=64)
# np_to_pil(hist_equ).show()
# hist_equ = filter_histogram_equalization(gray, nbins=32)
# np_to_pil(hist_equ).show()
# adapt_equ = filter_adaptive_equalization(gray)
# np_to_pil(adapt_equ).show()
# local_equ = filter_local_equalization(gray)
# np_to_pil(local_equ).show()
# hed = filter_rgb_to_hed(rgb)
# hema = filter_hed_to_hematoxylin(hed)
# np_to_pil(hema).show()
# eosin = filter_hed_to_eosin(hed)
# np_to_pil(eosin).show()
# add_text_and_display(hema, "Hematoxylin")
# np_to_pil(hema).show()
# fill_holes = filter_binary_fill_holes(hyst_mask)
# rgb_fill_holes = mask_rgb(rgb, fill_holes)
# np_to_pil(rgb_fill_holes).show()
# erosion = filter_binary_erosion(hyst_mask)
# np_to_pil(erosion).show()
# erosion = filter_binary_erosion(hyst_mask, iterations=5)
# np_to_pil(erosion).show()
# erosion_mask = uint8_to_bool(erosion)
# rgb_erosion = mask_rgb(rgb, erosion_mask)
# np_to_pil(rgb_erosion).show()
# dilation = filter_binary_dilation(hyst_mask, iterations=3)
# np_to_pil(dilation).show()
# dilation_mask = uint8_to_bool(dilation)
# rgb_dilation = mask_rgb(rgb, dilation_mask)
# np_to_pil(rgb_dilation).show()
# opening = filter_binary_opening(hyst_mask)
# np_to_pil(opening).show()
# opening_mask = uint8_to_bool(opening)
# rgb_opening = mask_rgb(rgb, opening_mask)
# np_to_pil(rgb_opening).show()
# closing = filter_binary_closing(hyst_mask)
# np_to_pil(closing).show()
# closing_mask = uint8_to_bool(closing)
# rgb_closing = mask_rgb(rgb, closing_mask)
# np_to_pil(rgb_closing).show()
# kmeans_seg = filter_kmeans_segmentation(rgb_hyst)
# np_to_pil(kmeans_seg).show()
# rag_thresh = filter_rag_threshold(rgb_hyst)
# np_to_pil(rag_thresh).show()
