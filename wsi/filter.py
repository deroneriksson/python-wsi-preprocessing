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

import multiprocessing
import numpy as np
import os
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from datetime import time

import wsi.slide as slide
from wsi.slide import Time
from PIL import Image, ImageDraw, ImageFont

# If True, display NumPy array stats for filters (min, max, mean, is_binary).
DISPLAY_FILTER_STATS = False
DISPLAY_MASK_PERCENTAGE = True


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
  return Image.fromarray(np_img)


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


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  skip = False
  if (avoid_overmask == True):
    skip_mask_percent_check = mask_percent(np_img)
    if skip_mask_percent_check >= overmask_thresh:
      skip = True

  if not skip:
    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh):
      new_min_size = min_size / 2
      print("Mask percentage %3.2f%% >= threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
        mask_percentage, overmask_thresh, min_size, new_min_size))
      rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  if skip:
    print("Mask percentage %3.2f%% >= threshold %3.2f%%, so Remove Small Objs skipped." % (
      skip_mask_percent_check, overmask_thresh))
  else:
    np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def filter_contrast_stretch(np_img, low=40, high=60):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range low value (0 to 255).
    high: Range high value (0 to 255).

  Returns:
    Image with contrast enhanced.
  """
  t = Time()
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
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


def filter_hed_to_hematoxylin(np_img, output_type="uint8"):
  """
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Hematoxylin channel.
  """
  t = Time()
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema


def filter_hed_to_eosin(np_img, output_type="uint8"):
  """
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Eosin channel.
  """
  t = Time()
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
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


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()

  skip = False
  if (avoid_overmask == True):
    skip_mask_percent_check = mask_percent(np_img)
    if skip_mask_percent_check >= overmask_thresh:
      skip = True

  if not skip:
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh):
      new_green_thresh = (255 - green_thresh) / 2 + green_thresh
      print(
        "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
          mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
      gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  if skip:
    print("Mask percentage %3.2f%% >= threshold %3.2f%%, so Filter Green Channel skipped." % (
      skip_mask_percent_check, overmask_thresh))
  else:
    np_info(np_img, "Filter Green Channel", t.elapsed())
  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info an filter time.

  Returns:
    NumPy array representing the mask.
  """
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
    np_info(result, "Filter Red", t.elapsed())
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter Red Pen", t.elapsed())
  return result


def filter_bluegreen(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                     display_np_info=False):
  """
  Create a mask to filter out blueish/greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info an filter time.

  Returns:
    NumPy array representing the mask.
  """
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
    np_info(result, "Filter BlueGreen", t.elapsed())
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green (actually blue-green) pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_bluegreen(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_bluegreen(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_bluegreen(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_bluegreen(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_bluegreen(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_bluegreen(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_bluegreen(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_bluegreen(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_bluegreen(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_bluegreen(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_bluegreen(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_bluegreen(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_bluegreen(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_bluegreen(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_bluegreen(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter Green Pen", t.elapsed())
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info an filter time.

  Returns:
    NumPy array representing the mask.
  """
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
    np_info(result, "Filter Blue", t.elapsed())
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  np_info(result, "Filter Blue Pen", t.elapsed())
  return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
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
  np_info(result, "Filter Grays", t.elapsed())
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


def apply_filters_to_image(slide_num, save=True, display=False, return_image=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.

  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Dictionary of image information (used for HTML page generation), or if return_image is True, return the resulting
    filtered image as a NumPy array.
  """
  t = Time()
  print("Processing slide #%d" % slide_num)

  info = dict()

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)
  img_path = slide.get_training_image_path(slide_num)
  img = slide.open_image(img_path)

  rgb = pil_to_np_rgb(img)
  save_display(save, display, info, rgb, slide_num, 1, "Original", "rgb")

  mask_not_green = filter_green_channel(rgb, green_thresh=200)
  rgb_not_green = mask_rgb(rgb, mask_not_green)
  save_display(save, display, info, rgb_not_green, slide_num, 2, "Not Green", "rgb-not-green")

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = mask_rgb(rgb, mask_not_gray)
  save_display(save, display, info, rgb_not_gray, slide_num, 3, "Not Gray", "rgb-not-gray")

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)
  save_display(save, display, info, rgb_no_red_pen, slide_num, 4, "No Red Pen", "rgb-no-red-pen")

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)
  save_display(save, display, info, rgb_no_green_pen, slide_num, 5, "No Green Pen", "rgb-no-green-pen")

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)
  save_display(save, display, info, rgb_no_blue_pen, slide_num, 6, "No Blue Pen", "rgb-no-blue-pen")

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
  save_display(save, display, info, rgb_gray_green_pens, slide_num, 7, "Not Gray, Not Green, No Pens",
               "rgb-no-gray-no-green-no-pens")

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = mask_rgb(rgb, mask_remove_small)
  save_display(save, display, info, rgb_remove_small, slide_num, 8,
               "Not Gray, Not Green, No Pens,\nRemove Small Objects",
               "rgb-not-green-not-gray-no-pens-remove-small")

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  if return_image:
    img_result = rgb_remove_small
    return img_result
  else:
    return info


def save_display(save, display, info, np_img, slide_num, filter_num, display_text, file_text):
  """
  Optionally save an image and/or display the image.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    info: Dictionary to store filter information.
    np_img: Image as a NumPy array.
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.
  """
  mask_percentage = None
  if DISPLAY_MASK_PERCENTAGE:
    mask_percentage = mask_percent(np_img)
    display_text = display_text + mask_percentage_text(mask_percentage)
  if slide_num is None and filter_num is None:
    pass
  elif filter_num is None:
    display_text = "S%03d " % slide_num + display_text
  else:
    display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
  if display: add_text_and_display(np_img, display_text)
  if save: save_filtered_image(np_img, slide_num, filter_num, file_text)
  info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)


def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.

  Args:
    mask_percentage: The mask percentage.

  Returns:
    The mask percentage formatted as a string.
  """
  return "\n(%3.2f%% masked)" % mask_percentage


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
  return "    <td>\n" + \
         "      <a href=\"" + slide.get_filter_image_path(slide_num, filter_num, file_text) + "\">\n" + \
         "        " + display_text + "<br/>\n" + \
         "        " + slide.get_filter_image_filename(slide_num, filter_num, file_text) + "<br/>\n" + \
         "        <img class=\"lazyload\" src=\"data:image/gif;base64,R0lGODdhAQABAPAAAMPDwwAAACwAAAAAAQABAAACAkQBADs=\" data-src=\"" + slide.get_filter_image_path(
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
  t = Time()
  filepath = slide.get_filter_image_path(slide_num, filter_num, filter_text)
  np_to_pil(np_img).save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))


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
    The starting index and the ending index of the slides that were converted to images.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
    result = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(result)
  return (image_num_list, html_page_info)


def apply_filters_to_image_range(start_ind, end_ind, save, display):
  """
  Apply filters to a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    The starting index and the ending index of the slides that were converted to images.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind + 1):
    result = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(result)
  return (start_ind, end_ind, html_page_info)


def singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    html: If True, generate HTML page to display filtered images.
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

  if html:
    generate_filter_html_page(html_page_info)


def multiprocess_apply_filters_to_images(save=False, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Applying filters to images (multiprocess)\n")

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  if image_num_list is not None:
    num_train_images = len(image_num_list)
  else:
    num_train_images = slide.get_num_training_slides()
  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    if image_num_list is not None:
      sublist = image_num_list[start_index - 1:end_index]
      tasks.append((sublist, save, display))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, save, display))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(apply_filters_to_image_list, t))
    else:
      results.append(pool.apply_async(apply_filters_to_image_range, t))

  html_page_info = dict()
  for result in results:
    if image_num_list is not None:
      (image_nums, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      print("Done filtering slides: %s" % image_nums)
    else:
      (start_ind, end_ind, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      if (start_ind == end_ind):
        print("Done filtering slide %d" % start_ind)
      else:
        print("Done filtering slides %d through %d" % (start_ind, end_ind))

  if html:
    generate_filter_html_page(html_page_info)

  print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))


# apply_filters_to_image(4, display=False, save=True)
# singleprocess_apply_filters_to_images(save=True, display=False)
# multiprocess_apply_filters_to_images(save=True, display=False, html=True)

# red_pen_slides = [4, 15, 24, 48, 63, 67, 115, 117, 122, 130, 135, 165, 166, 185, 209, 237, 245, 249, 279, 281, 282, 289,
#                   336, 349, 357, 380, 450, 482]
# red_pen_slides = [1, 2, 3]
# multiprocess_apply_filters_to_images(save=False, display=False, image_num_list=red_pen_slides)
# green_pen_slides = [51, 74, 84, 86, 125, 180, 200, 337, 359, 360, 375, 382, 431]
# green_pen_slides = [74]
# multiprocess_apply_filters_to_images(save=True, display=False, image_num_list=green_pen_slides)
# blue_pen_slides = [7, 28, 74, 107, 130, 140, 157, 174, 200, 221, 241, 318, 340, 355, 394, 410, 414, 457, 499]
# singleprocess_apply_filters_to_images(save=True, display=False, image_num_list=blue_pen_slides)
# overmasked_slides = [1, 21, 29, 37, 43, 88, 116, 126, 127, 142, 145, 173, 196, 220, 225, 234, 238, 284, 292, 294, 304,
#                      316, 401, 403, 424, 448, 452, 472, 494]
# overmasked_slides = [1, 2, 3, 4, 5, 21, 37, 294, 401, 424, 472]
# overmasked_slides = [21]
# multiprocess_apply_filters_to_images(save=True, display=False, image_num_list=overmasked_slides)

img_path = slide.get_training_image_path(4)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
not_red = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90, display_np_info=True)
add_text_and_display(not_red, "Red Filter (150, 80, 90)")
add_text_and_display(mask_rgb(rgb, not_red), "Not Red")
add_text_and_display(mask_rgb(rgb, ~not_red), "Red")


# add_text_and_display(grayscale, "Grayscale")
# complement = filter_complement(grayscale)
# contrast_stretch = filter_contrast_stretch(complement, low=100, high=200)
# add_text_and_display(contrast_stretch, "Contrast Stretch")
# r_hist = filter_histogram_equalization(rgb[:, :, 0])
# g_hist = filter_histogram_equalization(rgb[:, :, 1])
# b_hist = filter_histogram_equalization(rgb[:, :, 2])
# add_text_and_display(r_hist, "Histogram Equalization R")
# add_text_and_display(g_hist, "Histogram Equalization G")
# add_text_and_display(b_hist, "Histogram Equalization B")
# hist_equ = np.dstack((r_hist, g_hist, b_hist))
# np_info(hist_equ, "Histogram Equalization")
# add_text_and_display(hist_equ, "Histogram Equalization Separate Channels")
# np_info(hist_equ, "Histogram Equalization")
# rgb_hist = filter_histogram_equalization(rgb)
# add_text_and_display(rgb_hist, "RGB Histogram Equalization")

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
# mask_not_green = filter_green(rgb, green_thresh=200)
# add_text_and_display(mask_rgb(rgb, mask_not_green), "RGB Not Green")
# add_text_and_display(mask_rgb(rgb, ~mask_not_green), "RGB Not Green, Inverse")
# mask_not_gray = filter_grays(rgb)
# add_text_and_display(mask_rgb(rgb, mask_not_gray), "RGB Not Gray")
# add_text_and_display(mask_rgb(rgb, ~mask_not_gray), "RGB Not Gray, Inverse")
# mask_not_gray_or_green = filter_grays(mask_rgb(rgb, mask_not_green))
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

# np_img = apply_filters_to_image(15, display=False, save=False, return_image=True)
# add_text_and_display(np_img, "Filtered" + mask_percentage_text(mask_percent(np_img)))
# np_info(np_img)
# np_filt = filter_rgb_to_grayscale(np_img)
# np_filt = filter_entropy(np_filt, neighborhood=5, threshold=4, output_type="bool")
# np_img = mask_rgb(np_img, np_filt)
# add_text_and_display(np_img, "Entropy" + mask_percentage_text(mask_percent(np_img)))
