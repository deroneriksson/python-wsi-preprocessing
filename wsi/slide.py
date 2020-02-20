# ------------------------------------------------------------------------
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
#
# ------------------------------------------------------------------------

import pathlib 
from pathlib import Path
import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image
import re
import sys
from wsi import util, tiles
from wsi.util import Time
from typing import List, Callable, Union


def open_slide(path:Union[str, pathlib.Path]):
  """
  Open a whole-slide image (*.svs,*.ndpi, etc).

  Args:
    path: Path to the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  #try:
  slide = openslide.open_slide(str(path))
  #except OpenSlideError:
  #  slide = None
  #except FileNotFoundError:
  #  slide = None
  return slide


def open_image(filename):
  """
  Open an image (*.jpg, *.png, etc).

  Args:
    filename: Name of the image file.

  returns:
    A PIL.Image.Image object representing an image.
  """
  image = Image.open(filename)
  return image


def open_image_np(filename):
  """
  Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

  Args:
    filename: Name of the image file.

  returns:
    A NumPy representing an RGB image.
  """
  pil_img = open_image(filename)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def small_to_large_mapping(small_pixel, large_dimensions, scale_factor):
  """
  Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

  Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.

  Returns:
    Tuple consisting of the scaled-up width and height.
  """
  small_x, small_y = small_pixel
  large_w, large_h = large_dimensions
  large_x = round((large_w / scale_factor) / math.floor(large_w / scale_factor) * (scale_factor * small_x))
  large_y = round((large_h / scale_factor) / math.floor(large_h / scale_factor) * (scale_factor * small_y))
  return large_x, large_y