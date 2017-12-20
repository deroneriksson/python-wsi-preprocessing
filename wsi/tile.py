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
import wsi.filter as filter
import wsi.slide as slide
import math
from wsi.slide import Time
from PIL import Image, ImageDraw, ImageFont

ROW_TILE_SIZE = 128
COL_TILE_SIZE = 128
TISSUE_THRESHOLD_PERCENT = 50
TISSUE_LOW_THRESHOLD_PERCENT = 5


def get_num_tiles(np_img, row_tile_size, col_tile_size):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    np_img: Image as a NumPy array.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  rows, cols, _ = np_img.shape
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(np_img, row_tile_size, col_tile_size):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column).

  Args:
    np_img: Image as a NumPy array.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column.
  """
  indices = list()
  rows, cols, _ = np_img.shape
  num_row_tiles, num_col_tiles = get_num_tiles(np_img, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c))
  return indices


def tile_summary(slide_num, np_img, tile_indices, row_tile_size, col_tile_size, display=True, save=False,
                 thresh_color=(0, 255, 0), below_thresh_color=(255, 255, 0), below_lower_thresh_color=(255, 165, 0),
                 no_tissue_color=(255, 0, 0), text_color=(255, 255, 255), text_size=22,
                 font_path="/Library/Fonts/Arial Bold.ttf"):
  num_row_tiles, num_col_tiles = get_num_tiles(np_img, row_tile_size, col_tile_size)
  summary_img = np.zeros([ROW_TILE_SIZE * num_row_tiles, COL_TILE_SIZE * num_col_tiles, np_img.shape[2]],
                         dtype=np.uint8)
  # add gray edges so that summary text does not get cut off
  summary_img.fill(120)
  summary_img[0:np_img.shape[0], 0:np_img.shape[1]] = np_img
  summary = filter.np_to_pil(summary_img)
  draw = ImageDraw.Draw(summary)
  count = 0
  for t in tile_indices:
    count += 1
    r_s, r_e, c_s, c_e = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    tissue_percentage = filter.tissue_percent(np_tile)
    print("TILE [%d:%d, %d:%d]: Tissue %f%%" % (r_s, r_e, c_s, c_e, tissue_percentage))
    if (tissue_percentage >= TISSUE_THRESHOLD_PERCENT):
      draw.rectangle([(c_s, r_s), (c_e - 1, r_e - 1)], outline=thresh_color)
      draw.rectangle([(c_s + 1, r_s + 1), (c_e - 2, r_e - 2)], outline=thresh_color)
    elif (tissue_percentage >= TISSUE_LOW_THRESHOLD_PERCENT) and (tissue_percentage < TISSUE_THRESHOLD_PERCENT):
      draw.rectangle([(c_s, r_s), (c_e - 1, r_e - 1)], outline=below_thresh_color)
      draw.rectangle([(c_s + 1, r_s + 1), (c_e - 2, r_e - 2)], outline=below_thresh_color)
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESHOLD_PERCENT):
      draw.rectangle([(c_s, r_s), (c_e - 1, r_e - 1)], outline=below_lower_thresh_color)
      draw.rectangle([(c_s + 1, r_s + 1), (c_e - 2, r_e - 2)], outline=below_lower_thresh_color)
    else:
      draw.rectangle([(c_s, r_s), (c_e - 1, r_e - 1)], outline=no_tissue_color)
      draw.rectangle([(c_s + 1, r_s + 1), (c_e - 2, r_e - 2)], outline=no_tissue_color)
    # filter.display_img(np_tile, text=label, size=14, bg=True)
    label = "#%d\n%4.2f%%" % (count, tissue_percentage)
    font = ImageFont.truetype(font_path, size=text_size)
    draw.text((c_s + 2, r_s + 2), label, text_color, font=font)
  if display:
    summary.show()
  if save:
    save_tile_summary_image(summary, slide_num)


def save_tile_summary_image(pil_img, slide_num):
  """
  Save a tile summary image to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Summary Image", str(t.elapsed()), filepath))


def summary(slide_num, save=False, display=True):
  """
  Display and/or save a summary image of tiles.

  Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save: If True, save tile summary image.
  """
  img_path = slide.get_filter_image_result(slide_num)
  img = slide.open_image(img_path)
  np_img = filter.pil_to_np_rgb(img)

  tile_indices = get_tile_indices(np_img, ROW_TILE_SIZE, COL_TILE_SIZE)
  tile_summary(slide_num, np_img, tile_indices, ROW_TILE_SIZE, COL_TILE_SIZE, display=display, save=save)


def image_list_to_tile_summaries(image_num_list, save=True, display=False):
  """
  Generate tile summaries for a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save tile summary images.
    display: If True, display tile summary images to screen.
  """
  for slide_num in image_num_list:
    summary(slide_num, save, display)


def image_range_to_tile_summaries(start_ind, end_ind, save=True, display=False):
  """
  Generate tile summaries for a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save tile summary images.
    display: If True, display tile summary images to screen.
  """
  for slide_num in range(start_ind, end_ind + 1):
    summary(slide_num, save, display)


def singleprocess_images_to_tile_summaries(save=True, display=False, image_num_list=None):
  """
  Generate tile summaries to training images and optionally save/and or display the tile summaries.

  Args:
    save: If True, save images.
    display: If True, display images to screen.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Generating tile summaries\n")

  if image_num_list is not None:
    image_list_to_tile_summaries(image_num_list, save, display)
  else:
    num_training_slides = slide.get_num_training_slides()
    image_range_to_tile_summaries(1, num_training_slides, save, display)

  print("Time to generate tile summaries for all images: %s\n" % str(t.elapsed()))


# summary(25, save=True)
# summary(26, save=True)
# image_list_to_tile_summaries([1, 2, 3, 4, 5], display=True)
# image_range_to_tile_summaries(1, 50)
# singleprocess_images_to_tile_summaries(image_num_list=[54, 55, 56], display=True)
singleprocess_images_to_tile_summaries()