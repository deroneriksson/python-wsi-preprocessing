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

import math
import multiprocessing
import numpy as np
import os
import wsi.filter as filter
import wsi.slide as slide
from wsi.slide import Time
from PIL import Image, ImageDraw, ImageFont

TISSUE_THRESHOLD_PERCENT = 50
TISSUE_LOW_THRESHOLD_PERCENT = 5

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024

ROW_TILE_SIZE_BASED_ON_SUMMARY_IMAGE_SIZE = 128
COL_TILE_SIZE_BASED_ON_SUMMARY_IMAGE_SIZE = 128

DISPLAY_TILE_LABELS = False


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
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
  rows, cols, _ = np_img.shape
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary_img = np.zeros([row_tile_size * num_row_tiles, col_tile_size * num_col_tiles, np_img.shape[2]],
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
    if DISPLAY_TILE_LABELS == True:
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

  rows, cols, _ = np_img.shape

  if slide.RESIZE_ALL_BY_SCALE_FACTOR == True:
    row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
    col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  else:
    row_tile_size = ROW_TILE_SIZE_BASED_ON_SUMMARY_IMAGE_SIZE
    col_tile_size = COL_TILE_SIZE_BASED_ON_SUMMARY_IMAGE_SIZE

  tile_indices = get_tile_indices(rows, cols, row_tile_size, col_tile_size)
  tile_summary(slide_num, np_img, tile_indices, row_tile_size, col_tile_size, display=display, save=save)


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
  return image_num_list


def image_range_to_tile_summaries(start_ind, end_ind, save=True, display=False):
  """
  Generate tile summaries for a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save tile summary images.
    display: If True, display tile summary images to screen.
  """
  image_num_list = list()
  for slide_num in range(start_ind, end_ind + 1):
    summary(slide_num, save, display)
    image_num_list.append(slide_num)
  return image_num_list


def singleprocess_images_to_tile_summaries(save=True, display=False, html=True, image_num_list=None):
  """
  Generate tile summaries for training images and optionally save/and or display the tile summaries.

  Args:
    save: If True, save images.
    display: If True, display images to screen.
    html: If True, generate HTML page to display tiled images
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Generating tile summaries\n")

  if image_num_list is not None:
    image_list_to_tile_summaries(image_num_list, save, display)
  else:
    num_training_slides = slide.get_num_training_slides()
    image_num_list = image_range_to_tile_summaries(1, num_training_slides, save, display)

  print("Time to generate tile summaries: %s\n" % str(t.elapsed()))

  if html:
    generate_tiled_html_page(image_num_list)


def multiprocess_images_to_tile_summaries(save=True, display=False, html=True, image_num_list=None):
  """
  Generate tile summaries for all training images using multiple processes (one process per core).

  Args:
    save: If True, save images.
    display: If True, display images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display tiled images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Generating tile summaries (multiprocess)\n")

  if save and not os.path.exists(slide.TILE_SUMMARY_DIR):
    os.makedirs(slide.TILE_SUMMARY_DIR)

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
      results.append(pool.apply_async(image_list_to_tile_summaries, t))
    else:
      results.append(pool.apply_async(image_range_to_tile_summaries, t))

  slide_nums = list()
  for result in results:
    image_nums = result.get()
    slide_nums.extend(image_nums)
    print("Done tiling slides: %s" % image_nums)

  if html:
    generate_tiled_html_page(slide_nums)

  print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))


def html_header():
  """
  Generate an HTML header for viewing tiled images.

  Returns:
    HTML header for viewing tiled images.
  """
  html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
         "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
         "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
         "  <head>\n" + \
         "    <title>Tiled Images</title>\n" + \
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
  Generate an HTML footer for viewing tiled images.

  Returns:
    HTML footer for viewing tiled images.
  """
  html = "</table>\n" + \
         "<script>lazyload();</script>\n" + \
         "</body>\n" + \
         "</html>\n"
  return html


def image_row(slide_num):
  """
  Generate HTML for viewing a tiled image.

  Args:
    slide_num: The slide number.

  Returns:
    HTML for viewing a tiled image.
  """
  return "  <tr>" + \
         "    <td>\n" + \
         "      <a href=\"" + slide.get_training_image_path(slide_num) + "\">\n" + \
         "        " + "S%03d " % slide_num + "Original" + "<br/>\n" + \
         "        <img class=\"lazyload\" src=\"data:image/gif;base64,R0lGODdhAQABAPAAAMPDwwAAACwAAAAAAQABAAACAkQBADs=\" data-src=\"" + slide.get_training_image_path(
    slide_num) + "\" />\n" + \
         "      </a>\n" + \
         "    </td>\n" + \
         "    <td>\n" + \
         "      <a href=\"" + slide.get_filter_image_result(slide_num) + "\">\n" + \
         "        " + "S%03d " % slide_num + "Filtered" + "<br/>\n" + \
         "        <img class=\"lazyload\" src=\"data:image/gif;base64,R0lGODdhAQABAPAAAMPDwwAAACwAAAAAAQABAAACAkQBADs=\" data-src=\"" + slide.get_filter_image_result(
    slide_num) + "\" />\n" + \
         "      </a>\n" + \
         "    </td>\n" + \
         "    <td>\n" + \
         "      <a href=\"" + slide.get_tile_summary_image_path(slide_num) + "\">\n" + \
         "        " + "S%03d " % slide_num + "Tiled" + "<br/>\n" + \
         "        <img class=\"lazyload\" src=\"data:image/gif;base64,R0lGODdhAQABAPAAAMPDwwAAACwAAAAAAQABAAACAkQBADs=\" data-src=\"" + slide.get_tile_summary_image_path(
    slide_num) + "\" />\n" + \
         "      </a>\n" + \
         "    </td>\n" + \
         "  <tr>"


def generate_tiled_html_page(slide_nums):
  """
  Generate an HTML page to view the tiled images.

  Args:
    slide_nums: List of slide numbers.
  """
  html = ""
  html += html_header()

  row = 0
  for slide_num in sorted(slide_nums):
    html += image_row(slide_num)

  html += html_footer()
  text_file = open("tiles.html", "w")
  text_file.write(html)
  text_file.close()


# summary(1, save=True)
# summary(26, save=True)
# image_list_to_tile_summaries([1, 2, 3, 4], display=True)
# image_range_to_tile_summaries(1, 50)
# singleprocess_images_to_tile_summaries(image_num_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], display=False)
multiprocess_images_to_tile_summaries(image_num_list=[1, 2, 3, 4, 5, 6, 7, 8], display=False)
# singleprocess_images_to_tile_summaries()
# multiprocess_images_to_tile_summaries(image_num_list=[5,10,15,20,25,30])
# multiprocess_images_to_tile_summaries()
# summary(3)
