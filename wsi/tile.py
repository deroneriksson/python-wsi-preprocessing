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
from PIL import ImageDraw, ImageFont

TISSUE_THRESHOLD_PERCENT = 50
TISSUE_LOW_THRESHOLD_PERCENT = 5

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024

DISPLAY_TILE_LABELS = False  # If True, add text such as tissue percentage to summary tiles. Requires large tile size.

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.


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
                 no_tissue_color=(255, 0, 0), text_color=(255, 255, 255), text_size=18,
                 font_path="/Library/Fonts/Arial Bold.ttf"):
  """
  Generate summary image/thumbnail showing a 'heatmap' representation of the tissue segmentation of all tiles.

  Args:
    slide_num: The slide number.
    np_img: Image as a NumPy array.
    tile_indices: List of tuples consisting of starting row, ending row, starting column, ending column.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.
    display: If True, display tile summary to screen.
    save: If True, save tile summary image.
    thresh_color: Color representing tissue % above or equal to thresh (default green).
    below_thresh_color: Color representing tissue % below threshold and above or equal to lower thresh (default yellow).
    below_lower_thresh_color: Color representing tissue % below lower threshold and above 0 (default orange).
    no_tissue_color: Color representing no tissue (default red).
    text_color: Font color (default white).
    text_size: Font size.
    font_path: Path to the font to use.
  """
  rows, cols, _ = np_img.shape
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary_img = np.zeros([row_tile_size * num_row_tiles, col_tile_size * num_col_tiles, np_img.shape[2]],
                         dtype=np.uint8)
  # add gray edges so that summary text does not get cut off
  summary_img.fill(120)
  summary_img[0:np_img.shape[0], 0:np_img.shape[1]] = np_img
  summary = filter.np_to_pil(summary_img)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  orig_img = slide.open_image(original_img_path)
  draw_orig = ImageDraw.Draw(orig_img)

  count = 0
  for t in tile_indices:
    count += 1
    r_s, r_e, c_s, c_e = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    tissue_percentage = filter.tissue_percent(np_tile)
    print("TILE [%d:%d, %d:%d]: Tissue %f%%" % (r_s, r_e, c_s, c_e, tissue_percentage))
    if tissue_percentage >= TISSUE_THRESHOLD_PERCENT:
      tile_border(draw, r_s, r_e, c_s, c_e, thresh_color)
      tile_border(draw_orig, r_s, r_e, c_s, c_e, thresh_color)
    elif (tissue_percentage >= TISSUE_LOW_THRESHOLD_PERCENT) and (tissue_percentage < TISSUE_THRESHOLD_PERCENT):
      tile_border(draw, r_s, r_e, c_s, c_e, below_thresh_color)
      tile_border(draw_orig, r_s, r_e, c_s, c_e, below_thresh_color)
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESHOLD_PERCENT):
      tile_border(draw, r_s, r_e, c_s, c_e, below_lower_thresh_color)
      tile_border(draw_orig, r_s, r_e, c_s, c_e, below_lower_thresh_color)
    else:
      tile_border(draw, r_s, r_e, c_s, c_e, no_tissue_color)
      tile_border(draw_orig, r_s, r_e, c_s, c_e, no_tissue_color)
    # filter.display_img(np_tile, text=label, size=14, bg=True)
    if DISPLAY_TILE_LABELS:
      label = "#%d\n%4.2f%%\n[%d,%d] x\n[%d,%d]" % (count, tissue_percentage, r_s, c_s, r_e, c_e)
      font = ImageFont.truetype(font_path, size=text_size)
      draw.text((c_s + 3, r_s + 3), label, (0,0,0), font=font)
      draw.text((c_s + 2, r_s + 2), label, text_color, font=font)
  if display:
    summary.show()
    orig_img.show()
  if save:
    save_tile_summary_image(summary, slide_num)
    save_tile_summary_on_original_image(orig_img, slide_num)


def tile_border(draw, r_s, r_e, c_s, c_e, color):
  for x in range(0, TILE_BORDER_SIZE):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def save_tile_summary_image(pil_img, slide_num):
  """
  Save a tile summary image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Summary Image", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Summary Thumbnail", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Summary on Original Image", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Tile Summary on Original Thumbnail", str(t.elapsed()), thumbnail_filepath))


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

  row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?

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
    generate_tiled_html_result(image_num_list)


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
    generate_tiled_html_result(slide_nums)

  print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))


def image_row(slide_num):
  """
  Generate HTML for viewing a tiled image.

  Args:
    slide_num: The slide number.

  Returns:
    HTML table row for viewing a tiled image.
  """
  orig_img = slide.get_training_image_path(slide_num)
  orig_thumb = slide.get_training_thumbnail_path(slide_num)
  filt_img = slide.get_filter_image_result(slide_num)
  filt_thumb = slide.get_filter_thumbnail_result(slide_num)
  sum_img = slide.get_tile_summary_image_path(slide_num)
  sum_thumb = slide.get_tile_summary_thumbnail_path(slide_num)
  osum_img = slide.get_tile_summary_on_original_image_path(slide_num)
  osum_thumb = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  return "    <tr>\n" + \
         "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Original<br/>\n" % (orig_img, slide_num) + \
         "          <img class=\"lazyload\" src=\"%s\" data-src=\"%s\" />\n" % (filter.b64_img(), orig_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Filtered<br/>\n" % (filt_img, slide_num) + \
         "          <img class=\"lazyload\" src=\"%s\" data-src=\"%s\" />\n" % (filter.b64_img(), filt_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Tiled<br/>\n" % (sum_img, slide_num) + \
         "          <img class=\"lazyload\" src=\"%s\" data-src=\"%s\" />\n" % (filter.b64_img(), sum_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Original Tiled<br/>\n" % (osum_img, slide_num) + \
         "          <img class=\"lazyload\" src=\"%s\" data-src=\"%s\" />\n" % (filter.b64_img(), osum_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "    </tr>\n"


def generate_tiled_html_result(slide_nums):
  """
  Generate HTML to view the tiled images.

  Args:
    slide_nums: List of slide numbers.
  """
  slide_nums = sorted(slide_nums)
  if not slide.TILE_SUMMARY_PAGINATE:
    html = ""
    html += filter.html_header("Tiled Images")

    html += "  <table>\n"
    for slide_num in slide_nums:
      html += image_row(slide_num)
    html += "  </table>\n"

    html += filter.html_footer()
    text_file = open(slide.TILE_SUMMARY_HTML_DIR + os.sep + "tiles.html", "w")
    text_file.write(html)
    text_file.close()
  else:
    total_len = len(slide_nums)
    page_size = slide.TILE_SUMMARY_PAGINATION_SIZE
    num_pages = math.ceil(total_len / page_size)
    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html += filter.html_header("Tiled Images, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"tiles.html\">&lt;</a> "
        else:
          html += "<a href=\"tiles-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"tiles-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        html += image_row(slide_num)
      html += "  </table>\n"

      html += filter.html_footer()
      if page_num == 1:
        text_file = open(slide.TILE_SUMMARY_HTML_DIR + os.sep + "tiles.html", "w")
      else:
        text_file = open(slide.TILE_SUMMARY_HTML_DIR + os.sep + "tiles-%d.html" % page_num, "w")
      text_file.write(html)
      text_file.close()


# summary(1, save=True)
# summary(26, save=True)
# image_list_to_tile_summaries([1, 2, 3, 4], display=True)
# image_range_to_tile_summaries(1, 50)
# singleprocess_images_to_tile_summaries(image_num_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], display=False)
# multiprocess_images_to_tile_summaries(image_num_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], display=False)
# singleprocess_images_to_tile_summaries()
# multiprocess_images_to_tile_summaries(image_num_list=[1,2,3,4,5])
# multiprocess_images_to_tile_summaries(save=False, display=False, html=True)
# multiprocess_images_to_tile_summaries()
# summary(1, display=True, save=True)
# generate_tiled_html_result(slide_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# generate_tiled_html_result(slide_nums=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
