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

import datetime
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
import sys

BASE_DIR = ".." + os.sep + "data"
# BASE_DIR = os.sep + "Volumes" + os.sep + "BigData" + os.sep + "TUPAC"
SRC_TRAIN_IMG_DIR = BASE_DIR + os.sep + "training_slides"
TRAIN_THUMB_SUFFIX = "thumb-"
TRAIN_IMG_PREFIX = "TUPAC-TR-"
TRAIN_IMG_EXT = ".svs"
THUMB_EXT = ".jpg"
THUMB_SIZE = 4096
DEST_TRAIN_THUMB_DIR = BASE_DIR + os.sep + "training_thumbs_" + str(THUMB_SIZE)


def open_slide(filename):
  """
  Open a whole-slide image.

  Args:
    filename: Name of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


def get_training_slide_path(slide_number):
  """
  Convert slide number to a path to the corresponding WSI training slide file.

  Example:
    5 -> ../data/training_slides/TUPAC-TR-005.svs

  Args:
    slide_number: The slide number.

  Returns:
    Path to the WSI training image file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  slide_filepath = SRC_TRAIN_IMG_DIR + os.sep + TRAIN_IMG_PREFIX + padded_sl_num + TRAIN_IMG_EXT
  return slide_filepath


def get_training_thumb_path(slide_number):
  """
  Convert slide number to a path to the corresponding destination thumbnail file.

  Example:
    5 -> ../data/training_thumbs_4096/TUPAC-TR-005-thumb-4096.jpg

  Args:
    slide_number: The slide number.

  Returns:
    Path to the destination thumbnail file.
  """
  padded_sl_num = str(slide_number).zfill(3)
  thumb_path = DEST_TRAIN_THUMB_DIR + os.sep + TRAIN_IMG_PREFIX + padded_sl_num + "-" + TRAIN_THUMB_SUFFIX + str(
    THUMB_SIZE) + THUMB_EXT
  return thumb_path


def training_slide_to_thumb(slide_number):
  """
  Convert a WSI training slide to a thumbnail.

  Args:
    slide_number: The slide number.
  """
  slide_filepath = get_training_slide_path(slide_number)
  print("Opening Slide #%d: %s" % (slide_number, slide_filepath))
  slide = open_slide(slide_filepath)
  whole_slide_image = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
  whole_slide_image = whole_slide_image.convert("RGB")
  max_size = tuple(round(THUMB_SIZE * d / max(whole_slide_image.size)) for d in whole_slide_image.size)
  thumb = whole_slide_image.resize(max_size, PIL.Image.BILINEAR)
  thumb_path = get_training_thumb_path(slide_number)
  print("Saving thumbnail to: " + thumb_path)
  if not os.path.exists(DEST_TRAIN_THUMB_DIR):
    os.makedirs(DEST_TRAIN_THUMB_DIR)
  thumb.save(thumb_path)


def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.

  Returns:
    The total number of WSI training slide images.
  """
  num_training_slides = len(glob.glob1(SRC_TRAIN_IMG_DIR, "*" + TRAIN_IMG_EXT))
  return num_training_slides


def training_slide_range_to_thumbs(start_ind, end_ind):
  """
  Convert a range of WSI training slides to thumbnails.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).

  Returns:
    The starting index and the ending index of the slides that were converted to thumbnails.
  """
  for slide_num in range(start_ind, end_ind + 1):
    training_slide_to_thumb(slide_num)
  return (start_ind, end_ind)


def singleprocess_convert_training_slides_to_thumbs():
  """
  Convert all WSI training slides to thumbnails using a single process.
  """
  t = Time()

  num_train_images = get_num_training_slides()
  training_slide_range_to_thumbs(1, num_train_images)

  t.elapsed()


def multiprocess_convert_training_slides_to_thumbs():
  """
  Convert all WSI training slides to thumbnails using multiple processes (one process per core).
  Each process will process a range of slide numbers.
  """
  timer = Time()

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  num_train_images = get_num_training_slides()
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
    tasks.append((start_index, end_index))
    if start_index == end_index:
      print("Task #" + str(num_process) + ": Process slide " + str(start_index))
    else:
      print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    results.append(pool.apply_async(training_slide_range_to_thumbs, t))

  for result in results:
    (start_ind, end_ind) = result.get()
    if (start_ind == end_ind):
      print("Done converting slide %d" % start_ind)
    else:
      print("Done converting slides %d through %d" % (start_ind, end_ind))

  timer.elapsed()


def slide_stats():
  """
  Display statistics/graphs about training slides.
  """
  t = Time()

  num_train_images = get_num_training_slides()
  slide_stats = []
  for slide_num in range(1, num_train_images + 1):
    slide_filepath = get_training_slide_path(slide_num)
    print("Opening Slide #%d: %s" % (slide_num, slide_filepath))
    slide = open_slide(slide_filepath)
    (width, height) = slide.dimensions
    print("%dx%d" % (width, height))
    slide_stats.append((width, height))

  max_width = 0
  max_height = 0
  min_width = sys.maxsize
  min_height = sys.maxsize
  total_width = 0
  total_height = 0
  total_size = 0
  which_max_width = 0
  which_max_height = 0
  which_min_width = 0
  which_min_height = 0
  max_size = 0
  min_size = sys.maxsize
  which_max_size = 0
  which_min_size = 0
  for z in range(0, num_train_images):
    (width, height) = slide_stats[z]
    if width > max_width:
      max_width = width
      which_max_width = z + 1
    if width < min_width:
      min_width = width
      which_min_width = z + 1
    if height > max_height:
      max_height = height
      which_max_height = z + 1
    if height < min_height:
      min_height = height
      which_min_height = z + 1
    size = width * height
    if size > max_size:
      max_size = size
      which_max_size = z + 1
    if size < min_size:
      min_size = size
      which_min_size = z + 1
    total_width = total_width + width
    total_height = total_height + height
    total_size = total_size + size

  avg_width = total_width / num_train_images
  avg_height = total_height / num_train_images
  avg_size = total_size / num_train_images

  print("Max width: %d pixels" % max_width)
  print("Max height: %d pixels" % max_height)
  print("Max size: %d pixels (%dMP)" % (max_size, (max_size / 1024 / 1024)))
  print("Min width: %d pixels" % min_width)
  print("Min height: %d pixels" % min_height)
  print("Min size: %d pixels (%dMP)" % (min_size, (min_size / 1024 / 1024)))
  print("Avg width: %d pixels" % avg_width)
  print("Avg height: %d pixels" % avg_height)
  print("Avg size: %d pixels (%dMP)" % (avg_size, (avg_size / 1024 / 1024)))
  print("Max width slide #%d" % which_max_width)
  print("Max height slide #%d" % which_max_height)
  print("Max size slide #%d" % which_max_size)
  print("Min width slide #%d" % which_min_width)
  print("Min height slide #%d" % which_min_height)
  print("Min size slide #%d" % which_min_size)

  x, y = zip(*slide_stats)
  colors = np.random.rand(num_train_images)
  sizes = [10 for n in range(num_train_images)]
  plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
  plt.xlabel("width (pixels)")
  plt.ylabel("height (pixels)")
  plt.title("SVS Image Sizes")
  plt.set_cmap("prism")
  plt.show()

  plt.clf()
  plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
  plt.xlabel("width (pixels)")
  plt.ylabel("height (pixels)")
  plt.title("SVS Image Sizes (Labeled with slide numbers)")
  plt.set_cmap("prism")
  for i in range(num_train_images):
    snum = i + 1
    plt.annotate(str(snum), (x[i], y[i]))
  plt.show()

  plt.clf()
  area = [w * h / 1000000 for (w, h) in slide_stats]
  plt.hist(area, bins=64)
  plt.xlabel("width x height (M of pixels)")
  plt.ylabel("# images")
  plt.title("Distribution of image sizes in millions of pixels")
  plt.show()

  plt.clf()
  whratio = [w / h for (w, h) in slide_stats]
  plt.hist(whratio, bins=64)
  plt.xlabel("width to height ratio")
  plt.ylabel("# images")
  plt.title("Image shapes (width to height)")
  plt.show()

  plt.clf()
  hwratio = [h / w for (w, h) in slide_stats]
  plt.hist(hwratio, bins=64)
  plt.xlabel("height to width ratio")
  plt.ylabel("# images")
  plt.title("Image shapes (height to width)")
  plt.show()

  t.elapsed()


def slide_info(display_all_properties=False):
  """
  Display information (such as properties) about training images.

  Args:
    display_all_properties: If True, display all available slide properties.
  """
  t = Time()

  num_train_images = get_num_training_slides()
  obj_pow_20_list = []
  obj_pow_40_list = []
  obj_pow_other_list = []
  for slide_num in range(1, num_train_images + 1):
    slide_filepath = get_training_slide_path(slide_num)
    print("\nOpening Slide #%d: %s" % (slide_num, slide_filepath))
    slide = open_slide(slide_filepath)
    print("Level count: %d" % slide.level_count)
    print("Level dimensions: " + str(slide.level_dimensions))
    print("Level downsamples: " + str(slide.level_downsamples))
    print("Dimensions: " + str(slide.dimensions))
    print("Associated images: " + str(slide.associated_images))
    print("Format: " + str(slide.detect_format(slide_filepath)))
    if display_all_properties:
      print("Properties: " + str(slide.properties))
      propertymap = slide.properties
      keys = propertymap.keys()
      for key in keys:
        print("  Property: " + str(key) + ", value: " + str(propertymap.get(key)))
    objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    print("Objective power: " + str(objective_power))
    if objective_power == 20:
      obj_pow_20_list.append(slide_num)
    elif objective_power == 40:
      obj_pow_40_list.append(slide_num)
    else:
      obj_pow_other_list.append(slide_num)

  print("\n\nSlide Magnifications:")
  print("  20x Slides: " + str(obj_pow_20_list))
  print("  40x Slides: " + str(obj_pow_40_list))
  print("  ??x Slides: " + str(obj_pow_other_list) + "\n")

  t.elapsed()


class Time:
  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    print("Time elapsed: " + str(time_elapsed))


singleprocess_convert_training_slides_to_thumbs()
# multiprocess_convert_training_slides_to_thumbs()
# slide_stats()
# slide_info(display_all_properties=True)
