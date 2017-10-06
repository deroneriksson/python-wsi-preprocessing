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
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from wsi.filter import np_info
from wsi.slide import Time


def filter_autolevel(np_img, disk_size=1, output_type="uint8"):
  t = Time()
  autolevel = sk_filters.rank.autolevel(np_img, selem=sk_morphology.disk(disk_size))
  autolevel = np_img > autolevel
  if output_type == "bool":
    pass
  elif output_type == "float":
    autolevel = autolevel.astype(float)
  else:
    autolevel = autolevel.astype("uint8") * 255
  np_info(autolevel, "Auto Level", t.elapsed())
  return autolevel


def filter_subtract_mean(np_img, neigh=50, output_type="uint8"):
  t = Time()
  subtract_mean = sk_filters.rank.subtract_mean(np_img, selem=np.ones((neigh, neigh)))
  subtract_mean = np_img > subtract_mean
  if output_type == "bool":
    pass
  elif output_type == "float":
    subtract_mean = subtract_mean.astype(float)
  else:
    subtract_mean = subtract_mean.astype("uint8") * 255
  np_info(subtract_mean, "Subtract Mean", t.elapsed())
  return subtract_mean


def filter_modal(np_img, neigh=50, output_type="uint8"):
  t = Time()
  modal = sk_filters.rank.modal(np_img, selem=np.ones((neigh, neigh)))
  np_info(modal, "Modal", t.elapsed())
  return modal

# def filter_try(np_img, neigh=50, output_type="uint8"):
#   t = Time()
#   tryit = sk_filters.rank.pop_bilateral(np_img, selem=np.ones((neigh, neigh)))
#   # tryit = np_img < tryit
#   if output_type == "bool":
#     pass
#   elif output_type == "float":
#     tryit = tryit.astype(float)
#   else:
#     tryit = tryit.astype("uint8") * 255
#   np_info(tryit, "Try It", t.elapsed())
#   return tryit
