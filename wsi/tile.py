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

ROW_TILE_SIZE = 256
COL_TILE_SIZE = 256
TISSUE_THRESHOLD_PERCENT = 80

img_path = slide.get_filter_image_result(4)
img = slide.open_image(img_path)
np_img = filter.pil_to_np_rgb(img)


def get_tile_indices(np_img, row_tile_size, col_tile_size):
  indices = list()
  rows, cols, _ = np_img.shape
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c))
  return indices


tile_indices = get_tile_indices(np_img, ROW_TILE_SIZE, COL_TILE_SIZE)
print("length:" + str(len(tile_indices)))
print(str(tile_indices))
for t in tile_indices:
  r_s, r_e, c_s, c_e = t
  np_tile = np_img[r_s:r_e, c_s:c_e]
  tissue_percentage = filter.tissue_percent(np_tile)
  print("TILE [%d:%d, %d:%d]: Tissue %f%%" % (r_s, r_e, c_s, c_e, tissue_percentage))
  label = "[%d:%d, %d:%d]: %4.2f%%" % (r_s, r_e, c_s, c_e, tissue_percentage)
  filter.display_img(np_tile, text=label, size=14, bg=True)
