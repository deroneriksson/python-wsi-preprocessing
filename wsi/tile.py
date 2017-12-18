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

TILE_SIZE = 256

img_path = slide.get_filter_image_result(4)
img = slide.open_image(img_path)
# # img.show()
np_img = filter.pil_to_np_rgb(img)


# # print(str(np_img))
# tiles = np.array_split(np_img, [TILE_SIZE, TILE_SIZE])
# print(str(tiles))

# a = np.arange(1,101).reshape(10,10)
# print(str(a))
# tiles = np.array_split(a, 2)
# tiles = np.vsplit(a, 10/5)
# print(str(tiles))
# tiles2 = np.hsplit(tiles, 10/5)
# print(str(tiles2))

def get_tile_indices(np_img, row_tile_size, col_tile_size):
  indices = list()
  rows, cols, _ = np_img.shape
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  num_tiles = 0
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      num_tiles += 1
      print("TILE #%d: [%d:%d, %d:%d]" % (num_tiles, start_r, end_r, start_c, end_c))
      indices.append((start_r, end_r, start_c, end_c))
  return indices


tile_indices = get_tile_indices(np_img, TILE_SIZE, TILE_SIZE)
print("length:" + str(len(tile_indices)))
print(str(tile_indices))
