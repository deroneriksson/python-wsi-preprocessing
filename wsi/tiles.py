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

# To get around renderer issue on macOS going from Matplotlib image to NumPy image.
import matplotlib

matplotlib.use('Agg')

import PIL
import pathlib
from pathlib import Path
import colorsys
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from wsi import util, filter, slide
from wsi import openslide_overwrite
from wsi.util import Time
import openslide



TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10
HSV_PURPLE = 270
HSV_PINK = 330

############################# classes #########################################



class TileSummary:
    """
    Class for tile summary information.
    """

    wsi_path = None
    tiles_folder_path = None
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = None
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, wsi_path, 
                 tiles_folder_path, 
                 orig_w, 
                 orig_h, 
                 orig_tile_w, 
                 orig_tile_h, 
                 scale_factor, 
                 scaled_w, 
                 scaled_h, 
                 scaled_tile_w,
                 scaled_tile_h, 
                 tissue_percentage, 
                 num_col_tiles, 
                 num_row_tiles):
        self.wsi_path = wsi_path
        self.tiles_folder_path = tiles_folder_path
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scale_factor = scale_factor
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)

    def mask_percentage(self):
        """
        Obtain the percentage of the slide that is masked.

        Returns:
           The amount of the slide that is masked as a percentage.
        """
        return 100 - self.tissue_percentage

    def num_tiles(self):
        """
        Retrieve the total number of tiles.

        Returns:
          The total number of tiles (number of rows * number of columns).
        """
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        """
        Retrieve the tiles ranked by tissue percentage.

        Returns:
           List of the tiles ranked by tissue percentage.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score.

        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def get_tile(self, row, col):
        """
        Retrieve tile by row and column.

        Args:
          row: The row
          col: The column

        Returns:
          Corresponding Tile object.
        """
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile
    
    def top_tiles(self):
        """
        Retrieve the top-scoring tiles.

        Returns:
           List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles = [tile for tile in sorted_tiles
                     if self.check_tile(tile)]
        print(f'Number of saved tiles/all tiles: {len(top_tiles)}/{len(sorted_tiles)}')
        return top_tiles

    def check_tile(self, tile):
        width = tile.o_c_e - tile.o_c_s
        height = tile.o_r_e - tile.o_r_s
        return tile.score > 0.55 and width >= 0.7*self.scale_factor*self.scaled_tile_w and height >= 0.7*self.scale_factor*self.scaled_tile_h


class Tile:
    """
    Class for information about a tile.
    """
    tile_summary = None
    wsi_path = None
    tiles_folder_path = None
    np_scaled_tile = None
    tile_num = None
    r = None
    c = None
    r_s = None
    r_e = None
    c_s = None
    c_e = None
    o_r_s = None
    o_r_e = None
    o_c_s = None
    o_c_e = None
    t_p = None
    color_factor = None
    s_and_v_factor = None
    quantity_factor = None
    score = None
                
    def __init__(self, tile_summary, wsi_path, tiles_folder_path, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                   o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
        self.tile_summary = tile_summary
        self.wsi_path = wsi_path
        self.tiles_folder_path = tiles_folder_path
        self.np_scaled_tile = np_scaled_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
          self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    def get_np_tile(self):
        return tile_to_np_tile(self)

    def save_tile(self):
        save_display_tile(self, save=True, display=False)

    def display_tile(self):
        save_display_tile(self, save=False, display=True)

    def display_with_histograms(self):
        display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_tile(self):
        return self.np_scaled_tile

    def get_pil_scaled_tile(self):
        return util.np_to_pil(self.np_scaled_tile)
    
    def get_wsi_name(self):
        split = self.wsi_path.stem.split('-')
        return f'{split[0]}-{split[1]}-{split[2]}-{split[3]}'
    
    

class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3



############################# functions #########################################
    
def WsiToTiles(wsiPath:pathlib.Path, tilesFolderPath:pathlib.Path, tile_height, tile_width):
    """
    
    """
    
    scale_factor = 32    
    Image.MAX_IMAGE_PIXELS = 10000000000
    openslide.lowlevel._load_image = openslide_overwrite._load_image
    img_pil, original_width, original_height, scaled_width, scaled_height = wsi_to_pil_image(wsiPath, scale_factor)
    img_pil_filtered = filter.filter_img(img_pil)
    tilesummary = create_tilesummary(wsiPath, 
                                     tilesFolderPath, 
                                     img_pil, 
                                     img_pil_filtered, 
                                     original_width, 
                                     original_height, 
                                     scaled_width, 
                                     scaled_height, 
                                     tile_height, 
                                     tile_width, 
                                     scale_factor)
    for tile in tilesummary.top_tiles():
        tile.save_tile()

       
def wsi_to_pil_image(wsi_filepath:pathlib.Path, scale_factor):
    """
    Convert a WSI training slide to a PIL image.

    Args:

    Returns:

    """        
    wsi = openslide.open_slide(str(wsi_filepath))
    large_w, large_h = wsi.dimensions
    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)
    level = wsi.get_best_level_for_downsample(scale_factor)
    img = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    img = img.convert("RGB")
    if(scale_factor > 1):
        img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h



def create_tilesummary(wsiPath, 
                        tilesFolderPath,
                        img_pil:PIL.Image.Image, 
                        img_pil_filtered:PIL.Image.Image, 
                        wsi_original_width:int, 
                        wsi_original_height:int, 
                        wsi_scaled_width:int, 
                        wsi_scaled_height:int, 
                        tile_height:int, 
                        tile_width:int, 
                        scale_factor:int)->TileSummary:
    """
  
    Args:

    """
    np_img = util.pil_to_np_rgb(img_pil)
    np_img_filtered = util.pil_to_np_rgb(img_pil_filtered)

    tile_sum = score_tiles(np_img, 
                           np_img_filtered, 
                           wsiPath, 
                           tilesFolderPath,
                           tile_height,
                           tile_width,
                           scale_factor, 
                           wsi_original_width, 
                           wsi_original_height, 
                           wsi_scaled_width, 
                           wsi_scaled_height)

    return tile_sum


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
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices



def tile_to_pil_tile(tile:Tile):
      """
      Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

      Args:
        tile: Tile object.

      Return:
        Tile as a PIL image.
      """
      s = slide.open_slide(str(tile.wsi_path))
      x, y = tile.o_c_s, tile.o_r_s
      w, h = tile.o_c_e - tile.o_c_s, tile.o_r_e - tile.o_r_s
      tile_region = s.read_region((x, y), 0, (w, h))
      # RGBA to RGB
      pil_img = tile_region.convert("RGB")
      return pil_img


def tile_to_np_tile(tile):
  """
  Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a NumPy image.
  """
  pil_img = tile_to_pil_tile(tile)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img



def get_tile_image_path(tile:Tile):
  """
  Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.

  Args:
    tile: Tile object.

  Returns:
    Path to image tile.
  """
  t = tile
  return os.path.join(tile.tiles_folder_path, 
                      tile.get_wsi_name() + "-" + 'tile' + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + 'png') 



def save_display_tile(tile, save, display):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = get_tile_image_path(tile)
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    tile_pil_img.save(img_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()
    


def score_tiles(img_np:np.array, 
                img_np_filtered:np.array, 
                wsi_path:pathlib.Path,
                tilesFolderPath,
                tile_height:int, 
                tile_width:int, 
                scale_factor:int, 
                wsi_original_width:int, 
                wsi_original_height:int, 
                wsi_scaled_width:int, 
                wsi_scaled_height:int,) -> TileSummary:
    """
    Score all tiles for a slide and return the results in a TileSummary object.

    Args:

    Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
    """
    #img_path = slide.get_filter_image_result(slide_num)
    #o_w, o_h, w, h = slide.parse_dimensions_from_image_filename(img_path)
    #np_img = slide.open_image_np(img_path)

    tile_height_scaled = round(tile_height / scale_factor)  # use round?
    tile_width_scaled = round(tile_width / scale_factor)  # use round?

    num_row_tiles, num_col_tiles = get_num_tiles(wsi_scaled_height, 
                                                 wsi_scaled_width, 
                                                 tile_height_scaled, 
                                                 tile_width_scaled)

    tile_sum = TileSummary(wsi_path=wsi_path,
                           tiles_folder_path=tilesFolderPath,
                             orig_w=wsi_original_width,
                             orig_h=wsi_original_height,
                             orig_tile_w=tile_width,
                             orig_tile_h=tile_height,
                             scale_factor=scale_factor,
                             scaled_w=wsi_scaled_width,
                             scaled_h=wsi_scaled_height,
                             scaled_tile_w=tile_width_scaled,
                             scaled_tile_h=tile_height_scaled,
                             tissue_percentage=filter.tissue_percent(img_np_filtered),
                             num_col_tiles=num_col_tiles,
                             num_row_tiles=num_row_tiles)   
    

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    tile_indices = get_tile_indices(wsi_scaled_height, wsi_scaled_width, tile_height_scaled, tile_width_scaled)
    for t in tile_indices:
        count += 1  # tile_num
        r_s, r_e, c_s, c_e, r, c = t
        np_tile = img_np_filtered[r_s:r_e, c_s:c_e]
        t_p = filter.tissue_percent(np_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.MEDIUM:
            medium += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        elif amount == TissueQuantity.NONE:
            none += 1
            
        o_c_s, o_r_s = slide.small_to_large_mapping((c_s, r_s), (wsi_original_width, wsi_original_height), scale_factor)
        #print("o_c_s: " + str(o_c_s))
        #print("o_r_s: " + str(o_r_s))
        o_c_e, o_r_e = slide.small_to_large_mapping((c_e, r_e), (wsi_original_width, wsi_original_height), scale_factor)
        #print("o_c_e: " + str(o_c_e))
        #print("o_r_e: " + str(o_r_e))

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > tile_width:
            o_c_e -= 1
        if (o_r_e - o_r_s) > tile_height:
            o_r_e -= 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, r, c)

        np_tile #if small_tile_in_tile else None
        
        tile = Tile(tile_sum, wsi_path, tilesFolderPath, np_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                    o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
        tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none
      
    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
        t.rank = rank

    return tile_sum


def scoring_function_1(tissue_percent, combined_factor):
    """
    use this, if you want tissue with lots of cells
    """
    return tissue_percent * combined_factor / 1000.0
def scoring_function_2(tissue_percent, combined_factor):
    """
    use this, if you only care that there is any tissue in the tile
    """
    return (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0


def score_tile(np_tile, tissue_percent, row, col, scoring_function = scoring_function_1):
      """
      Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

      Args:
        np_tile: Tile as NumPy array.
        tissue_percent: The percentage of the tile judged to be tissue.
        slide_num: Slide number.
        row: Tile row.
        col: Tile column.

      Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
      """
      color_factor = hsv_purple_pink_factor(np_tile)
      s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
      amount = tissue_quantity(tissue_percent)
      quantity_factor = tissue_quantity_factor(amount)
      combined_factor = color_factor * s_and_v_factor
      score = scoring_function(tissue_percent, combined_factor)
      # scale score to between 0 and 1
      score = 1.0 - (10.0 / (10.0 + score))
      return score, color_factor, s_and_v_factor, quantity_factor



def tissue_quantity_factor(amount):
  """
  Obtain a scoring factor based on the quantity of tissue in a tile.

  Args:
    amount: Tissue amount as a TissueQuantity enum value.

  Returns:
    Scoring factor based on the tile tissue quantity.
  """
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor


def tissue_quantity(tissue_percentage):
  """
  Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage.

  Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE




def rgb_to_hues(rgb):
  """
  Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    1-dimensional array of hue values in degrees
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  h = filter.filter_hsv_to_h(hsv, display_np_info=False)
  return h


def hsv_saturation_and_value_factor(rgb):
  """
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  s = filter.filter_hsv_to_s(hsv)
  v = filter.filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor


def hsv_purple_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for purple.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV purple deviation.
  """
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for pink.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV pink deviation.
  """
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation


def hsv_purple_pink_factor(rgb):
  """
  Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
  """
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor


def hsv_purple_vs_pink_average_factor(rgb, tissue_percentage):
  """
  Function to favor purple (hematoxylin) over pink (eosin) staining based on the distance of the HSV hue average
  from purple and pink.

  Args:
    rgb: Image as RGB NumPy array
    tissue_percentage: Amount of tissue on the tile

  Returns:
    Factor, where >1 to boost purple slide scores, <1 to reduce pink slide scores, or 1 no effect.
  """

  factor = 1
  # only applies to slides with a high quantity of tissue
  if tissue_percentage < TISSUE_HIGH_THRESH:
    return factor

  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 200]  # Remove hues under 200
  if len(hues) == 0:
    return factor
  avg = np.average(hues)
  # pil_hue_histogram(hues).show()

  pu = HSV_PURPLE - avg
  pi = HSV_PINK - avg
  pupi = pu + pi
  # print("Av: %4d, Pu: %4d, Pi: %4d, PuPi: %4d" % (avg, pu, pi, pupi))
  # Av:  250, Pu:   20, Pi:   80, PuPi:  100
  # Av:  260, Pu:   10, Pi:   70, PuPi:   80
  # Av:  270, Pu:    0, Pi:   60, PuPi:   60 ** PURPLE
  # Av:  280, Pu:  -10, Pi:   50, PuPi:   40
  # Av:  290, Pu:  -20, Pi:   40, PuPi:   20
  # Av:  300, Pu:  -30, Pi:   30, PuPi:    0
  # Av:  310, Pu:  -40, Pi:   20, PuPi:  -20
  # Av:  320, Pu:  -50, Pi:   10, PuPi:  -40
  # Av:  330, Pu:  -60, Pi:    0, PuPi:  -60 ** PINK
  # Av:  340, Pu:  -70, Pi:  -10, PuPi:  -80
  # Av:  350, Pu:  -80, Pi:  -20, PuPi: -100

  if pupi > 30:
    factor *= 1.2
  if pupi < -30:
    factor *= .8
  if pupi > 0:
    factor *= 1.2
  if pupi > 50:
    factor *= 1.2
  if pupi < -60:
    factor *= .8

  return factor