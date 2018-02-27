---
layout: default
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

* Table of contents.
{:toc}


# Whole-Slide Image Preprocessing in Python


## Project Introduction

The primary goal of the [Tumor Proliferation Assessment Challenge 2016 (TUPAC16)](http://tupac.tue-image.nl/) is to
develop algorithms to automatically predict breast cancer tumor proliferation scores. In this challenge, the training
set consists of 500 whole-slide images which are scored (1, 2, or 3) by pathologists based on mitosis
counts. A higher proliferation score indicates a worse prognosis since higher tumor proliferation rates are
correlated with worse outcomes. The tissue samples are stained with hematoxylin and eosin (H&E).

One of our first approaches to this challenge was to apply deep learning to breast cancer whole-slide images,
following an approach similar to the process used by Ertosun and Rubin in
[Automated Grading of Gliomas using Deep Learning in Digital Pathology Images: A modular approach with ensemble of
convolutional neural networks](https://web.stanford.edu/group/rubinlab/pubs/2243353.pdf). One important part of the
technique described by Ertosun and Rubin involves image preprocessing, where large whole-slide images are divided into
tiles and only tiles that consist of at least 90% tissue are further analyzed. Tissue is determined by hysteresis
thresholding on the grayscale image complement.

The three TUPAC16 challenge tasks were won by Paeng et al, described in
[A Unified Framework for Tumor Proliferation Score Prediction in Breast
Histopathology](https://pdfs.semanticscholar.org/7d9b/ccac7a9a850cc84a980e5abeaeac2aef94e6.pdf). In their technique,
identification of tissue regions in whole-slide images is done using Otsu thresholding, morphological operations, and
binary dilation.

Tissue identification in whole-slide images can be a very important precursor to deep learning, since accurate tissue
identification can decrease the quantity of data and increase the quality of the data to be analyzed. This
can lead to faster, more accurate model training.
In this tutorial, we will take a look at whole-slide image processing and will describe various filters
that can be used to increase the accuracy of tissue identification.
After determining a useful set of filters for tissue segmentation, we'll divide slides into tiles and determine sets
of tiles that typically represent good tissue samples.

In summary, we will scale down whole-slide images, apply filters to these scaled-down images for tissue segmentation,
break the slides into tiles, score the tiles, and then retrieve the top tiles based on their scores.

| **5 Steps** |
| -------------------- |
| ![5 Steps](images/5-steps.png "5 Steps") |


## Setup

This project makes heavy use of Python3 and various Python packages. A full
description of Python is beyond the scope of this tutorial, but some quick setup steps on OS X
follow.

Install a package manager such as [Homebrew](https://brew.sh/).

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Install [Python3](https://www.python.org/).

    brew install python3

Install [OpenSlide](http://openslide.org/). OpenSlide can be used to read whole slide images.
Note that OpenSlide is licensed under the [LGPL 2.1
License](https://raw.githubusercontent.com/openslide/openslide/master/lgpl-2.1.txt).

    brew install openslide

Next, we can install a variety of useful Python packages using the [pip3](https://pip.pypa.io/en/stable/)
package manager. These packages include:
[ipython](https://pypi.python.org/pypi/ipython),
[jupyter](https://pypi.python.org/pypi/jupyter),
[matplotlib](https://pypi.python.org/pypi/matplotlib/),
[numpy](https://pypi.python.org/pypi/numpy),
[openslide-python](https://pypi.python.org/pypi/openslide-python),
[pandas](https://pypi.python.org/pypi/pandas),
[scikit-image](https://pypi.python.org/pypi/scikit-image),
[scikit-learn](https://pypi.python.org/pypi/scikit-learn),
and [scipy](https://pypi.python.org/pypi/scipy).

    pip3 install -U ipython jupyter matplotlib numpy openslide-python pandas scikit-image scikit-learn scipy

We will utilize scikit-image filters in this tutorial that are not present in the
latest released version of scikit-image at the time of this writing. We will install scikit-image
from source, as described in the README at [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image).

    git clone https://github.com/scikit-image/scikit-image.git
    cd scikit-image
    pip3 install -r requirements.txt
    pip3 install .


## Whole Slide Imaging Background

A whole-slide image is a digital representation of a microscopic slide, typically at a very high level of magnification
such as 20x or 40x. As a result of this high magnification, whole slide images are typically very large in size.
The maximum file size for a single whole-slide image in our training dataset was 3.4 GB, with an average over 1 GB.

| **WSI Example Slide** |
| -------------------- |
| ![WSI Example Slide](images/wsi-example.png "WSI Example Slide") |


A whole-slide image is created by a microscope that scans a slide and combines smaller images into a large image.
Techniques include combining scanned square tiles into a whole-slide image and combining scanned strips
into a resulting whole-slide image. Occasionally, the smaller constituent images can be
visually discerned, as in the shaded area at the top of the slide seen below.

| **Combining Smaller Images into a Whole-Slide Image** |
| -------------------- |
| ![Combining Smaller Images into a Whole-Slide Image](images/slide-scan.png "Combining Smaller Images into a Whole-Slide Image") |


A fairly unusual feature of whole-slide images is the very large image size.
For our training data set of 500 images, the width varied from 19,920 pixels to 198,220 pixels,
with an average of 101,688 pixels. The height varied from 13,347 pixels to 256,256 pixels,
with an average of 73,154 pixels. The image total pixel sizes varied from
369,356,640 to 35,621,634,048 pixels, with an average of
7,670,709,628 pixels. The 500 training images take up a total of 525 GB of storage space.

| **Training Image Sizes** |
| -------------------- |
| ![Training Image Sizes](images/svs-image-sizes.png "Training Image Sizes") |


Here we see a histogram distribution of the training image sizes in megapixels.

| **Distribution of Images Based on Number of Pixels** |
| -------------------- |
| ![Distribution of Image Sizes](images/distribution-of-svs-image-sizes.png "Distribution of Image Sizes") |


The [OpenSlide](http://openslide.org/) project can be used to read a variety of whole-slide
image formats, including the [Aperio *.svs slide format](http://openslide.org/formats/aperio/)
of our training image set. This is a pyramidal, tiled format, where the massive slide is composed of
a large number of constituent tiles.

To use the OpenSlide Python interface to view whole slide images, we can clone the
[OpenSlide Python interface from GitHub](https://github.com/openslide/openslide-python)
and utilize the included DeepZoom `deepzoom_multiserver.py` script.

    git clone https://github.com/openslide/openslide-python.git
    cd openslide-python/examples/deepzoom
    python3 deepzoom_multiserver.py -Q 100 WSI_DIRECTORY

The `deepzoom_multiserver.py` script starts a web interface on port 5000 and displays
the image files at the specified file system location (the `WSI_DIRECTORY` value above,
which could be a location such as `~/git/python-wsi-preprocessing/data/`). If image
files exist in subdirectories, they will also be displayed in the list of available
slides.

If this viewing application is installed on a server that also hosts the whole-slide image repository, this
offers a convenient mechanism for users to view the slides without requiring local storage space.

| **OpenSlide Available Slides** |
| -------------------- |
| ![OpenSlide Available Slides](images/openslide-available-slides.png "OpenSlide Available Slides") |


Here we can see the initial view of one of the whole-slide images viewed in a web browser.

| **OpenSlide Whole Slide Image** |
| -------------------- |
| ![OpenSlide Whole Slide Image](images/openslide-whole-slide-image.png "OpenSlide Whole Slide Image") |


Using this web interface, the whole-slide image can be zoomed to the highest magnification, revealing fine details at
the tile level. Zooming and scrolling operations make it relatively easy to visually peruse the whole slide image.

| **OpenSlide Whole Slide Image Zoomed** |
| -------------------- |
| ![OpenSlide Whole Slide Image Zoomed](images/openslide-whole-slide-image-zoomed.png "OpenSlide Whole Slide Image Zoomed") |


## WSI Format Conversion

To develop a set of filters that can be applied to an entire set of large whole-slide images, two of the first issues
we are confronted with are the size of the data and the format of the data. As mentioned, for our training dataset,
the average svs file size is over 1 GB and we have 500 total images. Additionally, the svs format is a fairly unusual
format which typically can't be visually displayed by common applications and operating systems. Therefore, we will
develop some code to overcome these important issues. Using OpenSlide and Python, we'll convert the training dataset to
smaller images in a common format.

In the `wsi/slide.py` file, we have many functions that can be used in relation to the original svs images. Of
particular importance are the following functions:

    open_slide()
    slide_info(display_all_properties=True)
    slide_stats()
    training_slide_to_image()
    singleprocess_training_slides_to_images()
    multiprocess_training_slides_to_images()

The `open_slide()` function uses OpenSlide to read in an svs file. The `slide_info()` function displays metadata
associated with each svs file. The `slide_stats()` function looks at all images and summarizes pixel size information
about the set of slides. It also generates a variety of charts for a visual representation of the slide statistics.
The `training_slide_to_image()` function converts a single svs slide to a smaller image in a more common format such as
jpg or png. The `singleprocess_training_slides_to_images()` function converts all svs slides to smaller images,
and the `multiprocess_training_slides_to_images()` function uses multiple processes (1 process per core) to
speed up the slide conversion process. For the last three functions, when an image is saved, a thumbnail image is also
saved. By default, the thumbnail has a maximum height or width of 300 pixels and is jpg format.

One of the first actions we can take to become more familiar with the training dataset is to have a look at the metadata
associated with each image, which we can do with the `slide_info()` function. Here we can see a sample of this
metadata for Slide #1:

```
Opening Slide #1: /Volumes/BigData/TUPAC/training_slides/TUPAC-TR-001.svs
Level count: 5
Level dimensions: ((130304, 247552), (32576, 61888), (8144, 15472), (2036, 3868), (1018, 1934))
Level downsamples: (1.0, 4.0, 16.0, 64.0, 128.0)
Dimensions: (130304, 247552)
Objective power: 40
Associated images:
  macro: <PIL.Image.Image image mode=RGBA size=497x1014 at 0x114B69F98>
  thumbnail: <PIL.Image.Image image mode=RGBA size=404x768 at 0x114B69FD0>
Format: aperio
Properties:
  Property: aperio.AppMag, value: 40
  Property: aperio.MPP, value: 0.16437
  Property: openslide.comment, value: Aperio Image Library v11.0.37
130304x247552 (256x256) JPEG/RGB Q=40;Mirax Digital Slide|AppMag = 40|MPP = 0.16437
  Property: openslide.level-count, value: 5
  Property: openslide.level[0].downsample, value: 1
  Property: openslide.level[0].height, value: 247552
  Property: openslide.level[0].tile-height, value: 256
  Property: openslide.level[0].tile-width, value: 256
  Property: openslide.level[0].width, value: 130304
  Property: openslide.level[1].downsample, value: 4
  Property: openslide.level[1].height, value: 61888
  Property: openslide.level[1].tile-height, value: 256
  Property: openslide.level[1].tile-width, value: 256
  Property: openslide.level[1].width, value: 32576
  Property: openslide.level[2].downsample, value: 16
  Property: openslide.level[2].height, value: 15472
  Property: openslide.level[2].tile-height, value: 256
  Property: openslide.level[2].tile-width, value: 256
  Property: openslide.level[2].width, value: 8144
  Property: openslide.level[3].downsample, value: 64
  Property: openslide.level[3].height, value: 3868
  Property: openslide.level[3].tile-height, value: 256
  Property: openslide.level[3].tile-width, value: 256
  Property: openslide.level[3].width, value: 2036
  Property: openslide.level[4].downsample, value: 128
  Property: openslide.level[4].height, value: 1934
  Property: openslide.level[4].tile-height, value: 256
  Property: openslide.level[4].tile-width, value: 256
  Property: openslide.level[4].width, value: 1018
  Property: openslide.mpp-x, value: 0.16436999999999999
  Property: openslide.mpp-y, value: 0.16436999999999999
  Property: openslide.objective-power, value: 40
  Property: openslide.quickhash-1, value: 0e0631ade42ae3384aaa727ce2e36a8272fe67039c513e17dccfdd592f6040cb
  Property: openslide.vendor, value: aperio
  Property: tiff.ImageDescription, value: Aperio Image Library v11.0.37
130304x247552 (256x256) JPEG/RGB Q=40;Mirax Digital Slide|AppMag = 40|MPP = 0.16437
  Property: tiff.ResolutionUnit, value: inch
```

The most important metadata for our purposes is that the slide has a width of 130,304 pixels and a height of
247,552 pixels. Note that these values are displayed as width followed by height. For most of our image processing,
we will be using NumPy arrays, where rows (height) are followed by columns (width).

If we visually look over the metadata associated with other images in the training dataset, we see that the slides
are not consistent in their various properties such as the number of levels contained in the svs files. The metadata
implies that the dataset comes from a variety of sources. The variability in the slides, especially regarding
issues such as H&E staining and pen marks on the slides, needs to be considered during our filter development.

If we call the `slide_stats()` function, in addition to the charts, we obtain a table of pixel statistics, shown
below.

**Training Images Statistics**<br/>

| Attribute  | Size                  | Slide # |
| ---------- | --------------------- | ------- |
| Max width  |        198,220 pixels | 10      |
| Max height |        256,256 pixels | 387     |
| Max size   | 35,621,634,048 pixels | 387     |
| Min width  |         19,920 pixels | 112     |
| Min height |         13,347 pixels | 108     |
| Min size   |    369,356,640 pixels | 112     |
| Avg width  |        101,688 pixels |         |
| Avg height |         73,154 pixels |         |
| Avg size   |  7,670,709,629 pixels |         |


The `wsi/slide.py` file contains constants that can be used to control various image conversion settings. For example,
the `SCALE_FACTOR` constant controls the factor by which the slides will be scaled down. Its default value is 32,
meaning that the height and width will be scaled down by a factor of 32. This means that when we perform filtering,
it will be performed on an image 1/1024th the size of the original high-resolution image.
The `DEST_TRAIN_EXT` constant controls the output format. We will use `png`.

Using OS X with an external hard drive containing the training set, the following conversion times using
`singleprocess_training_slides_to_images()` and `multiprocess_training_slides_to_images()`
on the 500 image training set were obtained:

**Training Image Dataset Conversion Times**<br/>

| Format | Processes      | Time   |
| ------ | -------------- | ------ |
| jpg    | single process | 26m09s  |
| jpg    | multi process  | 10m21s  |
| png    | single process | 42m59s |
| png    | multi process  | 11m58s  |


After calling `multiprocess_training_slides_to_images()` using the `png` format, we have 500 whole-slide
images in lossless `png` format that we will examine in greater detail in relation to our filters.


## Image Saving, Displaying, and Conversions

In order to load, save, and display images, we use the Python [Pillow](https://pillow.readthedocs.io/en/4.3.x/)
package. In particular, we make use of the Image module, which contains an Image class used to represent an image.
The `wsi/slide.py` file contains an `open_image()` method to open an image stored in the file system.
The `get_training_image_path()` function takes a slide number and returns the path to the corresponding training slide
file. These functions can be used to open a converted image file as a PIL Image.

```
img_path = slide.get_training_image_path(4)
img = slide.open_image(img_path)
```


To mathematically manipulate the images, we use NumPy arrays. The `wsi/util.py` file contains a
`pil_to_np_rgb()` function that converts a PIL Image to a 3-dimensional NumPy array. The first dimension
represents the number of rows, the second dimension represents the number of columns, and the third dimension
represents the channel (red, green, and blue).

```
rgb = util.pil_to_np_rgb(img)
```


The `wsi/util.py` file contains an `np_to_pil()` function that converts a NumPy array to a PIL Image.

For convenience, the `display_img()` function can be used to display a NumPy array image. Text can be added to
the displayed image, which can be very useful when visually comparing the results of multiple filters.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = util.pil_to_np_rgb(img)
util.display_img(rgb, "RGB")
```

| **Display Image with Text** |
| -------------------- |
| ![Display Image with Text](images/display-image-with-text.png "Display Image with Text") |


When performing operations on NumPy arrays, functions in the `wsi/filter.py` file will often utilize the
`util.np_info()` function to display information about the NumPy array and the amount of time required to perform the
operation. For example, the above call to `pil_to_np_rgb()` internally calls `np_info()`:

```
t = Time()
rgb = np.asarray(pil_img)
np_info(rgb, "RGB", t.elapsed())
return rgb
```

This call to `np_info()` generates console output such as the following:

```
RGB                  | Time: 0:00:00.162484  Type: uint8   Shape: (1385, 1810, 3)
```

We see that the PIL-to-NumPy array conversion took 0.16s. The type of the NumPy array is `uint8`, which means
that each pixel is represented by a red, green, and blue value from 0 to 255. The image has a height of 1385 pixels
and a width of 1810 pixels.

We can obtain additional information about NumPy arrays by setting the `ADDITIONAL_NP_STATS` constant to `True`.
If we rerun the above code with `ADDITIONAL_NP_STATS = True`, we see the following:

```
RGB                  | Time: 0:00:00.157696 Min:   2.00  Max: 255.00  Mean: 182.62  Binary: F  Type: uint8   Shape: (1385, 1810, 3)
```

The minimum value is 2, the maximum value is 255, the mean value is 182.62, and binary is false, meaning that the
image is not a binary image. A binary image is an image that consists of only two values (True or False, 1.0 or 0.0,
255 or 0). Binary images are produced by actions such as thresholding.

When interacting with NumPy image processing code, the information provided by `np_info()` can be extremely useful.
For example, some functions return boolean NumPy arrays, other functions return float NumPy arrays, and other
functions may return `uint8` NumPy arrays. Before performing actions on a NumPy array, it's usually necessary to know
the data type of the array and the nature of the data in that array. For performance reasons, normally
`ADDITIONAL_NP_STATS` should be set to `False`.


## Filters

Now, let's take a look at several ways that our images can be filtered. Filters are represented by functions
in the `wsi/filter.py` file and have `filter_` prepended to the function names.


### RGB to Grayscale

A very common task in image processing is to convert an RGB image to a grayscale image. In this process, the three
color channels are replaced by a single grayscale channel. The grayscale pixel value is computed by combining the
red, green, and blue values in set percentages. The `filter_rgb_to_grayscale()` function multiplies the red value by
21.25%, the green value by 71.54%, and the blue value by 7.21%, and these values are added together to obtain the
grayscale pixel value. 

Although the PIL Image `convert("L")` function can also be used to convert an RGB image to a grayscale image, we
will instead use the `filter_rgb_to_grayscale()` function, since having a reference to the RGB image as a NumPy array
can often be very useful during image processing.

Below, we'll open a slide as a PIL Image, convert this to an RGB NumPy array, and then convert this to a grayscale
NumPy array.


```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
display_img(grayscale, "Grayscale")
```

Here we see the displayed grayscale image.

| **Grayscale Filter** |
| -------------------- |
| ![Grayscale Filter](images/grayscale.png "Grayscale Filter") |


In the console, we see that the grayscale image is a two-dimensional NumPy array, since the 3 color channels have
been combined into a single grayscale channel. The data type is `uint8` and the pixel is represented by an integer
value between 0 and 255.


```
RGB                  | Time: 0:00:00.220014  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.130243  Type: uint8   Shape: (1567, 2048)
```


### Complement

In our whole-slide image training set, the slide backgrounds are illuminated by white light, which means that a `uint8`
pixel in the background of a grayscale image is usually close to or equal to 255. However, conceptually and
mathematically it is often useful to have background values close to or equal to 0. For example, this is useful in
thresholding, where we might ask if a pixel value is above a particular threshold value. This can also be useful in
masking out a background of 0 values from an image.

The `filter_complement()` function inverts the values and thus the colors in the NumPy array representation of an image.
Below, we use the `filter_complement()` function to invert the previously obtained grayscale image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
complement = filter_complement(grayscale)
display_img(complement, "Complement")
```

| **Complement Filter** |
| -------------------- |
| ![Complement Filter](images/complement.png "Complement Filter") |


In the console output, we see that computing the complement is a very fast operation.

```
RGB                  | Time: 0:00:00.225361  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.136738  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.002159  Type: uint8   Shape: (1567, 2048)
```


### Thresholding


#### Basic Threshold

With basic thresholding, a binary NumPy array is generated, where each value in the resulting NumPy array indicates
whether the corresponding pixel in the original image is above a particular threshold value. So, a
pixel with a value of 160 with a threshold of 150 would generate a True (or 255, or 1.0), and a pixel with a value
of 140 with a threshold of 150 would generate a False (or 0, or 0.0).

Here, we apply a basic threshold with a threshold value of 100 to the grayscale complement of the original image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
complement = filter_complement(grayscale)
hyst = filter_threshold(complement, threshold=100)
display_img(hyst, "Threshold")
```

The result is a binary image where pixel values that were above 100 are shown in white and pixel values that were 100 or
lower are shown in black.

| **Basic Threshold Filter** |
| -------------------- |
| ![Basic Threshold Filter](images/basic-threshold.png "Basic Threshold Filter") |


In the console output, we see that basic thresholding is a very fast operation.

```
RGB                  | Time: 0:00:00.207232  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.132555  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.001420  Type: uint8   Shape: (1567, 2048)
Threshold            | Time: 0:00:00.001329  Type: bool    Shape: (1567, 2048)
```


#### Hysteresis Threshold

Hysteresis thresholding is a two-level threshold. The top-level threshold is treated in a similar fashion as basic
thresholding. The bottom-level threshold must be exceeded and must be connected to the top-level threshold. This
processes typically results in much better thresholding than basic thresholding. The values of the top and bottom
thresholds for images can be tested through experimentation.

The `filter_hysteresis_threshold()` function uses default bottom and top threshold values of 50 and 100. The
default array output type from this function is `uint8`. Since the output of this function is a binary image, the
values in the output array will be either 255 or 0. The output type of this function can be specified using the
`output_type` parameter. Note that when performing masking, it is typically more useful to have a NumPy array of
boolean values.

Here, we perform a hysteresis threshold on the complement of the grayscale image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
complement = filter_complement(grayscale)
hyst = filter_hysteresis_threshold(complement)
display_img(hyst, "Hysteresis Threshold")
```

In the generated image, notice that the result is a binary image. All pixel values are either white (255) or black (0).
The red display text in the corner can be ignored since it is for informational purposes only and is not present when
we save the images to the file system.

Notice that the shadow area along the top edge of the slide makes it through the hysteresis threshold filter even
though conceptually it is background and should not be treated as tissue.

| **Hysteresis Threshold Filter** |
| -------------------- |
| ![Hysteresis Threshold Filter](images/hysteresis-threshold.png "Hysteresis Threshold Filter") |


Here we see the console output from our filter operations.

```
RGB                  | Time: 0:00:00.213741  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.126530  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.001428  Type: uint8   Shape: (1567, 2048)
Hysteresis Threshold | Time: 0:00:00.115570  Type: uint8   Shape: (1567, 2048)
```


#### Otsu Threshold

Thresholding using Otsu's method is another popular thresholding technique. This technique was used in the image
processing described in [A Unified Framework for Tumor Proliferation Score Prediction in Breast
Histopathology](https://pdfs.semanticscholar.org/7d9b/ccac7a9a850cc84a980e5abeaeac2aef94e6.pdf). This technique is
described in more detail at
[https://en.wikipedia.org/wiki/Otsu%27s_method](https://en.wikipedia.org/wiki/Otsu%27s_method).

Let's try Otsu's method on the complement image as we did when demonstrating hysteresis thresholding.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
complement = filter_complement(grayscale)
otsu = filter_otsu_threshold(complement)
display_img(otsu, "Otsu Threshold")
```


In the resulting image, we see that Otsu's method generates roughly similar results as hysteresis thresholding.
However, Otsu's method is less aggressive in terms of what it lets through for the tissue in the upper left
area of the slide. The background shadow area at the top of the slide is passed through the
filter in a similar fashion as hysteresis thresholding. Most of the slides in the training set do not have such a
pronounced shadow area, but it would be nice to have an image processing solution that treats the shadow area as
background.

| **Otsu Threshold Filter** |
| -------------------- |
| ![Otsu Threshold Filter](images/otsu-threshold.png "Otsu Threshold Filter") |


In terms of performance, thresholding using Otsu's method is very fast, as we see in the console output.

```
RGB                  | Time: 0:00:00.206994  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.132579  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.001439  Type: uint8   Shape: (1567, 2048)
Otsu Threshold       | Time: 0:00:00.018829  Type: uint8   Shape: (1567, 2048)
```


### Contrast

For an image, suppose we have a histogram of the number of pixels (y-axis) plotted against the range
of possible pixel values (x-axis, 0 to 255). Contrast is a measure of the difference in intensities. An image with
low contrast is typically dull and details are not clearly seen visually. An image with high contrast is typically
sharp and details can clearly be discerned. Increasing the contrast in an image can be used to bring out various details
in the image.


#### Contrast Stretching

One form of increasing the contrast in an image is contrast stretching. Suppose that all intensities in an image occur
between 100 and 150 on a scale from 0 to 255. If we rescale the intensities so that 100 now corresponds to 0 and
150 corresponds to 255 and we linearly rescale the intensities between these points, we have increased the contrast
in the image and differences in detail can more clearly be seen. This is contrast stretching.

As an example, here we perform contrast stretching with a low pixel value of 100 and a high pixel value of 200 on
the complement of the grayscale image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
complement = filter_complement(grayscale)
contrast_stretch = filter_contrast_stretch(complement, low=100, high=200)
display_img(contrast_stretch, "Contrast Stretch")
```

This can be used to visually inspect details in the previous intensity range of 100 to 200, since the image filter has
spread out this range across the full spectrum.


| **Contrast Stretching Filter** |
| -------------------- |
| ![Contrast Stretching Filter](images/contrast-stretching.png "Contrast Stretching Filter") |


Here we see the console output from this set of filters.

```
RGB                  | Time: 0:00:00.207501  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.146119  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.001615  Type: uint8   Shape: (1567, 2048)
Contrast Stretch     | Time: 0:00:00.075437  Type: uint8   Shape: (1567, 2048)
```


#### Histogram Equalization

Histogram equalization is another technique that can be used to increase contrast in an image. However, unlike
contrast stretching, which has a linear distribution of the resulting intensities, the histogram equalization
transformation is based on probabilities and is non-linear. For more information about histogram equalization, please
see [https://en.wikipedia.org/wiki/Histogram_equalization](https://en.wikipedia.org/wiki/Histogram_equalization).

As an example, here we display the grayscale image. We increase contrast in the grayscale image using histogram
equalization and display the resulting image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
display_img(grayscale, "Grayscale")
hist_equ = filter_histogram_equalization(grayscale)
display_img(hist_equ, "Histogram Equalization")
```

Comparing the grayscale image and the image after histogram equalization, we see that contrast in the image has been
increased.

| **Grayscale Filter** | **Histogram Equalization Filter** |
| -------------------- | --------------------------------- |
| ![Grayscale Filter](images/grayscale.png "Grayscale Filter") | ![Histogram Equalization Filter](images/histogram-equalization.png "Histogram Equalization Filter") |


Console output following histogram equalization is shown here.

```
RGB                  | Time: 0:00:00.201479  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.129065  Type: uint8   Shape: (1567, 2048)
Hist Equalization    | Time: 0:00:00.152975  Type: uint8   Shape: (1567, 2048)
```


#### Adaptive Equalization

Rather than applying a single transformation to all pixels in an image, adaptive histogram equalization applies
transformations to local regions in an image. As a result, adaptive equalization allows contrast to be enhanced to
different extents in different regions based on the regions' histograms. For more information about adaptive
equalization, please see
[https://en.wikipedia.org/wiki/Adaptive_histogram_equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization).

The `filter_adaptive_equalization()` function utilizes the scikit-image contrast limited adaptive histogram
equalization (CLAHE) implementation. Below, we apply adaptive equalization to the grayscale image and display both
the grayscale image and the image after adaptive equalization for comparison.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
grayscale = filter_rgb_to_grayscale(rgb)
display_img(grayscale, "Grayscale")
adaptive_equ = filter_adaptive_equalization(grayscale)
display_img(adaptive_equ, "Adaptive Equalization")
```

| **Grayscale Filter** | **Adaptive Equalization Filter** |
| -------------------- | --------------------------------- |
| ![Grayscale Filter](images/grayscale.png "Grayscale Filter") | ![Adaptive Equalization Filter](images/adaptive-equalization.png "Adaptive Equalization Filter") |


In the console output, we can see that adaptive equalization is fairly compute-intensive compared with other filters
that we have looked at so far.

```
RGB                  | Time: 0:00:00.209276  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.140665  Type: uint8   Shape: (1567, 2048)
Adapt Equalization   | Time: 0:00:00.278476  Type: uint8   Shape: (1567, 2048)
```


### Color

The WSI tissue samples in the training dataset have been H&E stained. Eosin stains basic structures such as
most cytoplasm proteins with a pink tone. Hematoxylin stains acidic structures such as DNA and RNA with a purple
tone. This means that cells tend to be stained pink, and particular areas of the cells such as the nuclei tend to be
stained purple. However, note that appearance can vary greatly based on the types of cells that are stained and the
amounts of stain applied.

As an example of staining differences, below we see a slide that has pink and purple staining next to another slide
where all tissue appears purple.

| **Pink and Purple Slide** | **Purple Slide** |
| -------------------- | --------------------------------- |
| ![Pink and Purple Slide](images/pink-and-purple-slide.png "Pink and Purple Slide") | ![Purple Slide](images/purple-slide.png "Purple Slide") |


Another factor regarding color is that many slides have been marked with red, green, and blue pens. Whereas in general
we would like our filters to include pink and purple colors, since these typically indicate stained tissue, we would
like our filters to exclude red, green, and blue colors, since these typically indicate pen marks on the slides which
are not tissue.

Below, we see an example of a slide that has been marked with red pen and some green pen.

| **Slide Marked with Red and Green Pen** |
| -------------------- |
| ![Slide Marked with Red and Green Pen](images/slide-pen.png "Slide Marked with Red and Green Pen") |


Developing color filters that can be used to filter tissue areas can be fairly challenging for a variety of reasons,
including:

1. Filters need to be general enough to work across all slides in the dataset.
2. Filters should handle issues such as variations in shadows and lighting.
3. Amount of H&E (purple and pink) staining can vary greatly from slide to slide.
4. Pen marks colors (red, green, and blue) vary due to issues such as lighting and pen marks over tissue.
5. There can be color overlap between stained tissue and pen marks, so we need to balance how aggressively stain
colors are inclusively filtered and how pen colors are exclusively filtered.


#### RGB to HED

The scikit-image `skimage.color` package features an `rgb2hed()` function that performs color deconvolution on the
original RGB image to create HED (Hematoxylin, Eosin, Diaminobenzidine) channels. The `filter_rgb_to_hed()` function
encapsulates `rgb2hed()`. The `filter_hed_to_hematoxylin()` and `filter_hed_to_eosin()` functions read the hematoxylin
and eosin channels and rescale the resulting 2-dimensional NumPy arrays (for example, 0 to 255 for `uint8`)
to increase contrast.

Here, we'll convert the RGB image to an HED image. We'll then obtain the hematoxylin and eosin channels and display
the resulting images.

```
img_path = slide.get_training_image_path(4)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
hed = filter_rgb_to_hed(rgb)
hema = filter_hed_to_hematoxylin(hed)
display_img(hema, "Hematoxylin Channel")
eosin = filter_hed_to_eosin(hed)
display_img(eosin, "Eosin Channel")
```

Notice that the hematoxylin channel does fairly well at detecting the purple areas of the original slide,
which could potentially be used to narrow in on tissue with cell nuclei and thus on regions that can be inspected for
mitoses. Both the hematoxylin and eosin channel filters include the background in the resulting image, which is
rather unfortunate in terms of differentiating tissue from non-tissue. Also, notice in the eosin channel that the red
pen is considered to be part of the eosin stain spectrum.


| **Hematoxylin Channel** | **Eosin Channel** |
| -------------------- | --------------------------------- |
| ![Hematoxylin Channel](images/hematoxylin-channel.png "Hematoxylin Channel") | ![Eosin Channel](images/eosin-channel.png "Eosin Channel") |


Console output:

```
RGB                  | Time: 0:00:00.185915  Type: uint8   Shape: (1804, 2048, 3)
RGB to HED           | Time: 0:00:00.515751  Type: uint8   Shape: (1804, 2048, 3)
HED to Hematoxylin   | Time: 0:00:00.063843  Type: uint8   Shape: (1804, 2048)
HED to Eosin         | Time: 0:00:00.042430  Type: uint8   Shape: (1804, 2048)
```


#### Green Channel Filter

If we look at a color wheel, we see that purple and pink are next to each other. On the other side of color wheel, we
have yellow and green. Since green is one of our 3 NumPy array RGB color channels, filtering out pixels that have a high
green channel value can be one way to potentially filter out parts of the slide that are not pink or purple. This
includes the white background, since white also has a high green channel value along with high red and blue channel
values.

We'll use the default green threshold value of 200 for the `filter_green_channel()` function, meaning that any pixels
with green channel values of 200 or greater will be rejected.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
not_green = filter_green_channel(rgb)
display_img(not_green, "Green Channel Filter")
```

The green channel filter does a decent job of differentiating the tissue from the white background. However, notice
that the shadow area at the top of the slide is not excluded by the filter.

| **Original Slide** | **Green Channel Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/display-image-with-text.png "Original Slide") | ![Green Channel Filter](images/green-channel-filter.png "Green Channel Filter") |

A filter such as the green channel filter most likely would be used in conjunction with other filters for masking
purposes. As a result, the default output type for the green channel filter is `bool`, as we see in the console
output. If another output type is desired, this can be set with the function's `output_type` parameter.

```
RGB                  | Time: 0:00:00.210066  Type: uint8   Shape: (1567, 2048, 3)
Filter Green Channel | Time: 0:00:00.027842  Type: bool    Shape: (1567, 2048)
```


#### Grays Filter

Next, let's utilize a filter that can filter out the annoying shadow area at the top of slide #2. Notice that the
shadow area consists of a gradient of dark-to-light grays. A gray pixel has red, green, and blue channel values that
are close together. The `filter_grays()` function will filter out pixels that have red, blue, and green values that
are within a certain tolerance of each other. The default tolerance for `filter_grays()` is 15. The grays filter will
also filter out white and black pixels, since they have similar red, green, and blue values.

Here, we run the grays filter on the original RGB image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
not_grays = filter_grays(rgb)
display_img(not_grays, "Grays Filter")
```

Notice that in addition to filtering out the white background, the grays filter has indeed filtered out the shadow
area at the top of the slide.

| **Original Slide** | **Grays Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/display-image-with-text.png "Original Slide") | ![Grays Filter](images/grays-filter.png "Grays Filter") |

Like the green channel filter, the default type of the returned array is `bool` since the grays filter will typically
be used in combination with other filters. Since the grays filter is fast, it offers a potentially
low-cost way to filter out shadows from the slides during preprocessing.

```
RGB                  | Time: 0:00:00.219749  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.091341  Type: bool    Shape: (1567, 2048)
```


#### Red Filter

Next, let's turn our attention to filtering out shades of red, which can be used to filter out the red pen color.
The red pen consists of a wide variety of closely related red shades. Certain shades are
reddish, others are maroonish, and others are pinkish, for example. These color gradations are a result of a variety of
factors, such as the amount of ink, lighting, shadowing, and tissue under the pen marks.

The `filter_red()` function filters out reddish colors through a red channel lower threshold value, a green channel
upper threshold value, and a blue channel upper threshold value. The generated mask is based on a pixel being above
the red channel threshold value and below the green and blue channel threshold values. One way to determine these
values is to display the slide image in a web browser and use a tool such as the Chrome ColorPick Eyedropper to
click on a red pen pixel to determine the approximate red, green, and blue values.

In this example with slide #4, we'll use a red threshold value of 150, a green threshold value of 80, and a blue
threshold value of 90. In addition, to help us visualize the filter results, we will apply the red filter to the
original RGB image as a mask, and we will also apply the inverse of the red filter to the original image as a mask.

```
img_path = slide.get_training_image_path(4)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
not_red = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90, display_np_info=True)
display_img(not_red, "Red Filter (150, 80, 90)")
display_img(mask_rgb(rgb, not_red), "Not Red")
display_img(mask_rgb(rgb, ~not_red), "Red")
```

In the generated image, we can see that much of the red pen has been filtered out.

| **Original Slide** | **Red Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/slide-pen.png "Original Slide") | ![Red Filter](images/red-filter.png "Red Filter") |


Applying the red filter and the inverse of the red filter as masks to the original image, we see that our threshold
values did quite well at filtering out a large amount of the red pen.

| **Not Red** | **Red** |
| -------------------- | --------------------------------- |
| ![Not Red](images/not-red.png "Not Red") | ![Red](images/red.png "Red") |


Console output from the above image generation:

```
RGB                  | Time: 0:00:00.182861  Type: uint8   Shape: (1804, 2048, 3)
Filter Red           | Time: 0:00:00.013861  Type: bool    Shape: (1804, 2048)
Mask RGB             | Time: 0:00:00.015051  Type: uint8   Shape: (1804, 2048, 3)
Mask RGB             | Time: 0:00:00.018000  Type: uint8   Shape: (1804, 2048, 3)
```


#### Red Pen Filter

Next, let's turn our attention to a more inclusive red pen filter that handles more shades of red. Since the
`filter_red()` function returns a boolean array result, we can easily combine multiple sets of `filter_red()` threshold
values (`red_lower_thresh`, `green_upper_thresh`, `blue_upper_thresh`) using boolean operators such as &. We can
determine these values using a color picker tool such as the Chrome ColorPick Eyedropper, as mentioned previously.
In addition to determining various shades of red pen on a single slide, shades of red pen from other slides should be
identified and included. Note that we need to be careful with pinkish shades of red due to the similarity of these
shades to eosin staining.

Using the color picker technique, the `filter_red_pen()` function utilizes the following sets of red threshold values.

```
result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
         filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
         filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
         filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
         filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
         filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
         filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
         filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
         filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
```

Let's apply the red pen filter to slide #4.

```
img_path = slide.get_training_image_path(4)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
not_red_pen = filter_red_pen(rgb)
display_img(not_red_pen, "Red Pen Filter")
display_img(mask_rgb(rgb, not_red_pen), "Not Red Pen")
display_img(mask_rgb(rgb, ~not_red_pen), "Red Pen")
```

| **Original Slide** | **Red Pen Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/slide-pen.png "Original Slide") | ![Red Pen Filter](images/red-pen-filter.png "Red Pen Filter") |

Compared with using a single set of red threshold values, we can see that the red pen filter is significantly
more inclusive in terms of the shades of red that are accepted. As a result, more red pen is filtered. However, notice
that some of the pinkish-red from eosin-stained tissue is also included as a result of this more aggressive filtering.


| **Not Red Pen** | **Red Pen** |
| -------------------- | --------------------------------- |
| ![Not Red Pen](images/not-red-pen.png "Not Red Pen") | ![Red Pen](images/red-pen.png "Red Pen") |


Even though the red pen filter ANDs nine sets of red filter results together, we see that the performance is excellent.

```
RGB                  | Time: 0:00:00.192713  Type: uint8   Shape: (1804, 2048, 3)
Filter Red Pen       | Time: 0:00:00.113712  Type: bool    Shape: (1804, 2048)
Mask RGB             | Time: 0:00:00.014232  Type: uint8   Shape: (1804, 2048, 3)
Mask RGB             | Time: 0:00:00.019559  Type: uint8   Shape: (1804, 2048, 3)
```

#### Blue Filter

If we visually examine the 500 slides in the training dataset, we see that several of the slides have been marked
with blue pen. Rather than blue lines, many of the blue marks consist of blue dots surrounding particular areas of
interest on the slides, although this is not always the case. Some of the slides also have blue pen lines. Once again,
the blue pen marks consist of several gradations of blue.

We'll start by creating a filter to filter out blue. The `filter_blue()` function operates in a similar way as the
`filter_red()` function. It takes a red channel upper threshold value, a green channel upper threshold value, and
a blue channel lower threshold value. The generated mask is based on a pixel being below the red channel threshold
value, below the green channel threshold value, and above the blue channel threshold value.

Once again, we'll also apply the results of the blue filter and the inverse of the blue filter as masks to the original
RGB image to help visualize the filter results.

```
img_path = slide.get_training_image_path(241)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
not_blue = filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180, display_np_info=True)
display_img(not_blue, "Blue Filter (130, 155, 180)")
display_img(mask_rgb(rgb, not_blue), "Not Blue")
display_img(mask_rgb(rgb, ~not_blue), "Blue")
```

We see that a lot of the blue pen has been filtered out.

| **Original Slide** | **Blue Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/blue-original.png "Original Slide") | ![Blue Filter](images/blue-filter.png "Blue Filter") |


| **Not Blue** | **Blue** |
| -------------------- | --------------------------------- |
| ![Not Blue](images/not-blue.png "Not Blue") | ![Blue](images/blue.png "Blue") |


Console output:

```
RGB                  | Time: 0:00:00.142685  Type: uint8   Shape: (1301, 2048, 3)
Filter Blue          | Time: 0:00:00.011197  Type: bool    Shape: (1301, 2048)
Mask RGB             | Time: 0:00:00.010248  Type: uint8   Shape: (1301, 2048, 3)
Mask RGB             | Time: 0:00:00.009995  Type: uint8   Shape: (1301, 2048, 3)
```


#### Blue Pen Filter

In `filter_blue_pen()`, we can AND together various blue shade ranges using `filter_blue()` with
sets of red, green, and blue threshold values to create a blue pen filter that filters out various shades of blue.

```
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
```

Once again, we'll apply the filter and its inverse to the original slide to help us visualize the results.

```
img_path = slide.get_training_image_path(241)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
not_blue_pen = filter_blue_pen(rgb)
display_img(not_blue_pen, "Blue Pen Filter")
display_img(mask_rgb(rgb, not_blue_pen), "Not Blue Pen")
display_img(mask_rgb(rgb, ~not_blue_pen), "Blue Pen")
```

For this slide, we see that `filter_blue_pen()` filters out more blue than the previous `filter_blue()` example.

| **Original Slide** | **Blue Pen Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/blue-original.png "Original Slide") | ![Blue Pen Filter](images/blue-pen-filter.png "Blue Pen Filter") |


| **Not Blue Pen** | **Blue Pen** |
| -------------------- | --------------------------------- |
| ![Not Blue Pen](images/not-blue-pen.png "Not Blue Pen") | ![Blue Pen](images/blue-pen.png "Blue Pen") |


We see from the console output that the blue pen filter is quite fast.

```
RGB                  | Time: 0:00:00.146905  Type: uint8   Shape: (1301, 2048, 3)
Filter Blue Pen      | Time: 0:00:00.134946  Type: bool    Shape: (1301, 2048)
Mask RGB             | Time: 0:00:00.010695  Type: uint8   Shape: (1301, 2048, 3)
Mask RGB             | Time: 0:00:00.005944  Type: uint8   Shape: (1301, 2048, 3)
```

As an aside, we can easily quantify the differences in filtering between the `filter_blue()` and `filter_blue_pen()`
results.

```
not_blue = filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180, display_np_info=True)
not_blue_pen = filter_blue_pen(rgb)
print("filter_blue:" + mask_percentage_text(mask_percent(not_blue)))
print("filter_blue_pen:" + mask_percentage_text(mask_percent(not_blue_pen)))
```

The `filter_blue()` example filtered out 0.45% of the slide pixels and the `filter_blue_pen()` example filtered out
0.68% of the slide pixels.

```
filter_blue:
(0.45% masked)
filter_blue_pen:
(0.68% masked)
```

#### Green Filter

To develop a filter for the green ink from the pen, we'll create a `filter_green()` function to handle the green
color shades. Using a color picker tool, if we examine the green pen marks on the slides, the green and blue channel
values for pixels appear to track together. As a result of this, this function will have a red channel upper
threshold value, a green channel lower threshold value, and a blue channel lower threshold value.


```
img_path = slide.get_training_image_path(51)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
not_green = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140, display_np_info=True)
display_img(not_green, "Green Filter (150, 160, 140)")
display_img(mask_rgb(rgb, not_green), "Not Green")
display_img(mask_rgb(rgb, ~not_green), "Green")
```

Using a red upper threshold of 150, a green lower threshold of 160, and a blue lower threshold of 140, we see that the
much of the green ink above the background is filtered out, but most of the green ink above the tissue is not filtered
out.

| **Original Slide** | **Green Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/green-original.png "Original Slide") | ![Green Filter](images/green-filter.png "Green Filter") |


| **Not Green** | **Green** |
| -------------------- | --------------------------------- |
| ![Not Green](images/not-green.png "Not Green") | ![Green](images/green.png "Green") |


Console output:

```
RGB                  | Time: 0:00:00.150075  Type: uint8   Shape: (1222, 2048, 3)
Filter Green         | Time: 0:00:00.010120  Type: bool    Shape: (1222, 2048)
Mask RGB             | Time: 0:00:00.009844  Type: uint8   Shape: (1222, 2048, 3)
Mask RGB             | Time: 0:00:00.006240  Type: uint8   Shape: (1222, 2048, 3)
```

#### Green Pen Filter

To handle the green pen shades, the `filter_green_pen()` function combines different shade results using sets of
red, green, and blue threshold values passed to the `filter_green()` function.

```
result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
         filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
         filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
         filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
         filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
         filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
         filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
         filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
         filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
         filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
         filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
         filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
         filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
         filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
         filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
```

If we apply the green pen filter, we see that it includes most of the green shades above the tissue in slide 51.

```
img_path = slide.get_training_image_path(51)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
not_green_pen = filter_green_pen(rgb)
display_img(not_green_pen, "Green Pen Filter")
display_img(mask_rgb(rgb, not_green_pen), "Not Green Pen")
display_img(mask_rgb(rgb, ~not_green_pen), "Green Pen")
```

| **Original Slide** | **Green Pen Filter** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/green-original.png "Original Slide") | ![Green Pen Filter](images/green-pen-filter.png "Green Pen Filter") |


| **Not Green Pen** | **Green Pen** |
| -------------------- | --------------------------------- |
| ![Not Green Pen](images/not-green-pen.png "Not Green Pen") | ![Green Pen](images/green-pen.png "Green Pen") |


Like the other pen filters, the green pen filter's performance is quite good.

```
RGB                  | Time: 0:00:00.165038  Type: uint8   Shape: (1222, 2048, 3)
Filter Green Pen     | Time: 0:00:00.118797  Type: bool    Shape: (1222, 2048)
Mask RGB             | Time: 0:00:00.011132  Type: uint8   Shape: (1222, 2048, 3)
Mask RGB             | Time: 0:00:00.005561  Type: uint8   Shape: (1222, 2048, 3)
```


#### K-Means Segmentation

The scikit-image library contains functionality that allows for image segmentation using k-means clustering based
on location and color. This allows regions of similarly colored pixels to be grouped together. These regions are
colored based on the average color of the pixels in the individual regions. This could potentially be used to filter
regions based on their colors, where we could filter on pink shades for eosin-stained tissue and purple shades for
hematoxylin-stained tissue.

The `filter_kmeans_segmentation()` function has a default value of 800 segments. We'll increase this to 3000 using
the `n_segments` parameter. In the example below, we'll perform k-means segmentation on the original image. In
addition, we'll create a threshold using Otsu's method and apply the resulting mask to the original image. We'll then
perform k-means segmentation on that image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
kmeans_seg = filter_kmeans_segmentation(rgb, n_segments=3000)
display_img(kmeans_seg, "K-Means Segmentation", color=(0, 0, 0))
otsu_mask = mask_rgb(rgb, filter_otsu_threshold(filter_complement(filter_rgb_to_grayscale(rgb)), output_type="bool"))
display_img(otsu_mask, "Image after Otsu Mask", color=(255, 255, 255))
kmeans_seg_otsu = filter_kmeans_segmentation(otsu_mask, n_segments=3000)
display_img(kmeans_seg_otsu, "K-Means Segmentation after Otsu Mask", color=(255, 255, 255))
```


| **Original Slide** | **K-Means Segmentation** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/kmeans-original.png "Original Slide") | ![K-Means Segmentation](images/kmeans-segmentation.png "K-Means Segmentation") |


| **Image after Otsu Mask** | **K-Means Segmentation after Otsu Mask** |
| -------------------- | --------------------------------- |
| ![Image after Otsu Mask](images/otsu-mask.png "Image after Otsu Mask") | ![K-Means Segmentation after Otsu Mask](images/kmeans-segmentation-after-otsu.png "K-Means Segmentation after Otsu Mask") |


Note that there are a couple practical difficulties in terms of implementing automated tissue detection using k-means
segmentation. To begin with, due to the variation in tissue stain colors across the image dataset, it can be difficult
to filter on "pinkish" and "purplish" colors across all the slides. In addition, the k-means segmentation technique
is very computationally expensive, as we can see in the console output. The compute time increases with the number
of segments. For 3000 segments, we have a filter time of 26 seconds, whereas all operations that we have seen up to
this point are subsecond. If we use the default value of 800 segments, compute time for the k-means segmentation filter
is 9 seconds.

```
RGB                  | Time: 0:00:00.206051  Type: uint8   Shape: (1567, 2048, 3)
K-Means Segmentation | Time: 0:00:26.106364  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.109594  Type: uint8   Shape: (1567, 2048)
Complement           | Time: 0:00:00.000841  Type: uint8   Shape: (1567, 2048)
Otsu Threshold       | Time: 0:00:00.016523  Type: bool    Shape: (1567, 2048)
Mask RGB             | Time: 0:00:00.009480  Type: uint8   Shape: (1567, 2048, 3)
K-Means Segmentation | Time: 0:00:26.232028  Type: uint8   Shape: (1567, 2048, 3)
```

---

The sci-kit image library also makes it possible to combine similarly colored regions. One way to do this with the
k-means segmentation results is to build a region adjacency graph (RAG) and combine regions based on a threshold value.
The `filter_rag_threshold()` function performs k-means segmentation, builds the RAG, and allows us to pass in the RAG
threshold value.

Here, we perform k-means segmentation, build a RAG, and apply different RAG thresholds to combine similar regions.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
rag_thresh = filter_rag_threshold(rgb)
display_img(rag_thresh, "RAG Threshold (9)")
rag_thresh = filter_rag_threshold(rgb, threshold=1)
display_img(rag_thresh, "RAG Threshold (1)")
rag_thresh = filter_rag_threshold(rgb, threshold=20)
display_img(rag_thresh, "RAG Threshold (20)")
```

| **Original Slide** | **RAG Threshold = 9** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/rag-thresh-original.png "Original Slide") | ![RAG Threshold = 9](images/rag-thresh-9.png "RAG Threshold = 9") |


| **RAG Threshold = 1** | **RAG Threshold = 20** |
| -------------------- | --------------------------------- |
| ![RAG Threshold = 1](images/rag-thresh-1.png "RAG Threshold = 1") | ![RAG Threshold = 20](images/rag-thresh-20.png "RAG Threshold = 20") |


Even using the default 800 number of segments for the k-means segmentation, we see that this technique is very
computationally expensive.

```
RGB                  | Time: 0:00:00.202184  Type: uint8   Shape: (1567, 2048, 3)
RAG Threshold        | Time: 0:00:30.311410  Type: uint8   Shape: (1567, 2048, 3)
RAG Threshold        | Time: 0:00:32.604226  Type: uint8   Shape: (1567, 2048, 3)
RAG Threshold        | Time: 0:00:28.637301  Type: uint8   Shape: (1567, 2048, 3)
```


### Morphology

Information about the field of morphology applied to images can be found at
[https://en.wikipedia.org/wiki/Mathematical_morphology](https://en.wikipedia.org/wiki/Mathematical_morphology).
The primary morphology operators are erosion, dilation, opening, and closing. With erosion, pixels along the edges
of an object are removed. For dilation, pixels along the edges of an object are added. Opening is erosion followed
by dilation. Closing is dilation followed by erosion. With morphology operators, a structuring element (such as
a square, circle, cross, etc) is passed along the edges of the objects to perform the operations. Morphology operators
can be performed on binary and grayscale images. In our examples, we will apply morphology operators to binary images
(2-dimensional arrays of 2 values, such as True/False, 1.0/0.0, and 255/0).


#### Erosion

We create a binary image mask by calling the `filter_grays()` function on the original RGB image. The
`filter_binary_erosion()` function uses a disk as the structuring element that will be used to erode the edges of the
"No Grays" binary image. We demonstrate erosion with disk structuring elements of radius 5 and radius 20.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
bin_erosion_5 = filter_binary_erosion(no_grays, disk_size=5)
display_img(bin_erosion_5, "Binary Erosion (5)")
bin_erosion_20 = filter_binary_erosion(no_grays, disk_size=20)
display_img(bin_erosion_20, "Binary Erosion (20)")
```

| **Original Slide** | **No Grays** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/binary-erosion-original.png "Original Slide") | ![No Grays](images/binary-erosion-no-grays.png "No Grays") |


| **Binary Erosion (disk_size = 5)** | **Binary Erosion (disk_size = 20)** |
| -------------------- | --------------------------------- |
| ![Binary Erosion (disk_size = 5)](images/binary-erosion-5.png "Binary Erosion (disk_size = 5)") | ![Binary Erosion (disk_size = 20)](images/binary-erosion-20.png "Binary Erosion (disk_size = 20)") |


Notice that increasing the structuring element radius increases the compute time.

```
RGB                  | Time: 0:00:00.208153  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.096831  Type: bool    Shape: (1567, 2048)
Binary Erosion       | Time: 0:00:00.156416  Type: uint8   Shape: (1567, 2048)
Binary Erosion       | Time: 0:00:00.900778  Type: uint8   Shape: (1567, 2048)
```


#### Dilation

The `filter_binary_dilation()` function utilizes a disk structuring element in a similar manner as the corresponding
erosion function. We'll utilize the same "No Grays" binary image from the previous example and dilate the image
utilizing a disk radius of 5 pixels followed by a disk radius of 20 pixels.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
bin_dilation_5 = filter_binary_dilation(no_grays, disk_size=5)
display_img(bin_dilation_5, "Binary Dilation (5)")
bin_dilation_20 = filter_binary_dilation(no_grays, disk_size=20)
display_img(bin_dilation_20, "Binary Dilation (20)")
```

We see that dilation expands the edges of the binary image as opposed to the erosion, which shrinks the edges.

| **Binary Dilation (disk_size = 5)** | **Binary Dilation (disk_size = 20)** |
| -------------------- | --------------------------------- |
| ![Binary Dilation (disk_size = 5)](images/binary-dilation-5.png "Binary Dilation (disk_size = 5)") | ![Binary Dilation (disk_size = 20)](images/binary-dilation-20.png "Binary Dilation (disk_size = 20)") |


Console output:

```
RGB                  | Time: 0:00:00.203026  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.096591  Type: bool    Shape: (1567, 2048)
Binary Dilation      | Time: 0:00:00.084602  Type: uint8   Shape: (1567, 2048)
Binary Dilation      | Time: 0:00:00.689930  Type: uint8   Shape: (1567, 2048)
```


#### Opening

As mentioned, opening is erosion followed by dilation. Opening can be used to remove small foreground objects.


```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
bin_opening_5 = filter_binary_opening(no_grays, disk_size=5)
display_img(bin_opening_5, "Binary Opening (5)")
bin_opening_20 = filter_binary_opening(no_grays, disk_size=20)
display_img(bin_opening_20, "Binary Opening (20)")
```

| **Binary Opening (disk_size = 5)** | **Binary Opening (disk_size = 20)** |
| -------------------- | --------------------------------- |
| ![Binary Opening (disk_size = 5)](images/binary-opening-5.png "Binary Opening (disk_size = 5)") | ![Binary Opening (disk_size = 20)](images/binary-opening-20.png "Binary Opening (disk_size = 20)") |


Opening is a fairly expensive operation, since it is an erosion followed by a dilation. The compute time increases
with the size of the structuring element. The 5-pixel disk radius for the structuring element results in a 0.3s
operation, whereas the 20-pixel disk radius results in a 3 second operation.

```
RGB                  | Time: 0:00:00.207179  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.099212  Type: bool    Shape: (1567, 2048)
Binary Opening       | Time: 0:00:00.299361  Type: uint8   Shape: (1567, 2048)
Binary Opening       | Time: 0:00:03.019376  Type: uint8   Shape: (1567, 2048)
```


#### Closing

Closing is a dilation followed by an erosion. Closing can be used to remove small background holes.


```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
bin_closing_5 = filter_binary_closing(no_grays, disk_size=5)
display_img(bin_closing_5, "Binary Closing (5)")
bin_closing_20 = filter_binary_closing(no_grays, disk_size=20)
display_img(bin_closing_20, "Binary Closing (20)")
```

| **Binary Closing (disk_size = 5)** | **Binary Closing (disk_size = 20)** |
| -------------------- | --------------------------------- |
| ![Binary Closing (disk_size = 5)](images/binary-closing-5.png "Binary Closing (disk_size = 5)") | ![Binary Closing (disk_size = 20)](images/binary-closing-20.png "Binary Closing (disk_size = 20)") |


Like opening, closing is a fairly expensive operation since it performs both a dilation and an erosion. Compute time
increases with structuring element size.

```
RGB                  | Time: 0:00:00.204637  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.099459  Type: bool    Shape: (1567, 2048)
Binary Closing       | Time: 0:00:00.309861  Type: uint8   Shape: (1567, 2048)
Binary Closing       | Time: 0:00:03.165193  Type: uint8   Shape: (1567, 2048)
```


#### Remove Small Objects

The scikit-image `remove_small_objects()` function removes objects less than a particular minimum size. The
`filter_remove_small_objects()` function wraps this and adds additional functionality. This can be useful for
removing small islands of noise from images. We'll demonstrate it here with two sizes, 100 pixels and 10,000 pixels,
and we'll perform this on the "No Grays" binary image.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
remove_small_100 = filter_remove_small_objects(no_grays, min_size=100)
display_img(remove_small_100, "Remove Small Objects (100)")
remove_small_10000 = filter_remove_small_objects(no_grays, min_size=10000)
display_img(remove_small_10000, "Remove Small Objects (10000)")
```

Notice in the "No Grays" mask that we see lots of scattered, small objects.

| **Original Slide** | **No Grays** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/binary-erosion-original.png "Original Slide") | ![No Grays](images/binary-erosion-no-grays.png "No Grays") |


After removing small objects with a connected size less than 100 pixels, we see that the smallest objects have been
removed from the binary image. With a minimum size of 10,000 pixels, we see that many larger objects have also been
removed from the binary image.

| **Remove Small Objects (100)** | **Remove Small Objects (10000)** |
| -------------------- | --------------------------------- |
| ![Remove Small Objects (100)](images/remove-small-objects-100.png "Remove Small Objects (100)") | ![Remove Small Objects (10000)](images/remove-small-objects-10000.png "Remove Small Objects (10000)") |


The performance of the filters to remove small objects is quite fast.

```
RGB                  | Time: 0:00:00.208705  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.100657  Type: bool    Shape: (1567, 2048)
Remove Small Objs    | Time: 0:00:00.055258  Type: uint8   Shape: (1567, 2048)
Remove Small Objs    | Time: 0:00:00.056445  Type: uint8   Shape: (1567, 2048)
```


#### Remove Small Holes

The scikit-image `remove_small_holes()` function is similar to the `remove_small_objects()` function except it removes
holes rather than objects from binary images. Here we demonstrate this using the `filter_remove_small_holes()`
wrapper with sizes of 100 pixels and 10,000 pixels.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
remove_small_100 = filter_remove_small_holes(no_grays, min_size=100)
display_img(remove_small_100, "Remove Small Holes (100)")
remove_small_10000 = filter_remove_small_holes(no_grays, min_size=10000)
display_img(remove_small_10000, "Remove Small Holes (10000)")
```

Notice that using a minimum size of 10,000 removes more holes than a size of 100, as we would expect.

| **Remove Small Holes (100)** | **Remove Small Holes (10000)** |
| -------------------- | --------------------------------- |
| ![Remove Small Holes (100)](images/remove-small-holes-100.png "Remove Small Holes (100)") | ![Remove Small Holes (10000)](images/remove-small-holes-10000.png "Remove Small Holes (10000)") |


Console output:

```
RGB                  | Time: 0:00:00.210587  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.105123  Type: bool    Shape: (1567, 2048)
Remove Small Holes   | Time: 0:00:00.055991  Type: uint8   Shape: (1567, 2048)
Remove Small Holes   | Time: 0:00:00.058003  Type: uint8   Shape: (1567, 2048)
```


#### Fill Holes

The scikit-image `binary_fill_holes()` function is similar to the `remove_small_holes()` function. Using its default
settings, it generates results similar but typically not identical to `remove_small_holes()` with a high minimum
size value.

Here, we'll display the result of `filter_binary_fill_holes()` on the image after gray shades have been removed. After
this, we'll perform exclusive-or operations to look at the differences between "Fill Holes" and "Remove Small Holes"
with size values of 100 and 10,000.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_grays = filter_grays(rgb, output_type="bool")
display_img(no_grays, "No Grays")
fill_holes = filter_binary_fill_holes(no_grays)
display_img(fill_holes, "Fill Holes")

remove_holes_100 = filter_remove_small_holes(no_grays, min_size=100, output_type="bool")
display_img(fill_holes ^ remove_holes_100, "Differences between Fill Holes and Remove Small Holes (100)")

remove_holes_10000 = filter_remove_small_holes(no_grays, min_size=10000, output_type="bool")
display_img(fill_holes ^ remove_holes_10000, "Differences between Fill Holes and Remove Small Holes (10000)")

```

| **Original Slide** | **Fill Holes** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/binary-erosion-original.png "Original Slide") | ![Fill Holes](images/fill-holes.png "Fill Holes") |


In this example, increasing the minimum small hole size results in less differences between "Fill Holes" and
"Remove Small Holes".

| **Differences between Fill Holes and Remove Small Holes (100)** | **Differences between Fill Holes and Remove Small Holes (10000)** |
| -------------------- | --------------------------------- |
| ![Differences between Fill Holes and Remove Small Holes (100)](images/fill-holes-remove-small-holes-100.png "Differences between Fill Holes and Remove Small Holes (100)") | ![Differences between Fill Holes and Remove Small Holes (10000)](images/fill-holes-remove-small-holes-10000.png "Differences between Fill Holes and Remove Small Holes (10000)") |


Console output:

```
RGB                  | Time: 0:00:00.208171  Type: uint8   Shape: (1567, 2048, 3)
Filter Grays         | Time: 0:00:00.108048  Type: bool    Shape: (1567, 2048)
Binary Fill Holes    | Time: 0:00:00.088519  Type: bool    Shape: (1567, 2048)
Remove Small Holes   | Time: 0:00:00.054506  Type: bool    Shape: (1567, 2048)
Remove Small Holes   | Time: 0:00:00.056771  Type: bool    Shape: (1567, 2048)
```


### Entropy

The scikit-image `entropy()` function allows us to filter images based on complexity. Since areas such as slide
backgrounds are less complex than area of interest such as cell nuclei, filtering on entropy offers interesting
possibilities for tissue identification.

Here, we use the `filter_entropy()` function to filter the grayscale image based on entropy. We display
the resulting binary image. After that, we mask the original image with the entropy mask and the inverse of entropy
mask.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
gray = filter_rgb_to_grayscale(rgb)
display_img(gray, "Grayscale")
entropy = filter_entropy(gray, output_type="bool")
display_img(entropy, "Entropy")
display_img(mask_rgb(rgb, entropy), "Original with Entropy Mask")
display_img(mask_rgb(rgb, ~entropy), "Original with Inverse of Entropy Mask")
```

| **Original Slide** | **Grayscale** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/entropy-original.png "Original Slide") | ![Grayscale](images/entropy-grayscale.png "Grayscale") |


| **Entropy Filter** |
| ------------------ |
| ![Entropy Filter](images/entropy.png "Entropy Filter") |


The results of the original image with the inverse of the entropy mask are particularly interesting. Notice that much
of the white background including the shadow region at the top of the slide has been filtered out. Additionally, notice
that for the stained regions, a significant amount of the pink eosin-stained area has been filtered out while a
smaller proportion of the purple-stained hemotoxylin area has been filtered out. This makes sense since hemotoxylin
stains regions such as cell nuclei, which are structures with significant complexity. Therefore, complexity seems
like a good candidate for identifying regions of interest where mitoses are occurring.


| **Original with Entropy Mask** | **Original with Inverse of Entropy Mask** |
| -------------------- | --------------------------------- |
| ![Original with Entropy Mask](images/entropy-original-entropy-mask.png "Original with Entropy Mask") | ![Original with Inverse of Entropy Mask](images/entropy-original-inverse-entropy-mask.png "Original with Inverse of Entropy Mask") |


A drawback of using entropy is that its computation is significant. The entropy filter takes over 4 seconds to run
in this example.

```
RGB                  | Time: 0:00:00.204830  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.136988  Type: uint8   Shape: (1567, 2048)
Entropy              | Time: 0:00:04.148810  Type: bool    Shape: (1567, 2048)
Mask RGB             | Time: 0:00:00.012648  Type: uint8   Shape: (1567, 2048, 3)
Mask RGB             | Time: 0:00:00.007009  Type: uint8   Shape: (1567, 2048, 3)
```


### Canny Edge Detection

Edges in images are areas where there is typically a significant, abrupt change in image brightness.
The Canny edge detection algorithm is implemented in sci-kit image. More information about
edge detection can be found at [https://en.wikipedia.org/wiki/Edge_detection](https://en.wikipedia.org/wiki/Edge_detection).
More information about Canny edge detection can be found at
[https://en.wikipedia.org/wiki/Canny_edge_detector](https://en.wikipedia.org/wiki/Canny_edge_detector).

The sci-kit image `canny()` function returns a binary edge map for the detected edges in an input image. In the
example below, we'll call the canny edge filter on the grayscale image and display the resulting Canny edges.
After this, we'll crop a 600x600 area of the original slide and display this. We'll then apply the inverse of the
canny mask to the cropped original slide and display this for comparison.

```
img_path = slide.get_training_image_path(2)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
gray = filter_rgb_to_grayscale(rgb)
canny = filter_canny(gray, output_type="bool")
display_img(canny, "Canny", bg=True)
rgb_crop = rgb[300:900, 300:900]
canny_crop = canny[300:900, 300:900]
display_img(rgb_crop, "Original", size=24, bg=True)
display_img(mask_rgb(rgb_crop, ~canny_crop), "Original with ~Canny Mask", size=24, bg=True)
```

| **Original** | **Canny Edges** |
| -------------------- | --------------------------------- |
| ![Original](images/canny-original.png "Original") | ![Canny Edges](images/canny.png "Canny Edges") |


By applying the inverse of the canny edge mask to the original image, the detected edges are colored black. This
visually accentuates the different structures in the slide.

| **Cropped Original** | **Cropped Original with Inverse Canny Edges Mask** |
| -------------------- | --------------------------------- |
| ![Cropped Original](images/canny-original-cropped.png "Cropped Original") | ![Cropped Original with Inverse Canny Edges Mask](images/canny-original-with-inverse-mask.png "Cropped Original with Inverse Canny Edges Mask") |


In the console output, we see that Canny edge detection is fairly expensive, since its computation took over 1 second.

```
RGB                  | Time: 0:00:00.204199  Type: uint8   Shape: (1567, 2048, 3)
Gray                 | Time: 0:00:00.134988  Type: uint8   Shape: (1567, 2048)
Canny Edges          | Time: 0:00:01.237061  Type: bool    Shape: (1567, 2048)
Mask RGB             | Time: 0:00:00.000929  Type: uint8   Shape: (600, 600, 3)
```


## Combining Filters

Since our image filter data utilizes NumPy arrays, it is straightforward to combine our filters. For example, when
we have filters that return boolean images for masking, we can use standard boolean algebra on our arrays to do
operations such as AND, OR, XOR, and NOT. We can also run filters on the results of other filters.

As an example, here we run our green pen and blue pen filters on the original RGB image to filter out the green and
blue pen marks on the slides. We combine the resulting masks with a boolean AND (&) operation. We display the resulting
mask and this mask applied to the original image, masking out the green and blue pen marks from the image.

```
img_path = slide.get_training_image_path(74)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
no_green_pen = filter_green_pen(rgb)
display_img(no_green_pen, "No Green Pen")
no_blue_pen = filter_blue_pen(rgb)
display_img(no_blue_pen, "No Blue Pen")
no_gp_bp = no_green_pen & no_blue_pen
display_img(no_gp_bp, "No Green Pen, No Blue Pen")
display_img(mask_rgb(rgb, no_gp_bp), "Original with No Green Pen, No Blue Pen")
```

| **Original Slide** |
| -------------------- |
| ![Original Slide](images/combine-pen-filters-original.png "Original Slide") |

| **No Green Pen** | **No Blue Pen** |
| -------------------- | --------------------------------- |
| ![No Green Pen](images/combine-pen-filters-no-green-pen.png "No Green Pen") | ![No Blue Pen](images/combine-pen-filters-no-blue-pen.png "No Blue Pen") |

| **No Green Pen, No Blue Pen** | **Original with No Green Pen, No Blue Pen** |
| -------------------- | --------------------------------- |
| ![No Green Pen, No Blue Pen](images/combine-pen-filters-no-green-pen-no-blue-pen.png "No Green Pen, No Blue Pen") | ![Original with No Green Pen, No Blue Pen](images/combine-pen-filters-original-with-no-green-pen-no-blue-pen.png "Original with No Green Pen, No Blue Pen") |


Console Output:

```
RGB                  | Time: 0:00:00.174402  Type: uint8   Shape: (1513, 2048, 3)
Filter Green Pen     | Time: 0:00:00.199907  Type: bool    Shape: (1513, 2048)
Filter Blue Pen      | Time: 0:00:00.100322  Type: bool    Shape: (1513, 2048)
Mask RGB             | Time: 0:00:00.011881  Type: uint8   Shape: (1513, 2048, 3)
```


---

Let's try another combination of filters that should give us a fairly good tissue extraction for this slide,
where the slide background and blue and green pen marks are removed. We can do this for this slide by ANDing
together the "No Grays" filter, the "Green Channel" filter, the "No Green Pen" filter, and the "No Blue Pen" filter.
In addition, we can use our "Remove Small Objects" filter to remove small islands from the mask. We display the
resulting mask. We apply this mask and the inverse of the mask to the original image to visually see which parts of the
slide are passed through and which parts are masked out.

```
img_path = slide.get_training_image_path(74)
img = slide.open_image(img_path)
rgb = pil_to_np_rgb(img)
display_img(rgb, "Original")
mask = filter_grays(rgb) & filter_green_channel(rgb) & filter_green_pen(rgb) & filter_blue_pen(rgb)
mask = filter_remove_small_objects(mask, min_size=100, output_type="bool")
display_img(mask, "No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
display_img(mask_rgb(rgb, mask),
                     "Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
display_img(mask_rgb(rgb, ~mask), "Original with Inverse Mask")
```

| **Original Slide** | **No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects** |
| -------------------- | --------------------------------- |
| ![Original Slide](images/combine-pens-background-original.png "Original Slide") | ![No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects](images/combine-pens-background-mask.png "No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects") |


We see that this combination does a good job at allowing us to filter the most relevant tissue sections of this slide.

| **Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects** | **Original with Inverse Mask** |
| -------------------- | --------------------------------- |
| ![Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects](images/combine-pens-background-original-with-mask.png "Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects") | ![Original with Inverse Mask](images/combine-pens-background-original-with-inverse-mask.png "Original with Inverse Mask") |


Console Output:

```
RGB                  | Time: 0:00:00.167955  Type: uint8   Shape: (1513, 2048, 3)
Filter Grays         | Time: 0:00:00.109165  Type: bool    Shape: (1513, 2048)
Filter Green Channel | Time: 0:00:00.044170  Type: bool    Shape: (1513, 2048)
Filter Green Pen     | Time: 0:00:00.194573  Type: bool    Shape: (1513, 2048)
Filter Blue Pen      | Time: 0:00:00.118921  Type: bool    Shape: (1513, 2048)
Remove Small Objs    | Time: 0:00:00.042549  Type: bool    Shape: (1513, 2048)
Mask RGB             | Time: 0:00:00.011598  Type: uint8   Shape: (1513, 2048, 3)
Mask RGB             | Time: 0:00:00.007193  Type: uint8   Shape: (1513, 2048, 3)
```


---

In the `wsi/filter.py` file, the `apply_filters_to_image(slide_num, save=True, display=False)` function will be the
primary way we'll apply a set of filters to an image with the goal of identifying the tissue in the slide. This
function allows us to see the results of each filter and the combined results of different filters. If the
`save` parameter is `True`, the various filter results will be saved to the file system. If the `display`
parameter is `True`, the filter results will be displayed on the screen. The function returns a tuple consisting of
the resulting NumPy array image and a dictionary of information that is used elsewhere for generating an HTML page
to view the various filter results for multiple slides, as we will see later.

The `apply_filters_to_image()` function will create green channel, grays, red pen, green pen, and blue pen masks
and combine these into a single mask using boolean ANDs. After this, small objects will be removed from the mask.

```
mask_not_green = filter_green_channel(rgb)
mask_not_gray = filter_grays(rgb)
mask_no_red_pen = filter_red_pen(rgb)
mask_no_green_pen = filter_green_pen(rgb)
mask_no_blue_pen = filter_blue_pen(rgb)
mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
```

After each of the above masks is created, it is applied to the original image and the resulting image is saved
to the file system, displayed to the screen, or both.

Let's try this function out. In this example, we'll run `apply_filters_to_image()` on slide 337 and display the results
to the screen.

```
apply_filters_to_image(337, display=True, save=False)
```

Here, we see the original slide #337 and the green channel filter applied to it. Notice that the green channel filter
with a default threshold of 200 removes most of the white background but only a relatively small fraction of the green
pen. The green channel filter masks 72.36% of the original slide.

| **Slide 337, F001** | **Slide 337, F002** |
| -------------------- | --------------------------------- |
| ![Slide 337, F001](images/337-001.png "Slide 337, F001") | ![Slide 337, F002](images/337-002.png "Slide 337, F002") |


Here, we see the results of the grays filter and the red pen filter. For this slide, the grays filter masks 67.66% of
the slide, which is actually less than the green channel filter. The red pen filter masks only 0.03% of the slide,
which makes sense since there are no red pen marks on the slide.

| **Slide 337, F003** | **Slide 337, F004** |
| -------------------- | --------------------------------- |
| ![Slide 337, F003](images/337-003.png "Slide 337, F003") | ![Slide 337, F004](images/337-004.png "Slide 337, F004") |


The green pen filter masks 3.71% of the slide. Visually, we see that it does a decent job of masking out the green
pen marks on the slide. The blue pen filter masks 0% of the slide, which is perfect since there are no blue pen marks on
the slide.

| **Slide 337, F005** | **Slide 337, F006** |
| -------------------- | --------------------------------- |
| ![Slide 337, F005](images/337-005.png "Slide 337, F005") | ![Slide 337, F006](images/337-006.png "Slide 337, F006") |


Combining the above filters with a boolean AND results in 74.37% masking. Cleaning up these results by remove small
objects results in a masking of 75.82%. This potentially gives a good tissue sample that we can use for deep learning.

| **Slide 337, F007** | **Slide 337, F008** |
| -------------------- | --------------------------------- |
| ![Slide 337, F007](images/337-007.png "Slide 337, F007") | ![Slide 337, F008](images/337-008.png "Slide 337, F008") |


In the console, we see the slide #337 processing time takes ~5.5s in this example. The filtering is only a relatively
small fraction of this time.

```
Processing slide #337
RGB                  | Time: 0:00:00.214366  Type: uint8   Shape: (1636, 2048, 3)
Filter Green Channel | Time: 0:00:00.030129  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.013435  Type: uint8   Shape: (1636, 2048, 3)
Filter Grays         | Time: 0:00:00.100411  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.012338  Type: uint8   Shape: (1636, 2048, 3)
Filter Red Pen       | Time: 0:00:00.082352  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.012389  Type: uint8   Shape: (1636, 2048, 3)
Filter Green Pen     | Time: 0:00:00.158260  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.012019  Type: uint8   Shape: (1636, 2048, 3)
Filter Blue Pen      | Time: 0:00:00.106823  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.011877  Type: uint8   Shape: (1636, 2048, 3)
Mask RGB             | Time: 0:00:00.015465  Type: uint8   Shape: (1636, 2048, 3)
Remove Small Objs    | Time: 0:00:00.066158  Type: bool    Shape: (1636, 2048)
Mask RGB             | Time: 0:00:00.013019  Type: uint8   Shape: (1636, 2048, 3)
Slide #337 processing time: 0:00:05.485500
```

Since `apply_filters_to_image()` returns the resulting image as a NumPy array, we can perform further processing on
the image. If we look at the `apply_filters_to_image()` results for slide #337, we can see that some grayish greenish
pen marks remain on the slide. We can filter some of these out using our `filter_green()` function with different
threshold values and our `filter_grays()` function with an increased tolerance value.

We'll compare the results by cropping two regions of the slide before and after this additional processing and
displaying all four of these regions together.

```
rgb, _ = apply_filters_to_image(337, display=False, save=False)

not_greenish = filter_green(rgb, red_upper_thresh=125, green_lower_thresh=30, blue_lower_thresh=30,
                            display_np_info=True)
not_grayish = filter_grays(rgb, tolerance=30)
rgb_new = mask_rgb(rgb, not_greenish & not_grayish)

row1 = np.concatenate((rgb[800:1200, 100:500], rgb[750:1150, 1350:1750]), axis=1)
row2 = np.concatenate((rgb_new[800:1200, 100:500], rgb_new[750:1150, 1350:1750]), axis=1)
result = np.concatenate((row1, row2), axis=0)
display_img(result)
```

After the additional processing, we see that the pen marks in the displayed regions have been significantly reduced.

| **Remove More Green and More Gray** |
| -------------------- |
| ![Remove More Green and More Gray](images/remove-more-green-more-gray.png "Remove More Green and More Gray") |


## Applying Filters to Multiple Images

When designing our set of tissue-selecting filters, one very important requirement is the ability to visually inspect
the filter results across multiple slides. Ideally we should easily be able to alternate between displaying the
results for a single image, a select subset of our training image dataset, and our entire dataset. Additionally,
multiprocessing can result in a significant performance boost, so we should be able to multiprocess our image
processing if desired.

A simple, powerful way to visually inspect our filter results is to generate an HTML page for a set of images.

The following functions in `wsi/filter.py` can be used to apply filters to multiple images:

```
apply_filters_to_image_list(image_num_list, save, display)
apply_filters_to_image_range(start_ind, end_ind, save, display)
singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)
multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)

```

The `apply_filters_to_image_list()` function takes a list of image numbers for processing. It does not generate an
HTML page but it does generate information that can be used by other functions to generate an HTML page.
The `save` parameter if `True` will save the generated images to the file system. If the `display` parameter
is `True`, the generated images will be displayed to the screen. If several images are being processed,
`display` should be set to False.

The `apply_filters_to_image_range()` function is similar to `apply_filters_to_image_list()` except than rather than
taking a list of image numbers, it takes a starting index number and ending index number for the slides in the
training set. Like `apply_filters_to_image_list()`, the `apply_filters_to_image_range()` function has `save` and
`display` parameters.

The `singleprocess_apply_filters_to_images()` and `multiprocess_apply_filters_to_images()` functions are the
primary functions that should be called to apply filters to multiple images. Both of these functions feature `save`
and `display` parameters. The additional `html` parameter if `True` generates an HTML page for displaying the filter
results on the image set. The `singleprocess_apply_filters_to_images()` and `multiprocess_apply_filters_to_images()`
functions also feature an `image_num_list` parameter which specifies a list of image numbers that should be
processed. If `image_num_list` is not supplied, all training images will be processed.

As an example, let's use a single process to apply our filters to images 1, 2, and 3. We can accomplish this with
the following:

```
singleprocess_apply_filters_to_images(image_num_list=[1, 2, 3])
```

In addition to saving the filtered images to the file system, this creates a `filters.html` file that displays all the
filtered slide images. This file performs lazy image loading since so many images can potentially be loaded.
If we open the `filters.html` file in a browser, we can see 8 images displayed for each slide. Each separate slide
is displayed as a separate row. Here, we see slides #1 and #2 displayed in a browser.

| **Filters 001 through 004** | **Fitlers 005 through 008** |
| -------------------- | --------------------------------- |
| ![Filters 001 through 004](images/filters-001-004.png "Filters 001 through 004") | ![Fitlers 005 through 008](images/filters-005-008.png "Fitlers 005 through 008") |


To apply all filters to all images in the training set using multiprocessing, we can utilize the
`multiprocess_apply_filters_to_images()` function. Since there are 8 images per slide and 500 slides,
this results in a total of 4000 images. Generating PNG images, this takes about 16 minutes on my Mac laptop.
This can be sped up significantly if JPEG images are used as the format.

```
multiprocess_apply_filters_to_images()
```

If we display the `filters.html` file in a browser, we see that all filter results for all images are displayed.

| **All Slides and Filter Results** |
| -------------------- |
| ![All Slides and Filter Results](images/slides-and-filters-browser.png "All Slides and Filter Results") |


Using this page, one useful action we can take is to manually group similar slides into categories. For example,
we could group slides into slides that have red, green, and blue pen marks on them. 

```
red_pen_slides = [4, 15, 24, 48, 63, 67, 115, 117, 122, 130, 135, 165, 166, 185, 209, 237, 245, 249, 279, 281, 282, 289, 336, 349, 357, 380, 450, 482]
green_pen_slides = [51, 74, 84, 86, 125, 180, 200, 337, 359, 360, 375, 382, 431]
blue_pen_slides = [7, 28, 74, 107, 130, 140, 157, 174, 200, 221, 241, 318, 340, 355, 394, 410, 414, 457, 499]
```

If we would like to increase the effectiveness of red pen filters, we could make tweaks to our
`apply_filters_to_image()` function and run these changes on the set of red pen slides:

```
multiprocess_apply_filters_to_images(image_num_list=red_pen_slides)
```

In this way, we can make tweaks to specific filters or combinations of specific filters and see how these changes apply
to the subset of relevant training images without requiring reprocessing of the entire training dataset.

| **Red Pen Slides with Filter Results** |
| -------------------- |
| ![Red Pen Slides with Filter Results](images/red-pen-slides-filters.png "Red Pen Slides with Filter Results") |


## Overmask Avoidance

When developing filters and filter settings to perform tissue identification on the entire training
set, we have to deal with a great amount of variation in the slide samples. To begin with, some slides have a large
amount of tissue on them, while other slides only have a minimal amount of tissue. There is a great deal of
variation in tissue staining. We also need to deal with additional issues such as pen marks and shadows on some of
the slides.

Slide #498 is an example of a slide with a large tissue sample. After filtering, the slide is 46% masked.

| **Slide with Large Tissue Sample** | **Slide with Large Tissue Sample after Filtering** |
| -- | -- |
| ![Slide with Large Tissue Sample](images/498-rgb.png "Slide with Large Tissue Sample") | ![Slide with Large Tissue Sample after Filtering](images/498-rgb-after-filters.png "Slide with Large Tissue Sample after Filtering") |


Slide #127 is an example of a small tissue sample. After filtering, the slide is 93% masked. With such a small tissue
sample to begin with, we need to be careful that our filters don't overmask this slide.

| **Slide with Small Tissue Sample** | **Slide with Small Tissue Sample after Filtering** |
| -- | -- |
| ![Slide with Small Tissue Sample](images/127-rgb.png "Slide with Small Tissue Sample") | ![Slide with Small Tissue Sample after Filtering](images/127-rgb-after-filters.png "Slide with Small Tissue Sample after Filtering") |


Being aggressive in our filtering may generate excellent results for many of the slides but may
result in overmasking of other slides, where the amount of non-tissue masking is too high. For example, if 99% of
a slide is masked, it has been overmasked.

Avoiding overmasking across the entire training dataset can be difficult. For example, suppose we have a slide that
has only a proportionaly small amount of tissue on it to start, say 10%. If this particular tissue sample has been
poorly stained so that it is perhaps a light purplish grayish color, applying our grays or green channel filters might
result in a significant portion of the tissue being masked out. This could also potentially result in small
islands of non-masked tissue, and since we utilize a filter to remove small objects, this could result in the
further masking out of additional tissue regions. In such a situation, masking of 95% to 100% of the slide is possible.

Slide #424 has a small tissue sample and its staining is a very faint lavender color. Slide #424 is
at risk for overmasking with our given combination of filters.

| **Slide with Small Tissue Sample and Faint Staining** |
| -- |
| ![Slide with Small Tissue Sample and Faint Staining](images/424-rgb.png "Slide with Small Tissue Sample and Faint Staining") |



Therefore, rather than having fixed settings, we can automatically have our filters tweak parameter values to avoid
overmasking if desired. As examples, the `filter_green_channel()` and `filter_remove_small_objects()` functions have
this ability. If this masking exceeds a certain overmasking threshold, a parameter value can be changed to lower
the amount of masking until the masking is below the overmasking threshold.

```
filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool")
filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8")
```

For the `filter_green_channel()` function, if a `green_thresh` value of 200 results in masking over 90%, the
function will try with a higher `green_thresh` value (227) and the masking level will be checked. This will continue
until the masking doesn't exceed the overmask threshold of 90%.

For the `filter_remove_small_objects()` function, if a `min_size` value of 3000 results in a masking level over 95%,
the function will try with a lower `min_size` value (1500) and the masking level will be checked. These `min_size`
reductions will continue until the masking level isn't over 95%.

Examining our full set of images using `multiprocess_apply_filters_to_images()`, we can identify slides that are
at risk for overmasking. We can create a list of these slide numbers and use `multiprocess_apply_filters_to_images()`
with this list of slide numbers to generate our HTML filters page that allows us to visually inspect the filters
applied to this set of slides.

```
overmasked_slides = [1, 21, 29, 37, 43, 88, 116, 126, 127, 142, 145, 173, 196, 220, 225, 234, 238, 284, 292, 294, 304,
                     316, 401, 403, 424, 448, 452, 472, 494]
multiprocess_apply_filters_to_images(image_num_list=overmasked_slides)
```

Let's have a look at how we reduce overmasking on slide 21, which is a slide that has very faint staining.

| **Slide 21** |
| -------------------- |
| ![Slide 21](images/21-rgb.png "Slide 21") |


We'll run our filters on slide #21 by calling `singleprocess_apply_filters_to_images(image_num_list=[21])`.

If we set the `filter_green_channel()` and `filter_remove_small_objects()` `avoid_overmask` parameters to False,
97.15% of the original image is masked by the "green channel" filter and 99.84% of the original image is
masked by the subsequent "remove small objects" filter. This is significant overmasking.

| **Overmasked by Green Channel Filter (97.15%)** | **Overmasked by Remove Small Objects Filter (99.84%)** |
| -- | -- |
| ![Overmasked by Green Channel Filter (97.15%)](images/21-overmask-green-ch.png "Overmasked by Green Channel Filter (97.15%)") | ![Overmasked by Remove Small Objects Filter (99.84%)](images/21-overmask-green-ch-overmask-rem-small-obj.png "Overmasked by Remove Small Objects Filter (99.84%)")

If we set `avoid_overmask` to True for `filter_remove_small_objects()`, we see that the "remove small objects"
filter does not perform any further masking since the 97.15% masking from the previous "green channel" filter
already exceeds its overmasking threshold of 95%.

| **Overmasked by Green Channel Filter (97.15%)** | **Avoid Overmask by Remove Small Objects Filter (97.15%)** |
| -- | -- |
| ![Overmasked by Green Channel Filter (97.15%)](images/21-overmask-green-ch.png "Overmasked by Green Channel Filter (97.15%)") | ![Avoid Overmask by Remove Small Objects Filter (97.15%)](images/21-overmask-green-ch-avoid-overmask-rem-small-obj.png "Avoid Overmask by Remove Small Objects Filter (97.15%)")


If we set `avoid_overmask` back to False for `filter_remove_small_objects()` and we set `avoid_overmask` to True for
`filter_green_channel()`, we see that 88.46% of the original image is masked by the "green channel" filter (under
the 90% overmasking threshold for the filter) and 97.81% of the image is masked by the subsequent
"remove small objects" filter.

| **Avoid Overmask by Green Channel Filter (88.46%)** | **Overmask by Remove Small Objects Filter (97.81%)** |
| -- | -- |
| ![Avoid Overmask by Green Channel Filter (88.46%)](images/21-avoid-overmask-green-ch.png "Avoid Overmask by Green Channel Filter (88.46%)") | ![Overmask by Remove Small Objects Filter (97.81%)](images/21-avoid-overmask-green-ch-overmask-rem-small-obj.png "Overmask by Remove Small Objects Filter (97.81%)")


If we set `avoid_overmask` to True for both `filter_green_channel()` and `filter_remove_small_objects()`, we see that
the resulting masking after the "remove small objects" filter has been reduced to 93.96%, which is under its
overmasking threshold of 95%.

| **Avoid Overmask by Green Channel Filter (88.46%)** | **Avoid Overmask by Remove Small Objects Filter (93.96%)** |
| -- | -- |
| ![Avoid Overmask by Green Channel Filter (88.46%)](images/21-avoid-overmask-green-ch-2.png "Avoid Overmask by Green Channel Filter (88.46%)") | ![Avoid Overmask by Remove Small Objects Filter (93.96%)](images/21-avoid-overmask-green-ch-avoid-overmask-rem-small-obj.png "Avoid Overmask by Remove Small Objects Filter (93.96%)")


Thus, in this example we've reduced the masking from 99.84% to 93.96%.

We can see the filter adjustments being made in the console output.

```
Applying filters to images

Processing slide #21
RGB                  | Time: 0:00:00.168369  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:01.044200  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-001-2048-rgb.png
Mask percentage 97.15% >= overmask threshold 90.00% for Remove Green Channel green_thresh=200, so try 227
Filter Green Channel | Time: 0:00:00.009298  Type: bool    Shape: (1944, 2048)
Filter Green Channel | Time: 0:00:00.021384  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.016247  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.537678  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-002-2048-rgb-not-green.png
Filter Grays         | Time: 0:00:00.130681  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.015921  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.505956  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-003-2048-rgb-not-gray.png
Filter Red Pen       | Time: 0:00:00.103038  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.014579  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.995982  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-004-2048-rgb-no-red-pen.png
Filter Green Pen     | Time: 0:00:00.148576  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.013955  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.981354  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-005-2048-rgb-no-green-pen.png
Filter Blue Pen      | Time: 0:00:00.133999  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.014755  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:01.020773  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-006-2048-rgb-no-blue-pen.png
Mask RGB             | Time: 0:00:00.015858  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.541936  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-007-2048-rgb-no-gray-no-green-no-pens.png
Mask percentage 97.81% >= overmask threshold 95.00% for Remove Small Objs size 500, so try 250
Mask percentage 97.04% >= overmask threshold 95.00% for Remove Small Objs size 250, so try 125
Mask percentage 96.06% >= overmask threshold 95.00% for Remove Small Objs size 125, so try 62
Mask percentage 95.02% >= overmask threshold 95.00% for Remove Small Objs size 62, so try 31
Remove Small Objs    | Time: 0:00:00.076443  Type: bool    Shape: (1944, 2048)
Remove Small Objs    | Time: 0:00:00.160442  Type: bool    Shape: (1944, 2048)
Remove Small Objs    | Time: 0:00:00.231779  Type: bool    Shape: (1944, 2048)
Remove Small Objs    | Time: 0:00:00.300997  Type: bool    Shape: (1944, 2048)
Remove Small Objs    | Time: 0:00:00.376790  Type: bool    Shape: (1944, 2048)
Mask RGB             | Time: 0:00:00.016087  Type: uint8   Shape: (1944, 2048, 3)
Save Image           | Time: 0:00:00.453725  Name: /Volumes/BigData/TUPAC/filter_2048_png/TUPAC-TR-021-008-2048-rgb-not-green-not-gray-no-pens-remove-small.png
Slide #021 processing time: 0:00:07.492792
```

