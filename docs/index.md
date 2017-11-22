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
set we looked at consisted of 500 whole-slide images which are scored (1, 2, or 3) by pathologists based on mitosis
counts. A higher proliferation score indicates a worse prognosis since higher tumor proliferation speeds are
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
identification decreases the quantity of data and increases the quality of the data to be analyzed. This
can lead to faster, more accurate model training.
In this tutorial, we will take a look at whole-slide image processing and will describe various filters
that can be used to increase the accuracy of tissue identification.


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

**WSI Example Slide**<br/>
![WSI Example Slide](images/wsi-example.png "WSI Example Slide")


A whole-slide image is created by a microscope that scans a slide and combines smaller images into a large image.
Techniques include combining scanned square tiles into a whole-slide image and combining scanned strips
into a resulting whole-slide image. Occasionally, the smaller constituent images can be
visually discerned, as in the shaded area at the top of the slide seen below.

**Combining Smaller Images into a Whole-Slide Image**<br/>
![Combining Smaller Images into a Whole-Slide Image](images/slide-scan.png "Combining Smaller Images into a Whole-Slide Image")


A fairly unusual feature of whole-slide images is the very large image size.
For our training data set of 500 images, the width varied from 19,920 pixels to 198,220 pixels,
with an average of 101,688 pixels. The height varied from 13,347 pixels to 256,256 pixels,
with an average of 73,154 pixels. The image total pixel sizes varied from
369,356,640 to 35,621,634,048 pixels, with an average of
7,670,709,628 pixels. The 500 training images take up a total of 525 GB of storage space.

**Training Image Sizes**<br/>
![Training Image Sizes](images/svs-image-sizes.png "Training Image Sizes")


Here we see a histogram distribution of the training image sizes in megapixels.

**Distribution of Images Based on Number of Pixels**<br/>
![Distribution of Image Sizes](images/distribution-of-svs-image-sizes.png "Distribution of Image Sizes")


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

**OpenSlide Available Slides**<br/>
![OpenSlide Available Slides](images/openslide-available-slides.png "OpenSlide Available Slides")


Here we can see the initial view of one of the whole-slide images viewed in a web browser.

**OpenSlide Whole Slide Image**<br/>
![OpenSlide Whole Slide Image](images/openslide-whole-slide-image.png "OpenSlide Whole Slide Image")


Using this web interface, the whole-slide image can be zoomed to the highest magnification, revealing fine details at
the tile level. Zooming and scrolling operations make it relatively easy to visually peruse the whole slide image.

**OpenSlide Whole Slide Image Zoomed**<br/>
![OpenSlide Whole Slide Image Zoomed](images/openslide-whole-slide-image-zoomed.png "OpenSlide Whole Slide Image Zoomed")


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
    singleprocess_convert_training_slides_to_images()
    multiprocess_convert_training_slides_to_images()

The `open_slide()` function uses OpenSlide to read in an svs file. The `slide_info()` function displays metadata
associated with each svs file. The `slide_stats()` function looks at all images and summarizes pixel size information
about the set of slides. It also generates a variety of charts for a visual representation of the slide statistics.
The `training_slide_to_image()` function converts a single svs slide to a smaller image in a more common format such as
jpg or png. The `singleprocess_convert_training_slides_to_images()` function converts all svs slides to smaller images,
and the `multiprocess_convert_training_slides_to_images()` function uses multiple processes (1 process per core) to
speed up the slide conversion process.

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
the `DEST_TRAIN_SIZE` constant controls the maximum width or height value. For our purposes, its default value will
be 2048. The `DEST_TRAIN_EXT` constant controls the output format. We will use `png` since it is lossless.
Note that `jpg` conversion can also be specified, but `jpg` is lossy. For later deep learning purposes with TensorFlow,
we have determined that a lossless format is preferable.

Using OS X with an external hard drive containing the training set, the following conversion numbers using
`singleprocess_convert_training_slides_to_images()` and `multiprocess_convert_training_slides_to_images()`
were obtained:

    jpg single process: 4m47s
    jpg multi process: 1n37s
    png single process: 11m22s
    png multi process: 3m08s


After calling `multiprocess_convert_training_slides_to_images()` using the `png` format, we have 500 whole-slide
images in lossless png format that we can now examine in much greater detail in relation to our filters.

