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

# Python WSI Preprocessing


## Outline

  1. [Project Introduction](#project-introduction)
  2. [Whole Slide Imaging Background](#whole-slide-imaging-background)
  3. [Setup](#setup)
  4. [View Individual Whole Slide Images](#view-individual-whole-slide-images)
  5. [View Multiple Images](#view-multiple-images)
  6. [Image Filtering](#image-filtering)
  7. [View Filter Results for Individual Images](#view-filter-results-for-individual-images)
  8. [View Filter Results for Multiple Images](#view-filter-results-for-multiple-images)


## Project Introduction


## Whole Slide Imaging Background


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
such as 20x or 40x. As a result of this high magnification, whole slide images are typically very large in size. In
the training dataset that we will examine here in relation to our breast cancer tumor proliferation prediction project,
image sizes were as high as 35.6 gigapixels, with a maximum width of
almost 200,000 pixels and a maximum height over 250,000 pixels. The maximum file size for a single whole-slide
image in the training dataset was 3.4 GB.

**WSI Example Slide**<br/>
![WSI Example Slide](images/wsi-example.png "WSI Example Slide")


A whole-slide image is created by a microscope that scans a slide and combines smaller images into a large image.
Techniques include combining scanned square tiles into a whole-slide image, and scanning strips and combining these
scanned strips into the resulting whole-slide image. Occasionally, the smaller images can clearly be delineated
visually, as in the shaded area at the top of the slide seen below.

**Combining Smaller Images into a Whole-Slide Image**<br/>
![Combining Smaller Images into a Whole-Slide Image](images/slide-scan.png "Combining Smaller Images into a Whole-Slide Image")


## View Individual Whole Slide Images

A fairly unusual feature of whole slide images is the very large image size. As an example,
for our training data set of 500 images, the width varied from 19,920 pixels to 198,220 pixels,
with an average of 101,688 pixels. The height varied from 13,347 pixels to 256,256 pixels,
with an average of 73,154 pixels. The image pixel sizes varied from
369,356,640 to 35,621,634,048 pixels, with an average of
7,670,709,628 pixels. The 500 training images take up a total of 525 GB of space, with
each image averaging over 1 GB in size.

![Training Image Sizes](images/graph-image-sizes.png "Training Image Sizes")
TODO: Clean up image size graph.


Here we see a histogram distribution of the training image sizes in megapixels.

![Distribution of Image Sizes](images/distribution-of-image-sizes.png "Distribution of Image Sizes")
TODO: Clean up graph, change millions of pixels to megapixels.


The [OpenSlide](http://openslide.org/) project can be used to read a variety of whole-slide
image formats, including the [Aperio *.svs slide format](http://openslide.org/formats/aperio/)
of our image set. This is a pyramidal, tiled format, where the massive slide is composed of
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

![OpenSlide Available Images](images/openslide-available-slides.png "OpenSlide Available Slides")


Here we can see the initial view of one of the whole-slide images.

![OpenSlide Whole Slide Image](images/openslide-whole-slide-image.png "OpenSlide Whole Slide Image")


The whole-slide image can be zoomed to the highest magnification, revealing fine details at the
tile level. Zooming and scrolling operations make it relatively easy to visually peruse the
whole slide image.

![OpenSlide Whole Slide Image Zoomed](images/openslide-whole-slide-image-zoomed.png "OpenSlide Whole Slide Image Zoomed")


## View Multiple Images


## Image Filtering


## View Filter Results for Individual Images


## View Filter Results for Multiple Images


