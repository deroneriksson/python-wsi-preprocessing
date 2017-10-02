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
description of Python is beyond the scope of this journey, but some quick setup steps on OS X
follow.

Install a package manager such as [Homebrew](https://brew.sh/).

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Install [Python3](https://www.python.org/).

    brew install python3

Install [OpenSlide](http://openslide.org/). OpenSlide can be used to read whole slide images.
Note that OpenSlide is licensed under the [LGPL 2.1
License](https://raw.githubusercontent.com/openslide/openslide/master/lgpl-2.1.txt).

    brew install openslide

Optionally, install OpenCV3. [OpenCV](http://opencv.org/) contains a variety of filters and other image processing
utilities.

    brew install opencv3

Next, we can install a variety of useful Python packages using the [pip3](https://pip.pypa.io/en/stable/)
package manager. These packages include:
[ipython](https://pypi.python.org/pypi/ipython),
[jupyter](https://pypi.python.org/pypi/jupyter),
[matplotlib](https://pypi.python.org/pypi/matplotlib/),
[numpy](https://pypi.python.org/pypi/numpy),
[opencv-python](https://pypi.python.org/pypi/opencv-python),
[openslide-python](https://pypi.python.org/pypi/openslide-python),
[pandas](https://pypi.python.org/pypi/pandas),
[scikit-image](https://pypi.python.org/pypi/scikit-image),
[scikit-learn](https://pypi.python.org/pypi/scikit-learn),
and [scipy](https://pypi.python.org/pypi/scipy).

    pip3 install -U ipython jupyter matplotlib numpy opencv-python openslide-python pandas scikit-image scikit-learn scipy


## View Individual Whole Slide Images

## View Multiple Images

## Image Filtering

## View Filter Results for Individual Images

## View Filter Results for Multiple Images


