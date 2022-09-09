#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# VOC_CLASSES = ( '__background__', # always index 0
# VOC_CLASSES = (
#     "aeroplane",
#     "bicycle",
#     "bird",
#     "boat",
#     "bottle",
#     "bus",
#     "car",
#     "cat",
#     "chair",
#     "cow",
#     "diningtable",
#     "dog",
#     "horse",
#     "motorbike",
#     "person",
#     "pottedplant",
#     "sheep",
#     "sofa",
#     "train",
#     "tvmonitor",
# )

# VOC_CLASSES = ("person",)

VOC_CLASSES = (
    "no_ppe",
    "helmet",
    "mask",
    "vest",
)
# i believe voc classes is in string in xml so sequence dont matter here?
VOC_CLASSES = (
    "no_ppe",
    "all_ppe",
    "helmet",
    "mask",
    "vest",
    "mask-vest",
    "helmet-mask",
    "helmet-vest",
)
