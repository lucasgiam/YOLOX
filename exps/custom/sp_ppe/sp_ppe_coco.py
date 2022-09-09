#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(
            "."
        )[0]

        # Define yourself dataset path
        self.data_dir = r"./datasets/sp_ppe"
        self.train_ann = "train2017.json"
        self.val_ann = "val2017.json"

        self.num_classes = 1

        self.max_epoch = 20
        self.data_num_workers = 0  # windows user
        self.eval_interval = 1

        self.input_size = (640, 640)

        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    # def get_data_loader(
    #     self, batch_size, is_distributed, no_aug=False, cache_img=False
    # ):
    #     print("Get data loader from SP_PPE")
    #     from yolox.data import (
    #         COCODataset,
    #         TrainTransform,
    #         YoloBatchSampler,
    #         DataLoader,
    #         InfiniteSampler,
    #         MosaicDetection,
    #         worker_init_reset_seed,
    #     )
    #     from yolox.utils import wait_for_the_master

    #     # TODO: added name to be compatible with my folder structure
    #     with wait_for_the_master():
    #         dataset = COCODataset(
    #             data_dir=self.data_dir,
    #             json_file=self.train_ann,
    #             img_size=self.input_size,
    #             preproc=TrainTransform(
    #                 max_labels=50,
    #                 flip_prob=self.flip_prob,
    #                 hsv_prob=self.hsv_prob,
    #             ),
    #             cache=cache_img,
    #             name="images/train2017",
    #         )
    #         print("ABC", len(dataset))

    #     dataset = MosaicDetection(
    #         dataset,
    #         mosaic=not no_aug,
    #         img_size=self.input_size,
    #         preproc=TrainTransform(
    #             max_labels=120,
    #             flip_prob=self.flip_prob,
    #             hsv_prob=self.hsv_prob,
    #         ),
    #         degrees=self.degrees,
    #         translate=self.translate,
    #         mosaic_scale=self.mosaic_scale,
    #         mixup_scale=self.mixup_scale,
    #         shear=self.shear,
    #         enable_mixup=self.enable_mixup,
    #         mosaic_prob=self.mosaic_prob,
    #         mixup_prob=self.mixup_prob,
    #     )

    #     self.dataset = dataset

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()

    #     sampler = InfiniteSampler(
    #         len(self.dataset), seed=self.seed if self.seed else 0
    #     )

    #     batch_sampler = YoloBatchSampler(
    #         sampler=sampler,
    #         batch_size=batch_size,
    #         drop_last=False,
    #         mosaic=not no_aug,
    #     )

    #     dataloader_kwargs = {
    #         "num_workers": self.data_num_workers,
    #         "pin_memory": True,
    #     }
    #     dataloader_kwargs["batch_sampler"] = batch_sampler

    #     # Make sure each process has different random seed, especially for 'fork' method.
    #     # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    #     dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    #     train_loader = DataLoader(self.dataset, **dataloader_kwargs)

    #     return

    # def get_eval_loader(
    #     self, batch_size, is_distributed, testdev=False, legacy=False
    # ):
    #     from yolox.data import COCODataset, ValTransform

    #     # TODO: changed name to be compatible with my folder structure
    #     valdataset = COCODataset(
    #         data_dir=self.data_dir,
    #         json_file=self.val_ann if not testdev else self.test_ann,
    #         name="images/val2017" if not testdev else "images/test2017",
    #         img_size=self.test_size,
    #         preproc=ValTransform(legacy=legacy),
    #     )

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             valdataset, shuffle=False
    #         )
    #     else:
    #         sampler = torch.utils.data.SequentialSampler(valdataset)

    #     dataloader_kwargs = {
    #         "num_workers": self.data_num_workers,
    #         "pin_memory": True,
    #         "sampler": sampler,
    #     }
    #     dataloader_kwargs["batch_size"] = batch_size
    #     val_loader = torch.utils.data.DataLoader(
    #         valdataset, **dataloader_kwargs
    #     )

    #     return
