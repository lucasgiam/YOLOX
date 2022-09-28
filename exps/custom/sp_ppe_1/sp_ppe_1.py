# encoding: utf-8
"""
Remember to change self.num_classes as well as coco_classes.py and voc_classes.py as it can cause cuda run time error.
"""

import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


# Root directory for dataset
datadir = "./datasets/sp_ppe_1/"


# For debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------- model config ------------ #
        self.num_classes = 8
        self.seed = 1993
        self.depth = 0.67           # yolo-s = 0.33, yolo-m = 0.67, yolo-l = 1.00
        self.width = 0.75           # yolo-s = 0.50, yolo-m = 0.75, yolo-l = 1.00
        self.warmup_epochs = 1
        self.max_epoch = 50
        self.data_num_workers = 2
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.print_interval = 1     # log period in iter, for example, if set to 1, user could see log every iteration
        self.eval_interval = 1      # eval period in epoch, for example, if set to 1, model will be evaluate after every epoch
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2.0)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        # ---------------------------------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        print("num_classes: ", self.num_classes)

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):

            dataset = VOCDetection(
                data_dir=os.path.join(datadir, "VOCdevkit"),
                image_sets=[("2012", "train",)],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        # dataset = torch.utils.data.Subset(dataset, range(64))
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            data_dir=os.path.join(datadir, "VOCdevkit"),
            image_sets=[("2012", "val")],  # image_sets=[("2012", "train")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(
            valdataset, **dataloader_kwargs
        )

        return val_loader

    def get_evaluator(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(
            batch_size, is_distributed, testdev, legacy
        )
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
