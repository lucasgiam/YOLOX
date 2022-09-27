# encoding: utf-8
"""
1. Changed data_dir to root in __init__ of voc.py's VOCDetection for naming consistency.
Note that the root is the parent directory of VOCdevkit and remain untouched.
2. Remember to change self.num_classes as well as coco_classes.py and voc_classes.py as it can cause cuda run time error.
"""

import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp


# System/model configs
datadir = "./datasets/sp_ppe_1/"
custom_data_folder_name = ""
num_classes = 8
data_num_workers = 32
seed = 1993
depth = 0.67    # yolo-s = 0,33, yolo-m = 0.67, yolo-l = 1.00
width = 0.75    # yolo-s = 0.50, yolo-m = 0.75, yolo-l = 1.00
warmup_epochs = 1


# Data augmentation configs
mosaic_prob = 1.0
mixup_prob = 1.0
hsv_prob = 1.0
flip_prob = 0.5
degrees = 10.0
translate = 0.1
scale = (0.1, 2)
mosaic_scale = (0.8, 1.6)
shear = 2.0
perspective = 0.0
enable_mixup = True


# Training configs
input_size = (640, 640)   # (height, width)
max_epoch = 50
test_size = (640, 640)    # (height, width)
# log period in iter, for example, if set to 1, user could see log every iteration.
print_interval = 1
# eval period in epoch, for example, if set to 1, model will be evaluate after every epoch.
eval_interval = 1


# For debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = num_classes
        print("num_classes", self.num_classes)
        self.depth = depth
        self.width = width
        self.warmup_epochs = warmup_epochs
        # ---------- transform config ------------ #
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.hsv_prob = hsv_prob
        self.flip_prob = flip_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.mosaic_scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.enable_mixup = enable_mixup
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(
            "."
        )[0]
        self.data_num_workers = data_num_workers
        self.seed = seed
        self.input_size = input_size
        self.max_epoch = max_epoch
        self.test_size = test_size
        self.print_interval = print_interval
        self.eval_interval = eval_interval

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
                root=os.path.join(datadir, "VOCdevkit"),
                image_sets=[("2012", "train",)],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
                custom_data_folder_name=custom_data_folder_name,
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
            root=os.path.join(datadir, "VOCdevkit"),
            image_sets=[("2012", "val")],  # image_sets=[("2012", "train")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            custom_data_folder_name=custom_data_folder_name,
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
