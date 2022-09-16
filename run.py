#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
import argparse
import os
import time
from loguru import logger
import cv2

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp, Exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

__all__ = [
    "create_yolox_model",
    "yolox_nano",
    "yolox_tiny",
    "yolox_s",
    "yolox_m",
    "yolox_l",
    "yolox_x",
    "yolov3",
    "yolox_custom"
]

_CKPT_ROOT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download"
_CKPT_FULL_PATH = {
    "yolox-nano": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_nano.pth",
    "yolox-tiny": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_tiny.pth",
    "yolox-s": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_s.pth",
    "yolox-m": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_m.pth",
    "yolox-l": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_l.pth",
    "yolox-x": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_x.pth",
    "yolov3": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_darknet.pth",
}


def create_yolox_model(name: str, pretrained: bool = True, num_classes: int = 80, device=None,
                       exp_path: str = None, ckpt_path: str = None) -> nn.Module:
    """creates and loads a YOLOX model

    Args:
        name (str): name of model. for example, "yolox-s", "yolox-tiny" or "yolox_custom"
        if you want to load your own model.
        pretrained (bool): load pretrained weights into the model. Default to True.
        device (str): default device to for model. Default to None.
        num_classes (int): number of model classes. Default to 80.
        exp_path (str): path to your own experiment file. Required if name="yolox_custom"
        ckpt_path (str): path to your own ckpt. Required if name="yolox_custom" and you want to
            load a pretrained model


    Returns:
        YOLOX model (nn.Module)
    """

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    assert name in _CKPT_FULL_PATH or name == "yolox_custom", \
        f"user should use one of value in {_CKPT_FULL_PATH.keys()} or \"yolox_custom\""
    if name in _CKPT_FULL_PATH:
        exp: Exp = get_exp(exp_name=name)
        exp.num_classes = num_classes
        yolox_model = exp.get_model()
        if pretrained and num_classes == 80:
            weights_url = _CKPT_FULL_PATH[name]
            ckpt = load_state_dict_from_url(weights_url, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            yolox_model.load_state_dict(ckpt)
    else:
        assert exp_path is not None, "for a \"yolox_custom\" model exp_path must be provided"
        exp: Exp = get_exp(exp_file=exp_path)
        yolox_model = exp.get_model()
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            yolox_model.load_state_dict(ckpt)

    yolox_model.to(device)
    return yolox_model


EXP_PATH = "./exps/custom/sp_ppe/sp_ppe_voc_all_combinations.py"
SAVE_RESULT = "./YOLOX_outputs/sp_ppe_voc_all_combinations/vis_res"
SOURCE_PATH = "./datasets/pascal_voc/images/all_ppe--1-_jpg.rf.90bf10c92a532b472017aae1ff221085.jpg"
CKPT_FILE = "./YOLOX_outputs/sp_ppe_voc_all_combinations/best_ckpt.pth"


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        print(
            "REMEMBER TO CHANGE COCO CLASSES AS WELL EVEN IF YOU ARE USING VOC CLASSES"
        )
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(
            self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1]
        )
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        print("outputs", outputs)
        result_image = predictor.visual(
            outputs[0], img_info, predictor.confthre
        )
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(
                save_folder, os.path.basename(image_name)
            )
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, demo_type, save_result, source_path):
    cap = cv2.VideoCapture(source_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if demo_type == "video":
            save_path = os.path.join(save_folder, os.path.basename(source_path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(
                outputs[0], img_info, predictor.confthre
            )
            if save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, save_result, device, demo_type):
    file_name = exp.output_dir
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    model = exp.get_model()
    print("model", model)

    ckpt = torch.load(CKPT_FILE, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    logger.info(
        "Model Summary: {}".format(get_model_info(model, exp.test_size))
    )

    if device == "gpu":
        model.cuda()
    model.eval()

    predictor = Predictor(
        model,
        exp,
        COCO_CLASSES,
        device,
    )
    current_time = time.localtime()
    if demo_type == "image":
        image_demo(
            predictor, vis_folder, SOURCE_PATH, current_time, save_result
        )
    elif demo_type == "video" or demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, SOURCE_PATH)


if __name__ == "__main__":
    # create_yolox_model(name="yolox_custom", pretrained=True, num_classes=8, device=None,
    # exp_path="./exps/custom/sp_ppe/sp_ppe_voc_all_combinations.py", ckpt_path="./YOLOX_outputs/sp_ppe_voc_all_combinations/best_ckpt.pth")
    exp = get_exp(EXP_PATH)
    main(exp, SAVE_RESULT, "gpu", "image")
