# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from tqdm import tqdm
import cv2
import numpy as np
from predictor import COCODemo
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from PIL import Image

import argparse
import os

import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets import FolderDataset
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.imports import import_file

# from predictor import Cococompute_prediction, select_top_predictions, Resize
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(
            description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()


    config_file = args.config_file #"../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    for conf_thresh in [0.1,0.3,0.5,0.7,0.9]:
        coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=conf_thresh,
        )
        
        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        DatasetCatalog = paths_catalog.DatasetCatalog    
        for dataset_name in cfg.DATASETS.TEST:

            print(dataset_name)
            dataset = DatasetCatalog.get(dataset_name)
            # print(dataset)
            # print(len(dataset))
            print(dataset)

            dataset = FolderDataset(
                dataset['args']['data_dir'], dataset['args']['split'])
            COCODemo.CATEGORIES = dataset.CLASSES
            for image, target, index in tqdm(dataset):

                image_name = dataset.img_files[index].split("/")[-1]

                image = np.array(image)

                all_labels = [coco_demo.CATEGORIES[i]
                            for i in target.get_field("labels").tolist()]
                if len(all_labels) > 1:
                    print(all_labels)

                    


                ### GROND TRUTH
                result = image.copy()
                if coco_demo.show_mask_heatmaps:
                    return coco_demo.create_mask_montage(result, target)
                result = coco_demo.overlay_boxes(result, target)
                # result = coco_demo.overlay_boxes(result, target)
                if coco_demo.cfg.MODEL.MASK_ON:
                    result = coco_demo.overlay_mask(result, target)
                if coco_demo.cfg.MODEL.KEYPOINT_ON:
                    result = coco_demo.overlay_keypoints(result, target)
                # result = coco_demo.overlay_class_names(result, top_predictions)

                labels = [coco_demo.CATEGORIES[i]
                        for i in target.get_field("labels").tolist()]
                boxes = target.bbox

                for box,  label in zip(boxes,  labels):
                    x, y = box[:2]
                    cv2.putText(
                        result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, 
                        (255, 255, 255), 1
                    )

                result = Image.fromarray(result)

                for label_GT in all_labels:
                    if ".tif" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                        image_name.replace(".tif", "_GT.tif"))
                    if ".jpg" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                        image_name.replace(".jpg", "_GT.jpg"))
                    if ".JPG" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                        image_name.replace(".JPG", "_GT.JPG"))

                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    # result.save(out)
                    if not os.path.exists(out):
                        result.save(out)
        

                ### PREDICTION
                predictions = coco_demo.compute_prediction(image)
                top_predictions = coco_demo.select_top_predictions(predictions)
                # print(top_predictions)

                result = image.copy()
                if coco_demo.show_mask_heatmaps:
                    return coco_demo.create_mask_montage(result, top_predictions)
                result = coco_demo.overlay_boxes(result, top_predictions)
                # result = coco_demo.overlay_boxes(result, target)
                if coco_demo.cfg.MODEL.MASK_ON:
                    result = coco_demo.overlay_mask(result, top_predictions)
                if coco_demo.cfg.MODEL.KEYPOINT_ON:
                    result = coco_demo.overlay_keypoints(
                        result, top_predictions)
                result = coco_demo.overlay_class_names(result, top_predictions)

                result = Image.fromarray(result)

                for label_GT in all_labels:
                    out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                       image_name)

                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    if not os.path.exists(out):
                        result.save(out)

                ### PREDICTION BEST only
                # predictions = coco_demo.compute_prediction(image)
                top_predictions = coco_demo.select_top_predictions(predictions, best_only=True)
                # print(top_predictions)

                result = image.copy()
                if coco_demo.show_mask_heatmaps:
                    return coco_demo.create_mask_montage(result, top_predictions)
                result = coco_demo.overlay_boxes(result, top_predictions)
                # result = coco_demo.overlay_boxes(result, target)
                if coco_demo.cfg.MODEL.MASK_ON:
                    result = coco_demo.overlay_mask(result, top_predictions)
                if coco_demo.cfg.MODEL.KEYPOINT_ON:
                    result = coco_demo.overlay_keypoints(
                        result, top_predictions)
                result = coco_demo.overlay_class_names(result, top_predictions)

                result = Image.fromarray(result)

                for label_GT in all_labels:
                    if ".tif" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                           image_name.replace(".tif", "_best.tif"))
                    if ".jpg" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                           image_name.replace(".jpg", "_best.jpg"))
                    if ".JPG" in image_name:
                        out = os.path.join(cfg.OUTPUT_DIR, f"inference_{conf_thresh}", dataset_name, label_GT,
                                           image_name.replace(".JPG", "_best.JPG"))

                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    if not os.path.exists(out):
                        result.save(out)



if __name__ == "__main__":
    main()
