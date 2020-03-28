from PIL import Image, ImageDraw
import xml.etree.ElementTree
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class FolderDataset(torch.utils.data.Dataset):

    # CLASSES = (
    #     "__background__ ",
    #     "spine",
    # )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, binary=False):
        # self.root = data_dir
        # self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.binary = binary

        self.set_dir = os.path.join(data_dir, split)
        self.anno_files = []
        self.img_files = []
        for root, dirs, files in os.walk(self.set_dir):
            for file in files:
                if file.endswith(".xml"):
                    anno_file = os.path.join(root, file)
                    # for ext in [".tif", ".jpg", ".JPG"]:
                    if os.path.exists(anno_file.replace(".xml", ".tif")):
                        self.anno_files.append(anno_file)
                        self.img_files.append(anno_file.replace(".xml", ".tif"))
                    elif os.path.exists(anno_file.replace(".xml", ".jpg")):
                        self.anno_files.append(anno_file)
                        self.img_files.append(anno_file.replace(".xml", ".jpg"))
                    elif os.path.exists(anno_file.replace(".xml", ".JPG")):
                        self.anno_files.append(anno_file)
                        self.img_files.append(anno_file.replace(".xml", ".JPG"))
        self.anno_files.sort()
        self.img_files.sort()
        # print(self.anno_files)
        # print(self.img_files)
        self.typo_dict = {"spine-head portrusion": "spine-head protrusion",
                          "Ingey": "Inegy",
                          "Omancillin": "Omacillin",
                          "Falgyl": "Flagyl",
                          "Geminated": "Germinated",
                          "Non-geminated": "Non-germinated",
                          "Radical": "Radicle"}

        # if "Seeds_Striga_Strategy1" in data_dir:
        #     self.typo_dict["Dead]"] = "Germinated"
        #     self.typo_dict["Dead"] = "Germinated"
        #     self.typo_dict["dead"] = "Germinated"
        # elif "Seeds_Striga_Strategy2" in data_dir:
        #     self.typo_dict["Dead]"] = "Radicle"
        #     self.typo_dict["Dead"] = "Radicle"
        #     self.typo_dict["dead"] = "Radicle"

        self.ignore_dict = {"A": "__background__",
                            "B": "__background__",
                            "C": "__background__",
                            "D": "__background__",
                            "E": "__background__",
                            "F": "__background__",
                            "G": "__background__",
                            "H": "__background__",
                            "I": "__background__",
                            "K": "__background__",}
        self.ignore_dict["Dead]"] = "__background__"
        self.ignore_dict["Dead"] = "__background__"
        self.ignore_dict["dead"] = "__background__"
                        #   "Radical": "Radicle",
        # if "Seeds_Striga_Strategy2" in data_dir:
        #     self.typo_dict.update({"Germinated": "__background__"})
        #     self.typo_dict.update({"Non-germinated": "__background__"})

        self.CLASSES = []
        # self.rotated = [-1 for i in self.anno_files]
        for anno_file in self.anno_files:
            # anno = ET.parse(anno_file).getroot()
            # anno = self._preprocess_annotation(anno)
            # labels = anno["labels"]
            # print(anno_file)
            root = xml.etree.ElementTree.parse(anno_file).getroot()
            #     gernimated_string = 'germinated'
            #     non_gernimated_string = 'non-germinated'
            objects = root.findall('object')
            # b_boxes_xml = [obj.find('bndbox') for obj in objects]
            labels = [obj.find('name').text for obj in objects]
            for label in labels:
                label = label.replace(" ", "").capitalize()
                # if len(label) < 2:
                #     continue
                if self.binary:
                    label = "object"


                ignore_BB = False
                for ign in self.ignore_dict:
                    if label == ign:
                        ignore_BB = True
                if ignore_BB:
                    continue

                for typo in self.typo_dict:
                    if label == typo:
                        label = self.typo_dict[typo]
                # elif label == "spine-head portrusion":
                #     label = "spine-head protrusion"
                # elif label == "Ingey":
                #     label = "Inegy"
                # elif label == "Omancillin":
                #     label = "Omacillin"
                # elif label == "Falgyl":
                #     label = "Flagyl"
                    
                    # print(anno_file)
                if label not in self.CLASSES and label is not "__background__":
                    self.CLASSES.append(label)
            # b_boxes = [(int(box.find('xmin').text), int(box.find('ymin').text), int(
            #     box.find('xmax').text), int(box.find('ymax').text)) for box in b_boxes_xml]
            # return b_boxes, labels
        self.CLASSES.sort()
        self.CLASSES.insert(0, "__background__")
        print(self.CLASSES)
        # if "Seeds_Striga_Strategy1" in data_dir:
        #     self.CLASSES = ["__background__", "Germinated", "Non-germinated"]
        #     print("Seeds_Striga_Strategy1:", self.CLASSES)
        # print(data_dir)
        # if "Seeds_Striga_Strategy2" in data_dir and "merey" in data_dir:
        # self.CLASSES = ["__background__", "Seed", "Radicle"]
        # print("Seeds_Striga_Strategy2:", self.CLASSES)


        # image = Image.open(image_name).convert('RGB')


        # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        # self._imgsetpath = os.path.join(
        #     self.root, "ImageSets", "Main", "%s.txt")

        # with open(self._imgsetpath % self.image_set) as f:
        #     self.ids = f.readlines()
        # self.ids = [x.strip("\n") for x in self.ids]
        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(self.CLASSES, range(len(self.CLASSES))))
        self.categories = dict(zip(range(len(self.CLASSES)), self.CLASSES))

    def __getitem__(self, index):
        img_file = self.img_files[index]
        # print(anno_file)
        # img_file = anno_file.replace(".xml",".tif")
        # for ext in ["jpg","jpeg",""]
        img = Image.open(img_file).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        # h, w = img.size
        # print(img)
        # AR_before = h/w
        # self.rotated[index] = 0
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # print(img)
        # print(img.shape)
        # h = img.shape[1]
        # w = img.shape[2]
        # if h > w:
        #     self.rotated[index] = 1
        # AR_after = h/w
        # print(img.size)
        # if AR_before / AR_after 

        return img, target, index

    def __len__(self):
        return len(self.anno_files)

    def get_groundtruth(self, index):
        # img_id = self.ids[index]
        anno_file = self.anno_files[index]

        anno = ET.parse(anno_file).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            name = name.replace(" ", "").capitalize()
            if self.binary:
                name = "object"
            # if len(name) < 2:
            #     name = "__background__"


            ignore_BB = False
            for ign in self.ignore_dict:
                if name == ign:
                    ignore_BB = True
            if ignore_BB:
                continue

            for typo in self.typo_dict:
                if name == typo:
                    name = self.typo_dict[typo]
            if name not in self.CLASSES:
                continue
            # if name == "spine-head portrusion":
            #     name = "spine-head protrusion"
            # elif name == "Ingey":
            #     name = "Inegy"
            # elif name == "Omancillin":
            #     name = "Omacillin"
            # elif name == "Falgyl":
            #     name = "Flagyl"
            # print(name)
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        # if self.rotated[index] == -1:
        #     print("Too Early!")
        # image, target, index = self[index]
        # if self.rotated[index] == -1:
        #     h, w = image.size
        # return {"height": h, "width": w}

        # img_file = self.img_files[index]
        anno_file = self.anno_files[index]
        # img_id = self.ids[index]
        anno = ET.parse(anno_file).getroot()
        size = anno.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))
        # if self.rotated[index] == 1:
        #     return {"height": im_info[1], "width": im_info[0]}
        
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return self.CLASSES[class_id]


if __name__ == "__main__":

    import pandas as pd
    
    # for project in ["Seeds_Orobanche_Strategy1",
    #                 "Seeds_Orobanche_Strategy1_finetuning_201030",
    #                 "Seeds_Orobanche_Strategy1_scratch_201030",
    #                 "Seeds_Orobanche_Strategy1_test_201030",
    #                 "Seeds_Striga_Strategy1"]:
    for split in ["Testing", "Validation", "Training"]:
        dataset1 = FolderDataset(
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy1",
            data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy1_finetuning_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy1_scratch_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy1_test_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy1",
                split=split)
        dataset2 = FolderDataset(
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy2",
            data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy2_finetuning_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy2_scratch_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Orobanche_Strategy2_test_201030",
            # data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy2",
                split=split)
        print(len(dataset1), len(dataset2))

        df = pd.DataFrame(columns=["name1", "name2", "GS", "NGS", "R", "S",])
        from tqdm import tqdm
        for (_, target1, index),(_, target2, _), in tqdm(zip(dataset1, dataset2)):
            # print(dataset1.map_class_id_to_class_name(1), 
            #       dataset1.map_class_id_to_class_name(2))
            name1 = dataset1.anno_files[index]
            name2 = dataset2.anno_files[index]
            GS = len([label for label in target1.get_field("labels").numpy().tolist() if (label == 1)])
            NGS = len([label for label in target1.get_field("labels").numpy().tolist() if (label == 2)])
            R = len([label for label in target2.get_field("labels").numpy().tolist() if (label == 1)])
            S = len([label for label in target2.get_field("labels").numpy().tolist() if (label == 2)])
            if (not GS == R) or (not NGS == S-R):
                print(name1)
                print(name2)
                # print("GS:", GS)
                # print("NGS:", NGS)
                # print("R:", R)
                # print("S:", S)

            # df = df.append([name1, name2, GS, NGS, R, S,])
            df = df.append({"name1": name1, "name2": name2, "GS": GS,
                            "NGS": NGS, "R": R, "S": S}, ignore_index=True)


        df.to_csv(f"/home/giancos/Downloads/Consistency_Striga_{split}.csv")
        # print(target1.get_field("labels").numpy().tolist() == 2))
        # print(dataset2.map_class_id_to_class_name(1),
        #       dataset2.map_class_id_to_class_name(2))
        # print(target2.get_field("labels").numpy().tolist())
        # print(target1["labels"], target2)

    # from maskrcnn_benchmark.config import cfg
    # cfg.INPUT.ROTATE_PROB_TRAIN = 0.5
    # cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
    # cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.5
    # # min_size = cfg.INPUT.MIN_SIZE_TRAIN
    # # max_size = cfg.INPUT.MAX_SIZE_TRAIN
    # # cfg.INPUT.ROTATE_PROB_TRAIN = 1.0


    # # from maskrcnn_benchmark.data.build import build_transforms
    # from maskrcnn_benchmark.data.transforms import transforms as T
    # min_size = cfg.INPUT.MIN_SIZE_TRAIN
    # max_size = cfg.INPUT.MAX_SIZE_TRAIN
    # flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
    # flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
    # rotate_prob = cfg.INPUT.ROTATE_PROB_TRAIN
    # brightness = cfg.INPUT.BRIGHTNESS
    # contrast = cfg.INPUT.CONTRAST
    # saturation = cfg.INPUT.SATURATION
    # hue = cfg.INPUT.HUE
    
    # to_bgr255 = cfg.INPUT.TO_BGR255
    # normalize_transform = T.Normalize(
    #     mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    # )
    # color_jitter = T.ColorJitter(
    #     brightness=brightness,
    #     contrast=contrast,
    #     saturation=saturation,
    #     hue=hue,
    # )

    # transform = T.Compose(
    #     [
    #         color_jitter,
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_horizontal_prob),
    #         T.RandomVerticalFlip(flip_vertical_prob),
    #         T.RandomRightRotation(rotate_prob),
    #         # T.ToTensor(),
    #         # normalize_transform,
    #     ]
    # )
    # # transform = build_transforms(cfg)





    # # print(transform)

    # dataset = FolderDataset(
    #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy1", 
    #     split="Training",
    #     transforms=transform)

    # # print(dataset.CLASSES)
    # # print(len(dataset.CLASSES))
    # # print(len(dataset))
    # image, target, index = dataset[0]
    # print(image)
    # print(dataset.get_img_info(0))
    # print(dataset.get_img_info(0))
    # print(dataset.get_img_info(0))
    # print(dataset.get_img_info(0))
    # print(dataset.get_img_info(0))
    # print(dataset.get_img_info(0))
    # print(image.size)
    # print(target.bbox[0])

    # import numpy as np
    # import cv2

    # # image = np.array(image, dtype=np.int)
    # # result = image.copy()
    # print(image)
    # img1 = ImageDraw.Draw(image)
    # w = 10
    # h=30
    # shape = [(40, 40), (w - 10, h - 10)]

    # for box in target.bbox:
    #     # box = box.to(torch.int64)
    #     top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    #     top_left = tuple([int(i) for i in top_left])
    #     bottom_right = tuple([int(i) for i in bottom_right])
    #     # print(top_left, bottom_right)
    #     img1.rectangle([top_left, bottom_right], outline="red")
    # #     image = cv2.rectangle(
    # #         image, top_left, bottom_right, 255, 10
    # #     )

    # # image = Image.fromarray(image)
    # # image.save("/home/giancos/Downloads/test.jpg")
                


    # # TESTING
    # from tqdm import tqdm

    # # dataset = FolderDataset(
    # #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy1", 
    # #     split="Training",
    # #     transforms=transform)
    # # print(len(dataset))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # # dataset = FolderDataset(
    # #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy1", 
    # #     split="Validation",
    # #     transforms=transform)
    # # print(len(dataset))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # # dataset = FolderDataset(
    # #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy1", 
    # #     split="Testing",
    # #     transforms=transform)
    # # print(len(dataset))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # # dataset = FolderDataset(
    # #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy2",
    # #     split="Training",
    # #     transforms=transform)
    # # print(len(dataset))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # # dataset = FolderDataset(
    # #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy2",
    # #     split="Validation",
    # #     transforms=transform)
    # # print(len(dataset))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # dataset = FolderDataset(
    #     data_dir="/media/giancos/Football/CloudLabeling/Seeds_Striga_Strategy2",
    #     split="Testing",
    #     transforms=transform)
    # print(len(dataset), "images")
    # cnt_radicle = 0
    # cnt_seed = 0    
    # for img, target, index in tqdm(dataset):
    #     cnt_radicle += len([box for box in target.get_field("labels").tolist() if ("Rad" in dataset.map_class_id_to_class_name(box))])
    #     cnt_seed += len([box for box in target.get_field("labels").tolist() if ("Seed" in dataset.map_class_id_to_class_name(box))])

    # print("cnt_radicle:", cnt_radicle)
    # print("cnt_seed:", cnt_seed)
    # #     for box in target.get_field("labels").tolist():
    # #         print(dataset.map_class_id_to_class_name(box))
    # # print("nb_sample =", np.sum([len(target) for img, target, index in tqdm(dataset)]))

    # # # print(dataset.CLASSES)
    # # print(len(dataset.CLASSES))
    # # print(len(dataset))
    # # lst = []
    # # for image, boxes, idx in dataset:
    # #     for box in boxes.get_field("labels").tolist():
    # #         lst.append(dataset.map_class_id_to_class_name(box))

    # # from collections import Counter
    # # cnt = Counter(lst)
    # # print(cnt)
    # # print(len(cnt))
