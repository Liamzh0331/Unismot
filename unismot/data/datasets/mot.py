# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import cv2
import numpy as np
import os

from pycocotools.coco import COCO
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

mot_list = ['person_40103_1_511', 'person_40105_1_416', 'person_40106_1_372', 'person_40107_1_594', 'person_40109_1_350',
            'person_40109_2_1005', 'person_40112_1_284', 'person_40112_2_533', 'person_41007_1_400', 'person_41008_1_275',
            'person_41008_2_507', 'person_41008_3_510', 'person_41008_4_606', 'person_41008_5_762', 'person_41009_1_189',
            'person_41009_2_334', 'person_41009_3_512', 'person_41009_4_247', 'person_41009_5_491', 'person_41009_6_1662',
            'person_41009_7_175', 'person_41009_8_530', 'person_41009_9_378', 'person_41009_10_330', 'person_41010_1_156',
            'person_41010_2_461', 'person_41010_3_160', 'person_41010_4_1127', 'person_41011_1_132', 'person_41011_2_192',
            'person_41011_3_303', 'person_41012_1_871', 'person_41012_2_381', 'person_41012_3_569', 'person_41014_1_452',
            'person_41014_2_600', 'person_41014_3_300', 'person_41014_4_201', 'person_41019_1_554', 'person_41019_2_470',
            'person_41019_3_900', 'person_41019_4_770', 'person_41019_5_564', 'person_41020_1_505', 'person_41021_1_691',
            'person_41021_2_564', 'person_41021_3_641', 'person_41022_1_590', 'person_41022_2_500', 'person_41022_3_450']


class MOTDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(640, 480),
        preproc=None,
        nlow=False,
    ):
        """(608, 1088)
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        # if data_dir is None:
        #     data_dir = os.path.join(get_yolox_datadir(), "UniRTL")RGBT_mix
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "RGBT_mix")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.nlow = nlow

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "i{:04}".format(id_) + ".jpg"
        file_rgb_name = im_ann["file_name_rgb"] if "file_name_rgb" in im_ann else "i{:04}".format(id_) + ".jpg"
        if "file_name_low" in im_ann:
            file_low_name = im_ann["file_name_low"]
        else:
            file_low_name = file_rgb_name

        # # file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        # file_low_name = file_name.replace('IR/i', 'low/l')   # file_name.replace('IR/i', 'low/l')
        # file_rgb_name = file_name.replace('IR/i', 'rgb/v') # file_name.replace('IR/i', 'rgb/v')
        # print(file_low_name,file_rgb_name)
        img_info = (height, width, frame_id, video_id, file_name, file_low_name, file_rgb_name)

        del im_ann, annotations

        return (res, img_info, file_name, file_low_name, file_rgb_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name, file_low_name, file_rgb_name= self.annotations[index]

        # load image and preprocess
        imgi_file = os.path.join(self.data_dir, file_name)
        imgl_file = os.path.join(self.data_dir, file_low_name)
        imgv_file = os.path.join(self.data_dir, file_rgb_name)
        imgi = cv2.imread(imgi_file)
        imgl = cv2.imread(imgl_file)
        imgv = cv2.imread(imgv_file)

        assert imgi is not None
        assert imgl is not None
        assert imgv is not None

        return imgi, imgl, imgv, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        imgi, imgl, imgv, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            imgi, imgl, imgv, target = self.preproc(imgi, imgl, imgv, target, self.input_dim)
        return imgi, imgl, imgv, target, img_info, img_id
