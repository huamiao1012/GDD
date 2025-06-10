
from mmdet.registry import TRANSFORMS
import torch
from mmcv.transforms import BaseTransform
import numpy as np
# from tools import shuffle_data
# from preprocessing.Datasets import normalize_dataset
 
 
 
@TRANSFORMS.register_module()
class GenerateMaskFromBbox:
    def __init__(self,
                 img_size : tuple):
        self.img_shape = img_size
               
    def __call__(self, results):
        img_shape = self.img_shape
        bboxes = results['gt_bboxes'].tensor.numpy().astype(int)
        mask = np.zeros(img_shape, dtype=np.uint8)
        for box in bboxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
        results['gt_mask_bbox'] = mask[None, ...]  # 增加 channel 维度
        return results
