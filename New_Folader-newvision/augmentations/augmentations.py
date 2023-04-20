import math
import numbers
import random
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms as transforms
import numpy as np
import mmcv
from typing import Dict, List, Optional, Sequence, Tuple, Union

class Compose(object):
    def __init__(self, augmentations):     
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, imgs, mask=None, disp = None):   
        assert ( isinstance(imgs, list))
        imgs_ = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, mode="RGB")
                if mask is not None:
                    mask=mask.cpu().numpy()
                    mask = Image.fromarray(mask,mode="L")
            imgs_.append(img)
            self.PIL2Numpy = True
            if mask is not None:
                assert img.size == mask.size
            depth_map_scaled = np.clip(disp * (255 / np.max(disp)), 0, 255)

            # 将深度图转换为8位无符号整数类型的numpy数组
            depth_map_uint8 = depth_map_scaled.astype(np.uint8)

            # 将numpy数组转换为PIL图像对象
            disp = Image.fromarray(depth_map_uint8)
        for a in self.augmentations:       
            imgs_, mask, disp = a(imgs_, mask, disp)
        return imgs_, mask, disp
    
# class RandomFlip(object):
#     def __init__(self, p, direction):
#         if isinstance(p, float):
#             assert 0<= p <= 1
#         elif isinstance(p, list):
#             for i in p:
#                 assert 0<= i <= 1
#             assert 0<= sum(p) <= 1
        
#         self.prob = p
#         valid_directions= ['horizontal','vertical','diagonal']
#         if isinstance(direction, str):
#             assert direction in valid_directions
        
#         elif isinstance(direction, list):
#             for i in direction:
#                 assert direction in valid_directions
#             assert set(direction).issubset(set(valid_directions))

#         self.direction = direction

#         if isinstance (p, list):
#             assert len(p) == len(self.direction)
    
#     # def _choose_direction(self):

#     def __call__(self, imgs, mask):
#         assert (isinstance(imgs, list))
#         '情况1：Self.p  单一， direction 单一'
#         '情况2： Self.p 列表   direction 列表'
#         '情况三：Self.p 单一   direction 列表'
#         imgs_ = []
#         for (idx, img) in enumerate(imgs):
#             mask_ = mask
#             if idx==0:
#                 pro = random.random()       
#             if pro < self.p:
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)
#                 mask_ = mask_.transpose(Image.FLIP_LEFT_RIGHT)
#             if pro < self.p
#                 img = img.transpose(Image.FLIP_TOP_BOTTOM)
#                 mask_ = mask.transpose(Image.FLIP_TOP_BOTTOM)
#             imgs_.append(img)
#         return imgs_,mask_
class RandomFlip(object):         
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, mask, disp):

        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            mask_ = mask
            disp_ = disp
            if idx==0:
                pro = random.random()       
            if pro < self.p:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_ = mask_.transpose(Image.FLIP_LEFT_RIGHT)
                disp_ = disp_.transpose(Image.FLIP_LEFT_RIGHT)
            imgs_.append(img)

        return imgs_, mask_, disp_

class RandomCrop(object):      
    def __init__(self, crop_size, cat_max_ratio=1,ignore_index=255):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def __call__(self, imgs, mask, disp):
        assert ( isinstance(imgs, list))
        imgs_ = []
        def generate_crop_bbox(img: np.ndarray) -> tuple:
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        for (idx, img) in enumerate(imgs):
            
            mask_ = mask
            disp_ = disp
            if idx ==0 :
                crop_y1, crop_y2, crop_x1, crop_x2 = generate_crop_bbox(img):
                if self.cat_max_ratio < 1.:
                # Repeat 10 times
                    for _ in range(10):
                        seg_temp = mask_[crop_y1:crop_y2, crop_x1:crop_x2]
                        labels, cnt = np.unique(seg_temp, return_counts=True)
                        cnt = cnt[labels != self.ignore_index]
                        if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                            break
                        crop_y1, crop_y2, crop_x1, crop_x2 = generate_crop_bbox(img)
            
            img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            mask_[crop_y1:crop_y2, crop_x1:crop_x2]
            disp_[crop_y1:crop_y2, crop_x1:crop_x2]
            imgs_.append(img)
        return imgs_, mask_, disp
    
class RandomResize(object):
    def __init__(self, scale,  ratio_range):
        self.scale = scale
        self.ratio_range=  ratio_range
    def __call__(self, imgs, mask, disp):
        mask_=mask
        disp_=disp
        min_ratio, max_ratio = self.ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            img = img.resize(scale,Image.BILINEAR)
            mask = mask_.resize(scale,Image.BILINEAR)
            disp = disp_.resize(scale,Image.BILINEAR)
            imgs_.append(img)
        return imgs_, mask, disp


class PhotoMetricDistortion(object):    
    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, imgs, mask, disp):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            img = self.brightness(img)
            if idx == 0:
                mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)
            img = self.hue(img)

            if mode == 0:
                img = self.contrast(img)
            imgs_.append(img)
        return imgs_, mask, disp
