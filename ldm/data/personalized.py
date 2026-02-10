import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path
from captionizer import find_images
from random import choice, random
from PIL.ImageEnhance import Sharpness as sharpen



per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="lanczos",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 token_only=False,
                 reg=False
                 ):

        self.data_root = data_root

        self.image_paths = find_images(self.data_root)

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.odds = flip_p

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.reg = reg
        self.interp = {
            'bilinear': PIL.Image.BILINEAR,
            'bicubic': PIL.Image.BICUBIC,
            'nearest': PIL.Image.NEAREST,
            'lanczos': PIL.Image.LANCZOS
        }[interpolation]

        self.augment = {
            'direction': {
                'h_flip': Transpose.FLIP_LEFT_RIGHT,
                'v_flip': Transpose.FLIP_TOP_BOTTOM,
                '90_degree': Transpose.ROTATE_90,
                '180_degree': Transpose.ROTATE_180,
                '270_degree': Transpose.ROTATE_270
            },
            'clarity': {
                'sharpen': random() + 1.0,
                'blur': random() - 1.0
            },
        }

        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])


    def __len__(self):
        return self._length


    def chance(self):
        return f"{random():.2f}"


    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["caption"] = ""
        if self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

        if self.center_crop and image.width != image.height:
            img = np.array(image).astype(np.uint8)
            H, W = img.shape[0], img.shape[1]
            _max = min(H, W)
            img = img[(H - _max) // 2:(H + _max) // 2, (W - _max) // 2:(W + _max) // 2]
            image = Image.fromarray(img)

        if image.width > self.size or image.height > self.size:
            image = image.resize((self.size, self.size), resample=self.interp, reducing_gap=3)

        if self.chance() >= self.odds:
            direction = choice(['h_flip', 'v_flip', '90_degree', '180_degree', '270_degree'])
            image = image.transpose(self.augment['direction'][direction])
                
        if self.chance() >= self.odds:
            clarity = choice(['sharpen', 'blur'])
            image = sharpen(image).enhance(self.augment['clarity'][clarity])

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
