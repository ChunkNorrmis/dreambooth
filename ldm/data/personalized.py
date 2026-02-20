import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image, ImageFilter
from PIL.ImageEnhance import Sharpness as Sharpen
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path
from captionizer import find_images
import random

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=512,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=True,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 token_only=False,
                 reg=False
                 ):

        super().__init__()
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self._length * repeats

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        
        self.chance = flip_p
        self.reg = reg
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])


    def __len__(self):
        return self._length

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
            h, w = img.shape[0], img.shape[1]
            crop = min(h, w)
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]
            image = Image.fromarray(img)

        if image.width != self.size or image.height != self.size:
            image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)

        if self.chance > random.random():
            image = random.choice([
                image.transpose(random.randrange(0, 2)),
                image.transpose(random.randrange(2, 5)),
                Sharpen(image).enhance(random.uniform(0.5, 2.0))
            ])
            
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["image"] = image
        return example
