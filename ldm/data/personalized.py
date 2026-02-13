import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image
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
                 size,
                 repeats,
                 interpolation="bicubic",
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

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"nearest": 0,
                              "bilinear": 2,
                              "bicubic": 3,
                              "lanczos": 1,
                              }[interpolation]

        self.reg = reg
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

        self.aug = random.choice([
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.RandomPerspective(distortion_scale=0.5, p=flip_p ,interpolation=self.interpolation, fill=0),
            self.random_sharpen(p=flip_p)
        ])
                     
                                                                                                                                                                                                                                                                        
    def random_sharpen(self, image=image, p=None):
        if random.random() <= p:
            sharpness = random.choice([random.random() - 1.0, random.random() + 1.0])
            return Sharpen(image).enhance(sharpness)
    
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


        if self.center_crop:
            img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            img = np.asarray(image)
            H, W = img.shape[0], img.shape[1]
            crop = min(W, H)
            img = img[(H - crop) // 2: (H + crop) // 2,
                      (W - crop) // 2: (W + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)
        image = self.aug(image)

        img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        img = np.asarray(image)
        img = (img / 127.5 - 1).astype(np.float32)
        example['image'] = img
        
        return example
