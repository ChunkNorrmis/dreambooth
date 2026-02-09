import os
import numpy as np
import PIL
from PIL.ImageEnhance import Sharpness as sharpen
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_small = [
    'a painting in the style of {}',
    'a rendering in the style of {}',
    'a cropped painting in the style of {}',
    'the painting in the style of {}',
    'a clean painting in the style of {}',
    'a dirty painting in the style of {}',
    'a dark painting in the style of {}',
    'a picture in the style of {}',
    'a cool painting in the style of {}',
    'a close-up painting in the style of {}',
    'a bright painting in the style of {}',
    'a cropped painting in the style of {}',
    'a good painting in the style of {}',
    'a close-up painting in the style of {}',
    'a rendition in the style of {}',
    'a nice painting in the style of {}',
    'a small painting in the style of {}',
    'a weird painting in the style of {}',
    'a large painting in the style of {}',
]

imagenet_dual_templates_small = [
    'a painting in the style of {} with {}',
    'a rendering in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'the painting in the style of {} with {}',
    'a clean painting in the style of {} with {}',
    'a dirty painting in the style of {} with {}',
    'a dark painting in the style of {} with {}',
    'a cool painting in the style of {} with {}',
    'a close-up painting in the style of {} with {}',
    'a bright painting in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'a good painting in the style of {} with {}',
    'a painting of one {} in the style of {}',
    'a nice painting in the style of {} with {}',
    'a small painting in the style of {} with {}',
    'a weird painting in the style of {} with {}',
    'a large painting in the style of {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 repeats,
                 resampler='lanczos',
                 flip_p=0.5,
                 set="train",
                 placeholder_token=None,
                 per_image_tokens=False,
                 center_crop=False,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.inter = {'bilinear': Resampling.BILINEAR,
                              'bicubic': Resampling.BICUBIC,
                              'nearest': Resampling.NEAREST,
                              'lanczos': Resampling.LANCZOS
                              }[resampler]
        self.aug = flip_p * 10

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        im_path = self.image_paths[i % self.num_images]
        image = Image.open(im_path, 'r')

        image = image.mode('RGB')

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(self.placeholder_token)
        example["caption"] = text
        
        image = np.array(image).astype(np.uint8)
        
        if self.center_crop and image.shape[0] != image.shape[1]:
            H, W = image.shape[0], image.shape[1]
            _max = min(H, W)
            image = image[(H - _max) // 2:(H + _max) // 2, (W - _max) // 2:(W + _max) // 2]

        image = Image.fromarray(image)

        if self.size is not None and image.width > self.size:
            image = image.resize((self.size, self.size), resample=self.inter, reducing_gap=3)
                
        if randint(0, 9) >= self.aug:
            fl = {0: Transpose.FLIP_LEFT_RIGHT, 1: Transpose.TOP_TO_BOTTOM}
            image = image.transpose(method=fl[choice([0, 1])])
                        
        if randint(0, 9) >= self.aug:
            image = image.sharpen(image).enhance(choice([1.40, 0.65]))
            
        if randint(0, 9) >= self.aug:
            image = image.rotate(angle=float(randint(0, 45)), resampling=self.inter)
        
        image = np.array(image).astype(np.uint8)
        
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)        
        return example
