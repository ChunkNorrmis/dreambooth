import os
import numpy as np
import PIL
from PIL import Image
from PIL.ImageEnhance import Sharpness as sharpen
from torch.utils.data import Dataset
from torchvision import transforms
from random import random, choice



class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        self.odds = flip_p
        self.size = size
        self.interp = {
            'linear': PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
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

    
    def __len__(self):
        return self._length


    def chance(self):
        return f"{random():.2f}"    

    
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

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


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
