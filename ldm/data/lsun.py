import os
import numpy as np
from PIL import Image, ImageEngance
from torch.utils.data import Dataset
from torchvision import transforms
import random


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
        self.flip_p = flip_p
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def augment(self, image) -> Image.Image:
        if random.random() <= self.flip_p:
            rando = [
                image.transpose(method=random.choice([Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM])),
                image.rotate(random.choice([90, 180, 270])),
                ImageEnhance.Sharpness(image).enhance(random.choice([random.random()-1.0, random.random()+1.0]))
            ]
            choice = random.choice(rando)
            image = choice
            data = image.getdata()
            im = Image.new(mode=image.mode, size=image.size)
            im = im.putdata(data)
            return im
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
    
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
        
        image = self.augment(image)

        img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        img = np.asarray(image)
        img = (img / 127.5 - 1).astype(np.float32)
        example['image'] = img
        
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
