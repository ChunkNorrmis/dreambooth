import os
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import random


class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=None,
        interpolation="bicubic",
        flip_p=0.5
    ):

        super().__init__()
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.flip_p = flip_p
        self.size = size

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.center_crop and image.width != image.height:
            img = np.array(image).astype(np.uint8)
            H, W = img.shape[0], img.shape[1]
            crop = min(W, H)
            img = img[(H - crop) // 2: (H + crop) // 2, (W - crop) // 2: (W + crop) // 2]
            image = Image.fromarray(img)

        if self.size is not None:
            if image.width > self.size or image.height > self.size:
                image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)

        if random.random() < self.flip_p:
            image = random.choice([
                image.transpose(method=random.randint(0, 4)),
                ImageEnhance.Sharpness(image).enhance(random.uniform(-1.0, 2.0))
            ])

        img = np.array(image).astype(np.uint8)
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
