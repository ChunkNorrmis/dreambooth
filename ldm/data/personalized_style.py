import os
import numpy as np
from PIL import Image
from PIL.ImageEnhance import Sharpness as sharpen
from torch.utils.data import Dataset
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
    def __init__(
        self,
        data_root,
        size,
        repeats,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        per_image_tokens=False,
        center_crop=False,
    ):

        super().__init__()
        self.data_root = data_root
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.flip_p = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {"nearest": 0,
                              "bilinear": 2,
                              "bicubic": 3,
                              "lanczos": 1,
                              }[interpolation]


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(self.placeholder_token)

        example["caption"] = text

        if self.center_crop and image.width != image.height:
            im = np.array(image).astype(np.uint8)
            H, W = im.shape[0], im.shape[1]
            crop = min(W, H)
            im = im[(H - crop) // 2: (H + crop) // 2, (W - crop) // 2: (W + crop) // 2]
            image = Image.fromarray(im)

        if self.size is not None:
            if image.width > self.size or image.height > self.size:
                image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)

        if random.random() < self.flip_p:
            image = random.choice([
                image.transpose(method=random.randint(0, 4)),
                sharpen(image).enhance(random.uniform(-1.0, 2.0))
            ])

        img = np.array(image).astype(np.uint8)
        img = (img / 127.5 - 1).astype(np.float32)
        example['image'] = img

        return example
