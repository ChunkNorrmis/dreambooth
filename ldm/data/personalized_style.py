import os
import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageEnhance
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
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=True,
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
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.chance = flip_p


    def __len__(self):
        return self._length

    def augment(self, image):
        return random.choice([
            image.transpose(Image.Transpose.ROTATE_180),
            image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            image.transpose(Image.Transpose.ROTATE_90),
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            image.transpose(Image.Transpose.ROTATE_270),
            ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 2.0))
        ])

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
            img = np.asarray(image).astype(np.uint8)
            h, w = img.shape[0], img.shape[1]
            crop = min(h, w)
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]
            image = Image.fromarray(img)

        if image.width != self.size or image.height != self.size:
            image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)

        if self.chance > random.random():
            image = self.augment(image)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
