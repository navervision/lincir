# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset
import os

from PIL import Image
import json
import torch
import numpy as np

#import config as cfg

DILATION = 0.7
PAD_CROP = True

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class VAWDataset(Dataset):

    def __init__(self, transform=None, image_dir=None) -> None:
        super().__init__()

        self.image_dir = image_dir
        self.transform = transform
        self.dilate = DILATION
        self.pad_crop = PAD_CROP

    def load_cropped_image(self, img):

        image_id = img['image_id']
        bbox = img['instance_bbox']
        
        # Get image
        path = os.path.join(self.image_dir, f'{image_id}.jpg')
        im = Image.open(path)
        im_width, im_height = im.size

        width = bbox[2]     # Width of bounding box
        height = bbox[3]    # Height of bounding box


        if self.dilate:
            orig_left, orig_top = bbox[0], bbox[1]
            left, top = max(0, orig_left - self.dilate * width), max(0, orig_top - self.dilate * height)
            right, bottom = min(im_width, left + (1 + self.dilate) * width), min(im_height, top + (1 + self.dilate) * height)
        else:
            left, top = bbox[0], bbox[1]
            right, bottom = bbox[0] + width, bbox[1] + height

        im = im.crop((left, top, right, bottom))
        
        if self.pad_crop:
            if im.mode == 'L':
                bg_color = (0,)
            else:
                bg_color = (0, 0, 0)
            im = expand2square(im, bg_color)

        if self.transform is not None:
            im = self.transform(im, return_tensors='pt')['pixel_values'][0]

        return im

class VAWValSubset(VAWDataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
            (Except for returning reference object at the end)
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_cropped_image(i) for i in (reference, target)]
        gallery = [self.load_cropped_image(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            tokenized_caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, tokenized_caption, gallery_and_target, 0, caption

    def __len__(self):
        return len(self.val_samples)
