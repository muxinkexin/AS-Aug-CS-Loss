# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as TF


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 scale_training="off",
                 scale_low_range=(0.7, 0.9),
                 scale_high_range=(1.1, 2.0)):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # Pillow 10 removed the old ``Image.ANTIALIAS`` alias.
        self.interp = getattr(Image, "Resampling", Image).LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.scale_training = scale_training
        self.scale_low_range = tuple(scale_low_range)
        self.scale_high_range = tuple(scale_high_range)

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def get_color_aug(self):
        """Sample one colour transform shared by all frames and scales."""
        params = transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        # Older torchvision returned a callable; current versions return the
        # operation order and four factors. Sample once per item in either case.
        if callable(params):
            return params

        order, brightness, contrast, saturation, hue = params
        operations = (
            (TF.adjust_brightness, brightness),
            (TF.adjust_contrast, contrast),
            (TF.adjust_saturation, saturation),
            (TF.adjust_hue, hue),
        )

        def color_aug(image):
            for index in order:
                operation, factor = operations[int(index)]
                if factor is not None:
                    image = operation(image, factor)
            return image

        return color_aug

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = self.get_color_aug()
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.scale_training == "ra_v1":
            self._add_scale_views(inputs, color_aug, do_color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def _add_scale_views(self, inputs, color_aug, do_color_aug):
        from tools.step1.scale_views import make_scale_views

        if self.is_train:
            low_scale = random.uniform(*self.scale_low_range)
            high_scale = random.uniform(*self.scale_high_range)
        else:
            low_scale = sum(self.scale_low_range) / 2.0
            high_scale = sum(self.scale_high_range) / 2.0

        high_h = max(self.height, round(self.height * high_scale))
        high_w = max(self.width, round(self.width * high_scale))
        max_top, max_left = high_h - self.height, high_w - self.width
        if self.is_train:
            crop_top = random.randint(0, max_top) if max_top else 0
            crop_left = random.randint(0, max_left) if max_left else 0
        else:
            crop_top, crop_left = max_top // 2, max_left // 2

        base_intrinsics = inputs[("K", 0)][:3, :3]
        for frame_id in self.frame_idxs:
            native = inputs[("color", frame_id, -1)]
            raw = native if torch.is_tensor(native) else self.to_tensor(native)
            built = make_scale_views(raw, self.height, self.width, low_scale, high_scale,
                                     high_crop=(crop_top, crop_left), K=base_intrinsics)
            for view_id in ("L", "M", "H"):
                color = inputs[("color", frame_id, 0)] if view_id == "M" else built["views"][view_id]
                if view_id == "M":
                    color_augmented = inputs[("color_aug", frame_id, 0)]
                else:
                    color_augmented = color_aug(color)
                inputs[("scale_color", view_id, frame_id)] = color
                inputs[("scale_color_aug", view_id, frame_id)] = color_augmented
                inputs[("scale_valid", view_id)] = built["valid_mask"][view_id]
                inputs[("scale_A", view_id)] = built["metadata"][view_id]["A"]
                inputs[("scale_K", view_id)] = built["metadata"][view_id]["K"]
                inputs[("scale_inv_K", view_id)] = torch.linalg.pinv(
                    built["metadata"][view_id]["K"])
                inputs[("scale_resize", view_id)] = torch.tensor(
                    built["metadata"][view_id]["resize"], dtype=torch.int64)
                inputs[("scale_crop", view_id)] = torch.tensor(
                    built["metadata"][view_id]["crop"], dtype=torch.int64)
        inputs["scale_factors"] = torch.tensor([low_scale, high_scale], dtype=torch.float32)

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
