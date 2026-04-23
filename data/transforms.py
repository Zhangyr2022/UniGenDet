# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

try:
    from albumentations import (
        Compose,
        Downscale,
        GaussNoise,
        GaussianBlur,
        ImageCompression,
        MotionBlur,
    )
except Exception:  # pragma: no cover
    Compose = None
    Downscale = None
    GaussNoise = None
    GaussianBlur = None
    ImageCompression = None
    MotionBlur = None


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize image while enforcing min/max edge, stride divisibility, and max-pixel budget."""

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if (
            self.max_pixels is not None
            and new_width * new_height > self.max_pixels / img_num
        ):
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(
            img, (new_height, new_width), self.interpolation, antialias=self.antialias
        )


class ImageTransformTensor:
    """Tensor-only resize and normalization transform."""

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.stride = image_stride
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.max_pixels = max_pixels

        self.image_mean = torch.tensor(image_mean).view(-1, 1, 1)
        self.image_std = torch.tensor(image_std).view(-1, 1, 1)

    def _tensor_resize(self, img_tensor, img_num=1):
        """Resize tensor image with stride and max-pixel constraints."""
        _, h, w = img_tensor.shape

        long_edge = max(h, w)
        short_edge = min(h, w)
        scale = min(
            self.max_image_size / long_edge,
            max(self.min_image_size / short_edge, self.min_image_size / short_edge),
        )

        if self.max_pixels is not None:
            pixel_count = h * w * img_num
            if pixel_count > self.max_pixels:
                import math

                scale = math.sqrt(self.max_pixels / pixel_count) * scale

        new_h = int(round(h * scale / self.stride)) * self.stride
        new_w = int(round(w * scale / self.stride)) * self.stride
        new_h = max(new_h, self.min_image_size)
        new_w = max(new_w, self.min_image_size)

        if (new_h, new_w) != (h, w):
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img_tensor

    def _tensor_normalize(self, img_tensor):
        """Normalize tensor with channel-wise mean and std."""
        mean = self.image_mean.to(img_tensor.device)
        std = self.image_std.to(img_tensor.device)
        return (img_tensor - mean) / std

    def __call__(self, img_tensor, img_num=1):
        """Apply clamp, resize, and normalization to CHW tensor input."""
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        img_tensor = self._tensor_resize(img_tensor, img_num)
        return self._tensor_normalize(img_tensor)


class ImageTransform:
    """PIL-to-tensor transform with resize and normalization."""

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean, std=image_std, inplace=True
        )

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        return self.normalize_transform(img)


def decolorization(image):
    gray_image = image.convert("L")
    return (
        Image.merge(image.mode, [gray_image] * 3)
        if image.mode in ("RGB", "L")
        else gray_image
    )


def downscale(image, scale_factor):
    new_width = max(1, int(round(image.width * scale_factor)))
    new_height = max(1, int(round(image.height * scale_factor)))
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def crop(image, crop_factors):
    target_h, target_w = crop_factors
    img_w, img_h = image.size

    if target_h > img_h or target_w > img_w:
        raise ValueError("Crop size exceeds image dimensions")

    x = random.randint(0, img_w - target_w)
    y = random.randint(0, img_h - target_h)
    return image.crop((x, y, x + target_w, y + target_h)), [
        [x, y],
        [x + target_w, y + target_h],
    ]


def motion_blur_opencv(image, kernel_size=15, angle=0):
    """Apply directional motion blur to a PIL image using OpenCV kernels."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
    m = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, m, (kernel_size, kernel_size))
    rotated_kernel /= rotated_kernel.sum() if rotated_kernel.sum() != 0 else 1

    img = np.array(image)
    if img.ndim == 2:
        blurred = cv2.filter2D(img, -1, rotated_kernel, borderType=cv2.BORDER_REFLECT)
    else:
        blurred = np.zeros_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = cv2.filter2D(
                img[..., c], -1, rotated_kernel, borderType=cv2.BORDER_REFLECT
            )
    return Image.fromarray(blurred)


def shuffle_patch(image, num_splits, gap_size=2):
    """Shuffle image patches on a regular grid and stitch with optional gaps."""
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop(
                (current_x, current_y, current_x + patch_w, current_y + patch_h)
            )
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    random.shuffle(patches)

    total_width = sum(patch_widths) + (w_splits - 1) * gap_size
    total_height = sum(patch_heights) + (h_splits - 1) * gap_size
    new_image = Image.new(
        image.mode, (total_width, total_height), color=(255, 255, 255)
    )

    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch = patches[patch_idx]
            patch_w = patch_widths[j]
            new_image.paste(patch, (current_x, current_y))
            current_x += patch_w + gap_size
            patch_idx += 1
        current_y += patch_h + gap_size

    return new_image


def inpainting(image, num_splits, blank_ratio=0.3, blank_color=(255, 255, 255)):
    """Randomly blank out a ratio of grid patches to simulate inpainting corruption."""
    h_splits, w_splits = num_splits
    img_w, img_h = image.size

    base_patch_h = img_h // h_splits
    patch_heights = [base_patch_h] * (h_splits - 1)
    patch_heights.append(img_h - sum(patch_heights))

    base_patch_w = img_w // w_splits
    patch_widths = [base_patch_w] * (w_splits - 1)
    patch_widths.append(img_w - sum(patch_widths))

    patches = []
    current_y = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch_w = patch_widths[j]
            patch = image.crop(
                (current_x, current_y, current_x + patch_w, current_y + patch_h)
            )
            patches.append(patch)
            current_x += patch_w
        current_y += patch_h

    total_patches = h_splits * w_splits
    num_blank = int(total_patches * blank_ratio)
    num_blank = max(0, min(num_blank, total_patches))
    blank_indices = set(random.sample(range(total_patches), num_blank))

    processed_patches = []
    for idx, patch in enumerate(patches):
        if idx in blank_indices:
            processed_patches.append(Image.new("RGB", patch.size, color=blank_color))
        else:
            processed_patches.append(patch)

    result_image = Image.new("RGB", (img_w, img_h))
    current_y = 0
    patch_idx = 0
    for i in range(h_splits):
        current_x = 0
        patch_h = patch_heights[i]
        for j in range(w_splits):
            patch = processed_patches[patch_idx]
            patch_w = patch_widths[j]
            result_image.paste(patch, (current_x, current_y))
            current_x += patch_w
            patch_idx += 1
        current_y += patch_h

    return result_image


class ImageTransformAug:
    """Create two resolution branches from one image, with optional augmentation."""

    def __init__(
        self,
        image_size=[256, 224],
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        data_aug=True,
    ):
        self.image_size = image_size
        self.data_aug = data_aug and Compose is not None

        if self.data_aug:
            self.aug = Compose(
                [
                    ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                    GaussNoise(p=0.2),
                    MotionBlur(p=0.2),
                    GaussianBlur(blur_limit=3, p=0.5),
                    Downscale(
                        scale_min=0.25,
                        scale_max=0.75,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.25,
                    ),
                ]
            )

        self.base_transform = transforms.Compose(
            [
                transforms.Resize(image_size[0]),
                transforms.RandomCrop(image_size[0]),
            ]
        )
        self.resize_transform = transforms.Resize((image_size[1], image_size[1]))

        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std)

    def _albumentations_aug(self, img):
        img_np = np.array(img)
        augmented = self.aug(image=img_np)
        return Image.fromarray(augmented["image"])

    def __call__(self, img):
        if self.data_aug:
            img = self._albumentations_aug(img)

        img1 = self.base_transform(img)
        img2 = self.resize_transform(img1)

        tensor1 = self.normalize_transform(self.to_tensor_transform(img1))
        tensor2 = self.normalize_transform(self.to_tensor_transform(img2))
        return tensor1, tensor2


class DualImageTransform:
    """Generate two independently resized tensor branches from one input image."""

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        data_aug=True,
    ):
        self.data_aug = data_aug and Compose is not None

        if self.data_aug:
            self.aug = Compose(
                [
                    ImageCompression(quality_lower=60, quality_upper=100, p=0.25),
                    GaussNoise(p=0.2),
                    MotionBlur(p=0.2),
                    GaussianBlur(blur_limit=3, p=0.5),
                    Downscale(
                        scale_min=0.25,
                        scale_max=0.75,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.25,
                    ),
                ]
            )

        self.vae_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size[0],
            min_size=min_image_size[0],
            stride=image_stride[0],
            max_pixels=max_pixels,
        )
        self.vit_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size[1],
            min_size=min_image_size[1],
            stride=image_stride[1],
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std)

    def _albumentations_aug(self, img):
        img_np = np.array(img)
        augmented = self.aug(image=img_np)
        return Image.fromarray(augmented["image"])

    def __call__(self, img):
        aug_img = self._albumentations_aug(img) if self.data_aug else img

        img1 = self.vae_transform(aug_img)
        img2 = self.vit_transform(aug_img)

        tensor1 = self.normalize_transform(self.to_tensor_transform(img1))
        tensor2 = self.normalize_transform(self.to_tensor_transform(img2))
        return tensor1, tensor2
