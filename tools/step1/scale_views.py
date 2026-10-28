"""Fixed-canvas L/M/H view construction for one image triplet."""

import torch
import torch.nn.functional as F

from .geometry import view_affine


def _resize(image, height, width):
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear",
                         align_corners=False).squeeze(0)


def _crop_or_repeat(image, top, left, height, width):
    """Crop a high view or tail-repeat a low view onto the fixed canvas."""
    src_h, src_w = image.shape[-2:]
    if src_h >= height and src_w >= width:
        return image[:, top:top + height, left:left + width], torch.ones(
            1, height, width, dtype=torch.bool, device=image.device)
    canvas = image.new_empty(image.shape[0], height, width)
    ys = torch.arange(height, device=image.device).clamp_max(src_h - 1)
    xs = torch.arange(width, device=image.device).clamp_max(src_w - 1)
    canvas = image[:, ys[:, None], xs[None, :]]
    valid = torch.zeros(1, height, width, dtype=torch.bool, device=image.device)
    valid[:, :src_h, :src_w] = True
    return canvas, valid


def make_scale_views(image, base_height, base_width, low_scale, high_scale,
                     low_crop=(0, 0), high_crop=None, K=None):
    """Create L/M/H tensors and per-view geometric metadata.

    ``image`` is a float tensor ``[C,H,W]``. The returned views all use the
    base canvas while ``valid_mask`` excludes low-view copied tail pixels.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape [C,H,W]")
    if not (0 < low_scale < 1 < high_scale):
        raise ValueError("scales must satisfy 0 < low < 1 < high")
    if tuple(low_crop) != (0, 0):
        raise ValueError("the first implementation places the low view at the canvas origin")
    low_h, low_w = max(1, round(base_height * low_scale)), max(1, round(base_width * low_scale))
    high_h, high_w = max(base_height, round(base_height * high_scale)), max(base_width, round(base_width * high_scale))
    middle = _resize(image, base_height, base_width)
    low = _resize(image, low_h, low_w)
    low_canvas, low_valid = _crop_or_repeat(low, low_crop[0], low_crop[1], base_height, base_width)
    if high_crop is None:
        high_crop = ((high_h - base_height) // 2, (high_w - base_width) // 2)
    high = _resize(image, high_h, high_w)
    high_canvas, high_valid = _crop_or_repeat(high, high_crop[0], high_crop[1], base_height, base_width)
    views = {"L": low_canvas, "M": middle, "H": high_canvas}
    masks = {"L": low_valid, "M": torch.ones_like(low_valid), "H": high_valid}
    metadata = {"L": {"resize": [low_h, low_w], "crop": list(low_crop)},
                "M": {"resize": [base_height, base_width], "crop": [0, 0]},
                "H": {"resize": [high_h, high_w], "crop": list(high_crop)}}
    for name, (rh, rw), crop in (("L", (low_h, low_w), low_crop),
                                  ("M", (base_height, base_width), (0, 0)),
                                  ("H", (high_h, high_w), high_crop)):
        metadata[name]["A"] = view_affine(rh, rw, crop[0], crop[1],
                                            base_height, base_width,
                                            device=image.device, dtype=torch.float64)
        if K is not None:
            metadata[name]["K"] = (metadata[name]["A"] @
                                   K.to(dtype=torch.float64, device=image.device)).to(image.dtype)
    return {"views": views, "valid_mask": masks, "metadata": metadata}
