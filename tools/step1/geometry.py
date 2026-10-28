"""Geometry helpers for the valid-aware half-pixel scale-view convention."""

from __future__ import division

import torch
import torch.nn.functional as F


def view_affine(resized_height, resized_width, crop_top, crop_left,
                base_height=1, base_width=1, device=None, dtype=torch.float64):
    """Map base-view pixel centers through resize and crop."""
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError("resized dimensions must be positive")
    sx, sy = resized_width / float(base_width), resized_height / float(base_height)
    return torch.tensor([
        [sx, 0.0, (sx - 1.0) / 2.0 - crop_left],
        [0.0, sy, (sy - 1.0) / 2.0 - crop_top],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)


def apply_affine(coords, affine):
    """Apply a 3x3 pixel-coordinate affine to (..., 2) coordinates."""
    ones = torch.ones_like(coords[..., :1])
    homogeneous = torch.cat((coords, ones), dim=-1)
    if affine.ndim == 3 and homogeneous.ndim == 4:
        return torch.einsum("bhwj,bij->bhwi", homogeneous, affine)[..., :2]
    return torch.matmul(homogeneous, affine.transpose(-1, -2))[..., :2]


def pixel_to_grid(coords, height, width):
    """Convert pixel-center coordinates to grid_sample(align_corners=False)."""
    x = 2.0 * (coords[..., 0] + 0.5) / width - 1.0
    y = 2.0 * (coords[..., 1] + 0.5) / height - 1.0
    return torch.stack((x, y), dim=-1)


def sample_corresponding(image, coords, valid_rect=None):
    """Sample image at pixel coordinates; return values and strict in-bounds mask."""
    height, width = image.shape[-2:]
    inside = ((coords[..., 0] >= 0) & (coords[..., 0] <= width - 1) &
              (coords[..., 1] >= 0) & (coords[..., 1] <= height - 1))
    if valid_rect is not None:
        top, left, rect_height, rect_width = valid_rect
        inside &= ((coords[..., 0] >= left) & (coords[..., 0] <= left + rect_width - 1) &
                   (coords[..., 1] >= top) & (coords[..., 1] <= top + rect_height - 1))
    grid = pixel_to_grid(coords.to(dtype=image.dtype), height, width)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
    return sampled, inside.unsqueeze(1)


def erode_valid_mask(mask, radius=1):
    """Keep pixels whose full square neighborhood is valid."""
    if radius <= 0:
        return mask.bool()
    size = radius * 2 + 1
    padded = F.pad((~mask.bool()).float(), (radius, radius, radius, radius), value=1.0)
    invalid = F.max_pool2d(padded, size, stride=1)
    return invalid == 0


def masked_pair_mean(loss, mask):
    """Return per-sample means and whether each sample has valid pixels."""
    mask = mask.bool()
    count = mask.flatten(1).sum(1)
    values = (loss * mask.to(loss.dtype)).flatten(1).sum(1) / count.clamp_min(1)
    return values, count > 0


def cross_scale_loss(pred_l, pred_m, pred_h, affine_l, affine_h,
                     mask_l, mask_m, mask_h, ssim_loss_fn,
                     reliable_mask=None):
    """Compute masked LM/MH loss, reducing valid image-pairs without batch leakage."""
    if pred_l.shape[0] != pred_m.shape[0] or pred_h.shape[0] != pred_m.shape[0]:
        raise ValueError("all views must have the same batch size")
    batch, _, height, width = pred_m.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=pred_m.device, dtype=affine_l.dtype),
        torch.arange(width, device=pred_m.device, dtype=affine_l.dtype), indexing="ij")
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    terms, valid_terms = [], []
    for pred, affine, source_mask in ((pred_l, affine_l, mask_l),
                                      (pred_h, affine_h, mask_h)):
        coords = apply_affine(base, affine)
        sampled, in_bounds = sample_corresponding(pred, coords)
        sampled_mask, _ = sample_corresponding(source_mask.float(), coords)
        valid = mask_m.bool() & in_bounds & (sampled_mask >= 0.999)
        if reliable_mask is not None:
            reliability = reliable_mask.detach().bool()
            if reliability.ndim == 3:
                reliability = reliability.unsqueeze(1)
            valid &= reliability
        valid = erode_valid_mask(valid, 1)
        l1 = (sampled - pred_m).abs().mean(1, keepdim=True)
        ssim = ssim_loss_fn(sampled, pred_m).mean(1, keepdim=True)
        values, has_pixels = masked_pair_mean(0.85 * ssim + 0.15 * l1, valid)
        terms.append(values)
        valid_terms.append(has_pixels)
    values = torch.stack(terms, dim=1)
    has_pixels = torch.stack(valid_terms, dim=1)
    if not has_pixels.any():
        return (pred_l.sum() + pred_m.sum() + pred_h.sum()) * 0.0, has_pixels
    return values[has_pixels].mean(), has_pixels
