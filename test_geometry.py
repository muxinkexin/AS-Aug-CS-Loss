import unittest

import torch
import torch.nn.functional as F

from layers import (apply_affine, cross_scale_loss, erode_valid_mask,
                                  pixel_to_grid, view_affine)
from datasets.scale_views import make_scale_views


class GeometryTests(unittest.TestCase):
    def test_half_pixel_resize_crop_mapping(self):
        affine = view_affine(7, 9, 2, 3, base_height=4, base_width=4)
        mapped = apply_affine(torch.tensor([[0., 0.], [2., 3.]], dtype=torch.float64), affine)
        expected = torch.tensor([[-2.375, -1.625], [2.125, 3.625]], dtype=torch.float64)
        torch.testing.assert_close(mapped, expected, atol=1e-12, rtol=0)

    def test_projection_affine_covariance(self):
        dtype = torch.float64
        k = torch.tensor([[4., 0., 1.3], [0., 5., 2.1], [0., 0., 1.]], dtype=dtype)
        affine = view_affine(1.4, 0.8, 2, 1)
        point = torch.tensor([0.2, -0.1, 2.0], dtype=dtype)
        projected = k @ point
        lhs = affine @ projected
        rhs = (affine @ k) @ point
        torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=0)

    def test_align_corners_false_grid(self):
        image = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
        coords = torch.tensor([[[[2., 1.]]]])
        sampled = F.grid_sample(image, pixel_to_grid(coords, 4, 5), align_corners=False)
        self.assertEqual(float(sampled.item()), 7.0)

    def test_mask_erodes_seam(self):
        mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
        mask[:, :, :, 3] = False
        eroded = erode_valid_mask(mask, 1)
        self.assertFalse(eroded[:, :, :, 2:5].any())

    def test_cross_scale_empty_and_per_sample_reduction(self):
        class ZeroSSIM(torch.nn.Module):
            def forward(self, a, b):
                return torch.zeros_like(a)

        b, h, w = 2, 8, 8
        mid = torch.ones(b, 1, h, w, requires_grad=True)
        low = torch.stack((torch.ones(1, h, w), torch.full((1, h, w), 3.)), 0).requires_grad_()
        high = low.detach().clone().requires_grad_()
        identity = torch.eye(3, dtype=torch.float64).repeat(b, 1, 1)
        valid = torch.zeros(b, 1, h, w, dtype=torch.bool)
        zero, none = cross_scale_loss(low, mid, high, identity, identity,
                                      valid, valid, valid, ZeroSSIM())
        self.assertEqual(zero.item(), 0.0)
        self.assertFalse(none.any())
        mid_data = mid.detach().clone()
        low_data = low.detach().clone()
        high_data = high.detach().clone()
        mask = torch.ones_like(valid)
        loss, populated = cross_scale_loss(low_data, mid_data, high_data, identity,
                                           identity, mask, mask, mask, ZeroSSIM())
        self.assertTrue(populated.all())
        self.assertAlmostEqual(loss.item(), 0.15, places=5)

    def test_scale_views_fixed_canvas_and_metadata(self):
        image = torch.arange(3 * 10 * 14, dtype=torch.float32).reshape(3, 10, 14)
        result = make_scale_views(image, 8, 12, 0.75, 1.5,
                                  low_crop=(0, 0), high_crop=(2, 3))
        self.assertEqual(tuple(result["views"]["L"].shape), (3, 8, 12))
        self.assertEqual(tuple(result["views"]["H"].shape), (3, 8, 12))
        self.assertEqual(tuple(result["valid_mask"]["L"].shape), (1, 8, 12))
        self.assertEqual(int(result["valid_mask"]["L"].sum()), 6 * 9)
        self.assertEqual(result["metadata"]["H"]["resize"], [12, 18])

    def test_cross_scale_gradients_reach_both_predictions(self):
        class ZeroSSIM(torch.nn.Module):
            def forward(self, a, b):
                return torch.zeros_like(a)

        height = width = 10
        low = torch.full((1, 1, height, width), 2.0, requires_grad=True)
        middle = torch.ones(1, 1, height, width, requires_grad=True)
        high = torch.full((1, 1, height, width), 3.0, requires_grad=True)
        identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        mask = torch.ones(1, 1, height, width, dtype=torch.bool)
        loss, valid = cross_scale_loss(low, middle, high, identity, identity,
                                       mask, mask, mask, ZeroSSIM())
        loss.backward()
        self.assertTrue(valid.all())
        for prediction in (low, middle, high):
            self.assertIsNotNone(prediction.grad)
            self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_batch_specific_affines_do_not_broadcast_masks(self):
        class ZeroSSIM(torch.nn.Module):
            def forward(self, a, b):
                return torch.zeros_like(a)

        height = width = 12
        middle = torch.ones(2, 1, height, width)
        low = middle.clone()
        high = middle.clone()
        low[1] *= 2.0
        high[1] *= 2.0
        affine_l = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        affine_h = affine_l.clone()
        affine_l[1, 0, 2] = -1.0
        affine_h[1, 1, 2] = -1.0
        mask_l = torch.ones(2, 1, height, width, dtype=torch.bool)
        mask_h = mask_l.clone()
        mask_l[1, :, :, :3] = False
        mask_h[1, :, :3, :] = False
        mask_m = torch.ones_like(mask_l)
        loss_batch, valid_batch = cross_scale_loss(
            low, middle, high, affine_l, affine_h, mask_l, mask_m, mask_h, ZeroSSIM())
        loss_0, valid_0 = cross_scale_loss(
            low[:1], middle[:1], high[:1], affine_l[:1], affine_h[:1],
            mask_l[:1], mask_m[:1], mask_h[:1], ZeroSSIM())
        loss_1, valid_1 = cross_scale_loss(
            low[1:], middle[1:], high[1:], affine_l[1:], affine_h[1:],
            mask_l[1:], mask_m[1:], mask_h[1:], ZeroSSIM())
        expected = (loss_0 * valid_0.sum() + loss_1 * valid_1.sum()) / \
            (valid_0.sum() + valid_1.sum())
        torch.testing.assert_close(loss_batch, expected)
        self.assertEqual(tuple(valid_batch.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
