# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,
            scale_training=self.opt.scale_training,
            scale_low_range=self.opt.scale_low_range,
            scale_high_range=self.opt.scale_high_range)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
            scale_training=self.opt.scale_training,
            scale_low_range=self.opt.scale_low_range,
            scale_high_range=self.opt.scale_high_range)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        if self.opt.scale_training == "ra_v1":
            self.scale_ssim = self.ssim if not self.opt.no_ssim else SSIM().to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            if getattr(self.opt, "max_updates", None) is not None and \
                    self.step >= self.opt.max_updates:
                break

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            if getattr(self.opt, "max_updates", None) is not None and \
                    self.step >= self.opt.max_updates:
                break

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

            if getattr(self.opt, "max_updates", None) is not None and \
                    self.step >= self.opt.max_updates:
                break

        # Modern PyTorch expects scheduler updates after optimizer updates.
        # Step once after each completed epoch: use the initial LR for 15 epochs.
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.scale_training == "ra_v1":
            return self.process_scale_batch(inputs)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def process_scale_batch(self, inputs):
        """Run shared DepthNet on L/M/H views while preserving the B0 path."""
        if self.opt.pose_model_type != "separate_resnet" or self.opt.use_stereo:
            raise ValueError("ra_v1 currently supports monocular separate_resnet pose only")
        if self.opt.disable_automasking or self.opt.predictive_mask:
            raise ValueError("ra_v1 requires Monodepth2 automasking and does not support predictive masks")

        features_m = self.models["encoder"](inputs[("scale_color_aug", "M", 0)])
        outputs = self.models["depth"](features_m)
        pose_inputs = {("color_aug", frame_id, 0): inputs[("scale_color_aug", "M", frame_id)]
                       for frame_id in self.opt.frame_ids}
        outputs.update(self.predict_poses(pose_inputs, features_m))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        loss_m = losses["loss"]
        outputs["identity_selection/M"] = outputs["identity_selection/0"]

        for view_id in ("L", "H"):
            features = self.models["encoder"](inputs[("scale_color_aug", view_id, 0)])
            view_outputs = self.models["depth"](features)
            for scale in self.opt.scales:
                outputs[("disp", view_id, scale)] = view_outputs[("disp", scale)]
            self.generate_scale_images(inputs, outputs, view_id, outputs)

        loss_l, photo_l, smooth_l, automask_l, geometry_l, mask_l = \
            self.compute_scale_view_loss(inputs, outputs, "L")
        loss_h, photo_h, smooth_h, automask_h, geometry_h, mask_h = \
            self.compute_scale_view_loss(inputs, outputs, "H")
        if self.opt.cross_scale_weight > 0:
            cs_sources = [outputs[("disp", "L", self.opt.cross_scale_output)],
                          outputs[("disp", self.opt.cross_scale_output)],
                          outputs[("disp", "H", self.opt.cross_scale_output)]]
            cs_predictions = [F.interpolate(source, [self.opt.height, self.opt.width],
                                             mode="bilinear", align_corners=False)
                              for source in cs_sources]
            loss_cs, cs_pairs = cross_scale_loss(
                cs_predictions[0], cs_predictions[1], cs_predictions[2],
                inputs[("scale_A", "L")], inputs[("scale_A", "H")],
                inputs[("scale_valid", "L")], inputs[("scale_valid", "M")],
                inputs[("scale_valid", "H")], self.scale_ssim,
                reliable_mask=outputs["identity_selection/M"].bool())
        else:
            loss_cs = outputs[("disp", 0)].sum() * 0.0
            cs_pairs = torch.zeros((), device=self.device)
        losses["loss_M"] = loss_m
        losses["photo_L"] = photo_l
        losses["smooth_L"] = smooth_l
        losses["photo_H"] = photo_h
        losses["smooth_H"] = smooth_h
        losses["loss_cs"] = loss_cs
        losses["cs_valid_pairs"] = cs_pairs.float().sum()
        losses["valid_L"] = mask_l.float().mean()
        losses["valid_H"] = mask_h.float().mean()
        losses["automask_L"] = automask_l
        losses["automask_H"] = automask_h
        losses["geometry_L"] = geometry_l
        losses["geometry_H"] = geometry_h
        losses["loss"] = (loss_m + loss_l + loss_h) / 3.0 + \
            self.opt.cross_scale_weight * loss_cs
        return outputs, losses

    def _scale_backproject(self, depth, inv_K):
        batch, _, height, width = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij")
        pixels = torch.stack((xx.reshape(-1), yy.reshape(-1),
                              torch.ones(height * width, device=depth.device,
                                         dtype=depth.dtype)), 0).unsqueeze(0).expand(batch, -1, -1)
        cam = torch.bmm(inv_K[:, :3, :3], pixels) * depth.reshape(batch, 1, -1)
        return torch.cat((cam, torch.ones(batch, 1, height * width, device=depth.device,
                                          dtype=depth.dtype)), 1)

    def _scale_project(self, points, K, T):
        batch, _, _ = points.shape
        height, width = self.opt.height, self.opt.width
        projected = torch.bmm(torch.bmm(K, T[:, :3, :]), points)
        raw_z = projected[:, 2:3]
        pixels = (projected[:, :2] / raw_z.clamp_min(1e-7)).reshape(
            batch, 2, height, width).permute(0, 2, 3, 1)
        grid = pixels.clone()
        grid[..., 0] = (grid[..., 0] / (width - 1) - 0.5) * 2
        grid[..., 1] = (grid[..., 1] / (height - 1) - 0.5) * 2
        return grid, raw_z.reshape(batch, 1, height, width)

    def generate_scale_images(self, inputs, outputs, view_id, pose_outputs):
        for scale in self.opt.scales:
            disp = F.interpolate(outputs[("disp", view_id, scale)],
                                 [self.opt.height, self.opt.width], mode="bilinear",
                                 align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", view_id, scale)] = depth
            points = self._scale_backproject(
                depth, inputs[("scale_inv_K", view_id)])
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                grid, z = self._scale_project(
                    points, inputs[("scale_K", view_id)],
                    pose_outputs[("cam_T_cam", 0, frame_id)])
                source = inputs[("scale_color", view_id, frame_id)]
                outputs[("scale_color", view_id, frame_id, scale)] = F.grid_sample(
                    source, grid, padding_mode="border", align_corners=True)
                source_valid = erode_valid_mask(inputs[("scale_valid", view_id)].bool(), 1).float()
                sampled_valid = F.grid_sample(source_valid, grid, mode="nearest",
                                              padding_mode="zeros", align_corners=True) > 0.5
                outputs[("scale_valid", view_id, frame_id, scale)] = (
                    (grid.abs() <= 1).all(-1, keepdim=True).permute(0, 3, 1, 2) &
                    (z > 0) & inputs[("scale_valid", view_id)].bool() & sampled_valid)

    def _masked_smooth_loss(self, disp, image, mask):
        count = mask.sum((2, 3), keepdim=True).clamp_min(1)
        mean = (disp * mask).sum((2, 3), keepdim=True) / count
        norm = disp / (mean + 1e-7)
        dx = (norm[:, :, :, :-1] - norm[:, :, :, 1:]).abs()
        dy = (norm[:, :, :-1, :] - norm[:, :, 1:, :]).abs()
        ix = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(1, True)
        iy = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(1, True)
        mx = mask[:, :, :, :-1] & mask[:, :, :, 1:]
        my = mask[:, :, :-1, :] & mask[:, :, 1:, :]
        return ((dx * torch.exp(-ix) * mx).sum() / mx.sum().clamp_min(1) +
                (dy * torch.exp(-iy) * my).sum() / my.sum().clamp_min(1))

    def compute_scale_view_loss(self, inputs, outputs, view_id):
        mask = inputs[("scale_valid", view_id)].bool()
        per_scale, photo_scales, smooth_scales = [], [], []
        automask_scales, geometry_scales = [], []
        for scale in self.opt.scales:
            target = inputs[("scale_color", view_id, 0)]
            candidates, candidate_masks = [], []
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                pred = outputs[("scale_color", view_id, frame_id, scale)]
                cost = self.compute_reprojection_loss(pred, target)
                valid = outputs[("scale_valid", view_id, frame_id, scale)]
                candidates.append(torch.where(valid, cost, torch.full_like(cost, 1e3)))
                candidate_masks.append(valid)
            reprojection = torch.cat(candidates, 1)
            identities = torch.cat([
                self.compute_reprojection_loss(inputs[("scale_color", view_id, frame_id)], target)
                for frame_id in self.opt.frame_ids[1:] if frame_id != "s"], 1)
            identities = identities + torch.randn_like(identities) * 1e-5
            combined = torch.cat((identities, reprojection), 1)
            best, indices = torch.min(combined, dim=1, keepdim=True)
            reliable = indices >= identities.shape[1]
            valid = erode_valid_mask(mask, 1) & (best < 999.0) & reliable
            outputs[("identity_selection", view_id, scale)] = reliable.float()
            photo = best.masked_select(valid).mean() if valid.any() else best.sum() * 0
            disp = outputs[("disp", view_id, scale)]
            smooth_image = F.interpolate(target, size=disp.shape[-2:], mode="bilinear",
                                          align_corners=False)
            smooth_mask = F.adaptive_avg_pool2d(mask.float(), disp.shape[-2:]) >= 1.0
            smooth = self._masked_smooth_loss(disp, smooth_image, smooth_mask)
            smooth = self.opt.disparity_smoothness * smooth / (2 ** scale)
            loss = photo + smooth
            per_scale.append(loss)
            photo_scales.append(photo)
            smooth_scales.append(smooth)
            automask_scales.append(reliable.float().mean())
            geometry_scales.append(torch.cat(candidate_masks, 1).any(1, keepdim=True).float().mean())
        return (torch.stack(per_scale).mean(), torch.stack(photo_scales).mean(),
                torch.stack(smooth_scales).mean(), torch.stack(automask_scales).mean(),
                torch.stack(geometry_scales).mean(), mask)

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    # Project3D normalizes pixels by width/height minus one.
                    # This matches the original PyTorch sampling convention.
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device, weights_only=False)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device,
                                        weights_only=False)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
