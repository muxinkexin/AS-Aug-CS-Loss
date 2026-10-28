# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import json
import os
import random
import sys

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, "r") as stream:
        config = json.load(stream)
    options_reference = config.get("resolved_train_options")
    required = ["experiment_id", "baseline_manifest", "resolved_train_options", "seed",
                "scale_training", "scale_low_range", "scale_high_range",
                "cross_scale_weight", "scale_geometry_mode", "output_dir",
                "eval_resolutions"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("missing config keys: {}".format(", ".join(missing)))
    if config["scale_training"] not in ("off", "ra_v1"):
        raise ValueError("scale_training must be off or ra_v1")
    low, high = config["scale_low_range"], config["scale_high_range"]
    if len(low) != 2 or len(high) != 2 or not (0 < low[0] <= low[1] < 1 < high[0] <= high[1]):
        raise ValueError("scale ranges must satisfy 0 < low_min <= low_max < 1 < high_min <= high_max")
    if config["cross_scale_weight"] < 0:
        raise ValueError("cross_scale_weight must be non-negative")
    for width, height in config["eval_resolutions"]:
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("all evaluation dimensions must be positive multiples of 32")
    for relative in (config["baseline_manifest"], options_reference):
        if not os.path.isfile(os.path.join(ROOT, relative)):
            raise ValueError("missing required file: {}".format(relative))
    with open(os.path.join(ROOT, options_reference), "r") as stream:
        config["resolved_train_options"] = json.load(stream)
    options = config["resolved_train_options"]
    if options.get("use_stereo") or options.get("v1_multiscale") or \
            options.get("predictive_mask") or options.get("disable_automasking") or \
            options.get("pose_model_type") != "separate_resnet":
        raise ValueError("baseline options use a mode unsupported by the Step 1 migration")
    data_path = options.get("data_path")
    if not os.path.isdir(data_path):
        raise ValueError("baseline KITTI data_path is unavailable: {}".format(data_path))
    checkpoint_dir = config.get("checkpoint_dir")
    if checkpoint_dir and not os.path.isdir(os.path.join(ROOT, checkpoint_dir)):
        raise ValueError("checkpoint directory is unavailable: {}".format(checkpoint_dir))
    return config


def check(config):
    config = dict(config)
    config["repo_root"] = ROOT
    print(json.dumps(config, indent=2, sort_keys=True))
    print("configuration_check=ok")
    return 0


def main(argv=None):
    """Use the original CLI, or an experiment recipe with an explicit stage."""
    from options import MonodepthOptions

    parser = MonodepthOptions()
    args = parser.parse(argv)
    if args.config is None:
        if args.stage is not None:
            parser.parser.error("--stage requires --config")
        from trainer import Trainer
        trainer = Trainer(args)
        trainer.train()
        return 0
    if args.stage is None:
        parser.parser.error("--config requires --stage")
    # A recipe uses its recorded baseline options, as the previous runner did.
    # Reject ignored CLI overrides so the effective training setup is explicit.
    arguments = sys.argv[1:] if argv is None else argv
    allowed = {"--config", "--stage", "--max_updates"}
    overrides = [arg.split("=", 1)[0] for arg in arguments
                 if arg.startswith("--") and arg.split("=", 1)[0] not in allowed]
    if overrides:
        parser.parser.error("with --config, edit the recipe/baseline options instead of passing: " +
                            ", ".join(overrides))
    try:
        config = load_config(os.path.abspath(args.config))
        if args.stage == "check":
            return check(config)
        if args.stage == "regression":
            return run_training(config, args, regression=True)
        if not os.path.isdir("/home/zy/KITTI"):
            raise RuntimeError("configured KITTI data path /home/zy/KITTI does not exist")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to start {} stage".format(args.stage))
        return run_training(config, args)
    except (ValueError, RuntimeError) as error:
        print("step1_error={}".format(error), file=sys.stderr)
        return 2


def run_training(config, args, regression=False):
    """Construct Trainer options from the saved baseline and run one stage."""
    import copy
    import torch
    from types import SimpleNamespace
    from trainer import Trainer

    if regression:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; regression requires the configured GPU")
        stage = "regression"
    else:
        stage = args.stage
    run_dir = os.path.join(ROOT, config["output_dir"],
                           "step1_{}_{}".format(config["experiment_id"], stage))
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise RuntimeError("refusing to overwrite existing run directory: {}".format(run_dir))
    options = copy.deepcopy(config["resolved_train_options"])
    options.update({
        "model_name": "step1_{}_{}".format(config["experiment_id"], stage),
        "log_dir": os.path.join(ROOT, config["output_dir"]),
        "scale_training": config["scale_training"],
        "scale_low_range": config["scale_low_range"],
        "scale_high_range": config["scale_high_range"],
        "cross_scale_weight": config["cross_scale_weight"],
        "scale_geometry_mode": config["scale_geometry_mode"],
        "cross_scale_output": 0,
        "save_scale_debug": False,
        "no_cuda": False,
        "num_workers": min(int(options.get("num_workers", 0)), 4),
        "max_updates": args.max_updates if stage == "smoke" else None,
    })
    if stage == "smoke":
        checkpoint_dir = config.get("checkpoint_dir", "tmp/monodepth2-base/models/weights_19")
        options["load_weights_folder"] = os.path.join(ROOT, checkpoint_dir)
        options["weights_init"] = "scratch"
        options["num_epochs"] = max(1, options.get("num_epochs", 1))
    elif stage == "regression":
        options["scale_training"] = "off"
        options["cross_scale_weight"] = 0.0
        options["load_weights_folder"] = os.path.join(ROOT, "tmp/monodepth2-base/models/weights_19")
        options["weights_init"] = "scratch"
        options["num_epochs"] = 1
        options["max_updates"] = 1
    else:
        options["load_weights_folder"] = None
    seed = config["seed"]
    if seed == "unknown":
        seed = 0
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    options["seed"] = seed
    trainer = Trainer(SimpleNamespace(**options))
    if stage == "smoke":
        trainer.model_optimizer = torch.optim.Adam(trainer.parameters_to_train,
                                                    options["learning_rate"])
        trainer.model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.model_optimizer, options["scheduler_step_size"], 0.1)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
