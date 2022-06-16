#!/usr/bin/env python3

import argparse
import os
import os.path as path
import random
import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry

from rl.configs import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="path to output directory")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
        exp_config: str,
        run_type: str,
        output_dir: str,
        noisy_actions=False,
        noisy_rgb=False,
        noisy_depth=False,
        opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

    # set output dir
    if run_type == "eval":
        if path.isfile(config.EVAL_CKPT_PATH_DIR):
            ckpt = path.basename(config.EVAL_CKPT_PATH_DIR).replace('.', '_')
            config.LOG_FILE = path.join(output_dir,
                                        f'{run_type}_{config.EVAL.SPLIT}_'
                                        f'{ckpt}.log')
        else:
            config.LOG_FILE = path.join(output_dir,
                                        f'{run_type}_{config.EVAL.SPLIT}.log')

        config.TENSORBOARD_DIR = path.join(output_dir, 'tb')
        config.VIDEO_DIR = path.join(output_dir, 'video_dir')
    else:
        config.LOG_FILE = path.join(output_dir, f'{run_type}.log')
        config.TENSORBOARD_DIR = path.join(output_dir,
                                           config.TENSORBOARD_DIR)
        config.VIDEO_DIR = path.join(output_dir,
                                     config.VIDEO_DIR)
        config.EVAL_CKPT_PATH_DIR = path.join(output_dir,
                                              config.EVAL_CKPT_PATH_DIR)
        config.CHECKPOINT_FOLDER = path.join(output_dir,
                                             config.CHECKPOINT_FOLDER)

    config.freeze()

    # habitat.logger.info(config)

    if not path.exists(output_dir):
        os.makedirs(output_dir)
        # os.makedirs(config.TENSORBOARD_DIR)
        # os.makedirs(config.CHECKPOINT_FOLDER)

    # copy config to output dir
    with open(path.join(output_dir, f'config_{run_type}.yaml'), 'w') as f:
        f.write("{}".format(config))

    # fix seed
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
