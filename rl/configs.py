#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from xenv import get_config_ext as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.RANDOM_SEED = 123
_C.BASE_TASK_CONFIG_PATH = ""
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppox"
_C.ENV_NAME = ""
_C.SIMULATOR_GPU_ID = 0
_C.SIMULATOR_GPU_IDS = []
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = -1
_C.EVAL_CKPT_PATH_DIR = "checkpoints"  # path to ckpt or path to ckpts dir
_C.EVAL_PREV_CKPT_ID = -1  # The evaluation starts at (this value + 1)th ckpt
_C.NUM_PROCESSES = 4
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "checkpoints"
_C.NUM_EPISODES = 20000
_C.T_EXP = 1000
_C.LOG_FILE = "output.log"
_C.SAVE_STATISTICS_FLAG = False
_C.CHECKPOINT_INTERVAL = 50
_C.FORCE_BLIND_POLICY = False

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.REWARD_MEASURES = []
_C.RL.REWARD_SCALES = [1.0]
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.COLLISION_PENALTY_FACTOR = 0.0  # Disabled by default
# -----------------------------------------------------------------------------
# POLICY
# -----------------------------------------------------------------------------
_C.RL.POLICY = CN()
_C.RL.POLICY.name = "NavNetPolicy"
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 4
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 1e-3
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
# _C.RL.PPO.loss_stats_window_size = 100
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.use_normalized_advantage = True
# visual encoder config
_C.RL.PPO.backbone = "resnet18"
# _C.RL.PPO.visual_encoder_backbone = "resnet18"
_C.RL.PPO.visual_encoder_init = ""
_C.RL.PPO.visual_encoder_embedding_size = 512
_C.RL.PPO.visual_obs_inputs = ['*']
# state encoder config
_C.RL.PPO.rnn_type = "LSTM"
_C.RL.PPO.num_recurrent_layers = 1
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.pretrained_weights = ""
# Loads pretrained weights
_C.RL.PPO.pretrained = False
_C.RL.PPO.pretrained_type = 'ckpt'
# Loads just the visual encoder backbone weights
_C.RL.PPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.PPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.PPO.reset_critic = True
# use early fusion for observations
_C.RL.PPO.early_fuse = False
# size of visual input
_C.RL.PPO.input_size = 256

_C.RL.PPO.policy_type = 'rnn'  # rnn
# apply random cropping to observations
_C.RL.PPO.random_crop = False
# apply random color jitter to rgbs
_C.RL.PPO.rgb_color_jitter = 0.
# Heuristic stopping criterion
_C.RL.PPO.use_heuristic_stop = False
# use the same net for goal and input (e.g. imagegoal/rgb)
_C.RL.PPO.tie_inputs_and_goal_param = False
# exclude weight with this prefix
_C.RL.PPO.pretrained_exclude = ""
# loads state encoder weights
_C.RL.PPO.pretrained_state_encoder = False
# train the state encoder
_C.RL.PPO.train_state_encoder = True
# _C.RL.PPO.pretrained_critic = False
# goal encoder type
_C.RL.PPO.goal_encoder_type = ""
# init goal encoder from pretraiend weights
_C.RL.PPO.goal_encoder_init = ""
# goal embedding size
_C.RL.PPO.goal_embedding_size = 128
# train the goal encoder
_C.RL.PPO.train_goal_encoder = True

# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"


def get_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
