#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import tqdm
import torch
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO, PointNavBaselinePolicy

from common.utils import np_to_dtype, batch_obs
from common.base_trainer import BaseRLTrainer
from rl.policy import NavNetPolicy
import rl.models.utils as mutils
from utils.visualize import observations_to_image, generate_video


@baseline_registry.register_trainer(name="ppox")
class PPOXTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_weights(self, ppo_cfg) -> None:
        if (
            self.config.RL.PPO.pretrained_encoder
            or self.config.RL.PPO.pretrained_state_encoder
            or self.config.RL.PPO.pretrained
        ):
            logger.info(
                "load pretrained_weights from: "
                f"{self.config.RL.PPO.pretrained_weights}")
            pretrained_state = torch.load(
                self.config.RL.PPO.pretrained_weights, map_location="cpu"
            )

        exclude_prefix = None
        if self.config.RL.PPO.pretrained_exclude != "":
            exclude_prefix = self.config.RL.PPO.pretrained_exclude

        if self.config.RL.PPO.pretrained:
            logger.info("init actor_critic from pretrained_weights")
            mutils.init_model_from_state(
                self.actor_critic,
                pretrained_state["state_dict"],
                module_prefix="actor_critic.",
                exclude_prefix=exclude_prefix
            )

        if self.config.RL.PPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        mutils.model_param_info(self.actor_critic, 'actor_critic')

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        if self.config.RL.POLICY.name == "NavNetPolicy":
            self.actor_critic = NavNetPolicy.from_config(
                self.config, self.envs)
        else:
            self.actor_critic = PointNavBaselinePolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.action_space,
                hidden_size=ppo_cfg.hidden_size,
            )

        self.actor_critic.to(device=self.device, dtype=torch.float)

        self._setup_actor_critic_weights(ppo_cfg)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name),
            _use_new_zipfile_serialization=False
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self, device=None):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        checkpoints = [v for v in checkpoints if "ckpt.latest.pth" not in v]
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(
                checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location=device)
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        observations = np_to_dtype(observations, np.float)
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        observations = np_to_dtype(observations, np.float)
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # Try to resume at previous checkpoint (independent of interrupted states)
        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint(self.device)
        count_steps = count_steps_start

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for _step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    for k in metrics.keys():
                        writer.add_scalar("metrics/{}".format(k), metrics[k], count_steps)

                losses = [value_loss, action_loss]
                writer.add_scalar("losses/value", value_loss, count_steps)
                writer.add_scalar("losses/action", action_loss, count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()
        writer_prefix = 'eval'
        if config.EVAL.SPLIT != 'val':
            writer_prefix = config.EVAL.SPLIT

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("FLOOR_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        logger.info(f"load saved checkpoint weights: {checkpoint_path}")
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        observations = self.envs.reset()
        observations = np_to_dtype(observations, np.float)
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        num_episodes_per_env = self.envs.number_of_episodes
        logger.info(f"number of episodes per env: {num_episodes_per_env}")
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(num_episodes_per_env)
        else:
            total_num_eps = sum(num_episodes_per_env)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(
            total=number_of_eval_episodes,
            desc=f"{config.EVAL.SPLIT} : {checkpoint_path.split('/')[-1]} : "
                 f"Step {step_id/1e6:.3f} M")
        self.actor_critic.eval()

        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            observations = np_to_dtype(observations, np.float)
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        sc_name = current_episodes[i].scene_id
                        prefix = sc_name.split('/')[-1].split('.')[0]
                        obj_cat = getattr(
                            current_episodes[i], 'object_category', None)
                        if obj_cat is not None:
                            obj_cat = obj_cat.replace(' ', '_')
                            prefix += '_' + obj_cat

                        room_cat = getattr(
                            current_episodes[i], 'room_category', None)
                        if room_cat is not None:
                            room_cat = room_cat.replace(' ', '_')
                            prefix += '_' + room_cat

                        prefix += '_'
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                            prefix=prefix,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        logger.info(f"Checkpoint Index: {checkpoint_index}")
        logger.info(f"Total number of episodes: {num_episodes}")
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        logger.info(f"Step ID: {step_id}")
        writer.add_scalar(
            f"{writer_prefix}_reward",
            aggregated_stats["reward"],
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            for k in metrics.keys():
                writer.add_scalar(f"{writer_prefix}_metrics/{k}",
                                  metrics[k],
                                  step_id)

        # write to csv
        ckpt_name = os.path.basename(checkpoint_path).replace('.', '_')
        df = pd.DataFrame([aggregated_stats])
        df['step'] = step_id
        df['checkpoint'] = os.path.basename(checkpoint_path)
        df['split'] = config.EVAL.SPLIT
        df['seed'] = config.TASK_CONFIG.SEED
        df['num_episodes'] = num_episodes
        df.to_csv(os.path.join(
            self.config.TENSORBOARD_DIR,
            f"{config.EVAL.SPLIT}_{ckpt_name}.csv"),
            float_format='%.5f', header=True, index=False,
        )

        self.envs.close()
