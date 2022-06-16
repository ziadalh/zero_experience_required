import numpy as np
from gym import spaces
import torch
from torch import nn as nn

from habitat import logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat_baselines.utils.common import Flatten

from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy

from rl.models.obs_encoders import EarlyFuseCNNEncoder
from xenv import (
    ImageGoalSensorV2,
    PoseSensor,
)
from xenv.utils import (
    ResizeCenterCropper,
    ResizeRandomCropper,
)

VISUAL_SENSORS_UUID = [
    'rgb', 'depth', 'semantic',
]


# @baseline_registry.register_policy
class NavNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa : B008
        force_blind_policy=False,
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        **kwargs
    ):
        super().__init__(
            NavNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                force_blind_policy=force_blind_policy,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=visual_obs_inputs,
                visual_encoder_init=visual_encoder_init,
                **kwargs
            ),
            action_space.n,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    @classmethod
    def from_config(cls, config, envs):
        ppo_config = config.RL.PPO
        if ppo_config.random_crop:
            obs_transform = ResizeRandomCropper(
                size=(ppo_config.input_size, ppo_config.input_size))
        else:
            obs_transform = ResizeCenterCropper(
                size=(ppo_config.input_size, ppo_config.input_size))

        return cls(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=ppo_config.hidden_size,
            rnn_type=ppo_config.rnn_type,
            num_recurrent_layers=ppo_config.num_recurrent_layers,
            backbone=ppo_config.backbone,
            normalize_visual_inputs="rgb" in envs.observation_spaces[0].spaces,
            obs_transform=obs_transform,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            visual_encoder_embedding_size=ppo_config.visual_encoder_embedding_size,
            visual_obs_inputs=ppo_config.visual_obs_inputs,
            visual_encoder_init=ppo_config.visual_encoder_init,
            rgb_color_jitter=ppo_config.rgb_color_jitter,
            tie_inputs_and_goal_param=ppo_config.tie_inputs_and_goal_param,
            goal_encoder_type=ppo_config.goal_encoder_type,
            goal_encoder_init=ppo_config.goal_encoder_init,
            goal_embedding_size=ppo_config.goal_embedding_size,
        )


class NavNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
        force_blind_policy=False,
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        rgb_color_jitter=0.,
        tie_inputs_and_goal_param=False,
        goal_encoder_init=None,
        goal_encoder_type='',
        goal_embedding_size=128,
        **kwargs
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action
        ObsEncoder = EarlyFuseCNNEncoder

        logger.info('Type of observation encoder: {}'.format(ObsEncoder))
        tied_param = {}
        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, goal_embedding_size)
            rnn_input_size += goal_embedding_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        if PoseSensor.cls_uuid in observation_space.spaces:
            self.pose_embedding = nn.Linear(5, 16)
            rnn_input_size += 16

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(
                input_pointgoal_dim, goal_embedding_size)
            rnn_input_size += goal_embedding_size

        if ImageGoalSensorV2.cls_uuid in observation_space.spaces:
            logger.info('Create goal encoder for ImageGoalV2')
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensorV2.cls_uuid]}
            )
            self.goal_visual_encoder = ObsEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                backbone=backbone,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=["rgb"],
                visual_encoder_init=visual_encoder_init,
                rgb_color_jitter=rgb_color_jitter,
            )

            self.goal_visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape),
                    goal_embedding_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += goal_embedding_size

        self._hidden_size = hidden_size

        logger.info('Create visual inputs encoder')
        self.visual_encoder = ObsEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
            visual_encoder_embedding_size=visual_encoder_embedding_size,
            visual_obs_inputs=visual_obs_inputs,
            visual_encoder_init=visual_encoder_init,
            rgb_color_jitter=rgb_color_jitter,
            tied_params=tied_param if tie_inputs_and_goal_param else None,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _format_pose(self, pose):
        """
        Args:
            pose: (N, 4) Tensor containing x, y, heading, time
        """

        x, y, theta, time = torch.unbind(pose, dim=1)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        e_time = torch.exp(-time)
        formatted_pose = torch.stack([x, y, cos_theta, sin_theta, e_time], 1)

        return formatted_pose

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)

            x.append(visual_feats)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            goal_observations = torch.stack(
                [
                    goal_observations[:, 0],
                    torch.cos(-goal_observations[:, 1]),
                    torch.sin(-goal_observations[:, 1]),
                ],
                -1,
            ).float()

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            compass_feat = self.compass_embedding(
                compass_observations.squeeze(dim=1))

            x.append(compass_feat)

        if EpisodicGPSSensor.cls_uuid in observations:
            gps_feat = self.gps_embedding(
                observations[EpisodicGPSSensor.cls_uuid])

            x.append(gps_feat)

        if PoseSensor.cls_uuid in observations:
            pose_formatted = self._format_pose(
                observations[PoseSensor.cls_uuid])
            pose_feat = self.pose_embedding(pose_formatted.float())

            x.append(pose_feat)

        if ImageGoalSensorV2.cls_uuid in observations:
            goal_image = observations[ImageGoalSensorV2.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
        )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
