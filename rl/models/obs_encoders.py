import torch
from torch import nn as nn
from torchvision.transforms import ColorJitter

from habitat import logger

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)

from habitat_sim.utils.common import d3_40_colors_rgb

from rl.models.cnns import SimpleCNN
from rl.models import resnet as fast_resnet

from xenv.utils import ResizeCenterCropper

VISUAL_SENSORS_UUID = [
    'rgb', 'depth', 'semantic',
]

MOD_SEP = '/'


def make_resnet_encoder(make_backbone, n_input_channels, baseplanes, ngroups,
                        spatial_size=128, normalize_inputs=False,
                        after_compression_flat_size=2048):
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    backbone = make_backbone(n_input_channels, baseplanes, ngroups)
    modules.append(backbone)
    final_spatial = int(spatial_size * backbone.final_spatial_compress)
    num_compression_channels = int(
        round(after_compression_flat_size / (final_spatial ** 2))
    )
    compression = nn.Sequential(
        nn.Conv2d(
            backbone.final_channels,
            num_compression_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(1, num_compression_channels),
        nn.ReLU(True),
    )
    modules.append(compression)
    output_shape = (
        num_compression_channels,
        final_spatial,
        final_spatial,
    )
    return nn.Sequential(*modules), output_shape


def make_simplecnn_encoder(n_input_channels, spatial_size=128,
                           normalize_inputs=False, output_size=2048):
    input_shape = (spatial_size, spatial_size, n_input_channels)
    output_shape = (output_size, 1, 1)
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    backbone = SimpleCNN(input_shape, output_size, output_shape)
    modules.append(backbone)

    return nn.Sequential(*modules), output_shape


class EarlyFuseCNNEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        backbone=None,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        rgb_color_jitter=0.,
        tied_params=None,
    ):
        super().__init__()
        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            logger.info(f'use obs_transform: {type(self.obs_transform)}')
            observation_space = self.obs_transform.transform_observation_space(
                observation_space, trans_keys=VISUAL_SENSORS_UUID
            )
        self._n_input = {k: 0 for k in VISUAL_SENSORS_UUID}
        logger.info('observation_space kyes: {}'
                    .format(observation_space.spaces.keys()))
        logger.info('visual_obs_inputs: {}'.format(visual_obs_inputs))
        visual_uuid = VISUAL_SENSORS_UUID
        if ((visual_obs_inputs is not None) and (len(visual_obs_inputs) > 0) and (visual_obs_inputs[0] != '*')):
            if isinstance(visual_obs_inputs, list):
                visual_uuid = visual_obs_inputs
            else:
                visual_uuid = visual_obs_inputs.split(MOD_SEP)

        self._id2rgb = {k: False for k in VISUAL_SENSORS_UUID}
        self._colormap = torch.from_numpy(d3_40_colors_rgb)
        self._rgb_aug = None
        if "rgb" in visual_uuid:
            if rgb_color_jitter > 0:
                logger.info(f'use RGB color jitter= {rgb_color_jitter}')
                self._rgb_aug = ColorJitter(
                    brightness=rgb_color_jitter, contrast=rgb_color_jitter,
                    saturation=rgb_color_jitter, hue=rgb_color_jitter)

        num_input_channels = 0
        for v_uuid in visual_uuid:
            if v_uuid in observation_space.spaces:
                logger.info("{} observation_space: {}"
                            .format(v_uuid, observation_space.spaces[v_uuid]))
                spatial_size = observation_space.spaces[v_uuid].shape[0]
                if v_uuid == "semantic":
                    self._n_input[v_uuid] = 1
                else:
                    self._n_input[v_uuid] = \
                            observation_space.spaces[v_uuid].shape[2]

                num_input_channels += self._n_input[v_uuid]

        if not self.is_blind:
            if tied_params is not None:
                logger.info('encoder is tied to goal params')
                self.encoder = tied_params[0]
                v_output_shape = tied_params[1]
            else:
                if 'simple_cnn' in backbone:
                    self.encoder, v_output_shape = \
                        make_simplecnn_encoder(
                            num_input_channels,
                            spatial_size,
                            normalize_visual_inputs,
                            visual_encoder_embedding_size
                        )
                else:
                    if 'fast_resnet' in backbone:
                        restnet_type = backbone[5:]
                        backbone_enc = getattr(fast_resnet, restnet_type)
                    elif 'resnet' in backbone:
                        backbone_enc = getattr(resnet, backbone)
                    else:
                        raise ValueError('unknown type of backbone {}'
                                         .format(backbone))

                    self.encoder, v_output_shape = \
                        make_resnet_encoder(
                            backbone_enc,
                            num_input_channels,
                            baseplanes,
                            ngroups,
                            spatial_size,
                            normalize_visual_inputs,
                            visual_encoder_embedding_size
                        )

            num_compression_channels = v_output_shape[0]
            final_spatial = v_output_shape[1]
            logger.info('early fuse encoder(type: {}, in: {}, out: {})'.format(
                backbone,
                (num_input_channels, spatial_size, spatial_size),
                v_output_shape)
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        n_inputs = 0
        for v in self._n_input.values():
            n_inputs += v
        return n_inputs == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def _rgb_augment(self, rgb_obs, seeds=None):
        rgb_3 = None
        if rgb_obs.size(1) == 3:
            rgb_3 = rgb_obs
        elif rgb_obs.size(1) == 4:
            rgb_3 = self._rgb_aug(rgb_obs[:, :3, :, :])
        else:
            raise NotImplementedError(
                f"{rgb_obs.size(1)} channles are detected, "
                f"only 3 or 4 is currently supported for rgb_aug")

        if seeds is None:
            rgb_3 = self._rgb_aug(rgb_3)
        else:
            assert rgb_3.shape[0] == seeds.shape[0]
            outputs = []
            for i in range(seeds.shape[0]):
                torch.manual_seed(seeds[i])
                outputs.append(self._rgb_aug(rgb_3[i]).unsqueeze(0))

            rgb_3 = torch.cat(outputs, dim=0)

        rgb_obs[:, :3, :, :] = rgb_3
        return rgb_obs

    def forward(self, observations):
        if self.is_blind:
            return None

        seeds = None
        cnn_input = []
        for v_uuid in self._n_input.keys():
            if self._n_input[v_uuid] > 0:
                v_observations = observations[v_uuid].float()
                if v_uuid == "rgb":
                    v_observations = v_observations / 255.0  # normalize RGB

                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                v_observations = v_observations.permute(0, 3, 1, 2)
                if v_uuid == "rgb" and self._rgb_aug is not None:
                    v_observations = self._rgb_augment(v_observations, seeds)

                if self.obs_transform:
                    v_observations = self.obs_transform(v_observations, seeds)

                cnn_input.append(v_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        x = self.encoder(cnn_input)
        return x
