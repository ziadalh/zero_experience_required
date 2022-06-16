#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config
from habitat.config.default import (
    CONFIG_FILE_SEPARATOR,
)

_C = get_config()
_C.defrost()

# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR V2
# -----------------------------------------------------------------------------
_C.TASK.IMAGEGOAL_SENSOR_V2 = CN()
_C.TASK.IMAGEGOAL_SENSOR_V2.TYPE = "ImageGoalSensorV2"
# the type of sampling: uniform or random
_C.TASK.IMAGEGOAL_SENSOR_V2.SAMPLING_TYPE = 'uniform'
# the type of channels for the goal: rgb
_C.TASK.IMAGEGOAL_SENSOR_V2.CHANNELS = 'rgb'
# -----------------------------------------------------------------------------
# POSE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSE_SENSOR = CN()
_C.TASK.POSE_SENSOR.TYPE = "PoseSensor"
# -----------------------------------------------------------------------------
_C.TASK.VIEW_ANGLE = CN()
_C.TASK.VIEW_ANGLE.TYPE = "ViewAngle"
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_VIEW = CN()
_C.TASK.DISTANCE_TO_VIEW.TYPE = "DistanceToView"
# -----------------------------------------------------------------------------
_C.TASK.VIEW_MATCH = CN()
_C.TASK.VIEW_MATCH.TYPE = "ViewMatch"
_C.TASK.VIEW_MATCH.ANGLE_THRESHOLD = 25.
_C.TASK.VIEW_MATCH.VIEW_WEIGHT = .5
# -----------------------------------------------------------------------------
# FloorMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.FLOOR_MAP = CN()
_C.TASK.FLOOR_MAP.TYPE = "FloorMap"
_C.TASK.FLOOR_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.FLOOR_MAP.MAP_PADDING = 3
_C.TASK.FLOOR_MAP.MAP_RESOLUTION = 1024
_C.TASK.FLOOR_MAP.DRAW_SOURCE = True
_C.TASK.FLOOR_MAP.DRAW_BORDER = True
_C.TASK.FLOOR_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.FLOOR_MAP.FOG_OF_WAR = CN()
_C.TASK.FLOOR_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.FLOOR_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.FLOOR_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.FLOOR_MAP.DRAW_VIEW_POINTS = True
_C.TASK.FLOOR_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.FLOOR_MAP.DRAW_GOAL_AABBS = True


def get_config_ext(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
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
        config.merge_from_list(opts)

    config.freeze()
    return config
