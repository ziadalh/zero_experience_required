from typing import Any
import numpy as np
from gym import spaces

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    RGBSensor,
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

from .image_nav_dataset import ImageGoalNavEpisode


@registry.register_sensor(name="ImageGoalSensorV2")
class ImageGoalSensorV2(Sensor):
    r"""Sensor for ImageGoal observations which are used in
    ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoalv2"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalSensorV2 requires one RGB sensor, "
                f"{len(rgb_sensor_uuids)} detected"
            )

        # (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_scene_id = None
        self._current_episode_id = None
        self._current_image_goal = None
        self._sampling_type = getattr(config, 'SAMPLING_TYPE', 'uniform')
        channels = getattr(config, 'CHANNELS', ['rgb'])
        if isinstance(channels, list):
            self._channels = channels
        elif isinstance(channels, str):
            # string with / to separate modalities
            self._channels = channels.split('/')
        else:
            raise ValueError(f'Unknown data type for channels!')

        self._channel2uuid = {}
        self._channel2range = {}
        self._shape = None
        self._current_goal_views = []
        self._setup_channels()
        self._set_space()
        super().__init__(config=config)

    def _get_sensor_uuid(self, sensor_type):
        sensors = self._sim.sensor_suite.sensors
        sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalSensorV2 requires one {sensor_type} sensor, "
                f"{len(sensor_uuids)} detected"
            )

        return sensor_uuids[0]

    def _setup_channels(self):
        self._channel2uuid = {}
        self._channel2range = {}
        last_idx = 0
        if 'rgb' in self._channels:
            self._channel2uuid['rgb'] = self._get_sensor_uuid(RGBSensor)
            self._channel2range['rgb'] = (last_idx, last_idx + 3)
            last_idx += 3

        if len(self._channel2uuid.keys()) == 0:
            raise ValueError('ImageGoalSensorV2 requires at least one channel')

    def _set_space(self):
        self._shape = None
        for k in self._channel2uuid.keys():
            uuid = self._channel2uuid[k]
            ospace = self._sim.sensor_suite.observation_spaces.spaces[uuid]
            if self._shape is None:
                self._shape = [ospace.shape[0], ospace.shape[1], 0]
            else:
                if ((self._shape[0] != ospace.shape[0]) or
                    (self._shape[1] != ospace.shape[1])):
                    raise ValueError('ImageGoalSensorV2 requires all '
                                     'base sensors to have the same with '
                                     'and hight, {uuid} has shape {ospace.shape}')

            if len(ospace.shape) == 3:
                self._shape[2] += ospace.shape[2]
            else:
                self._shape[2] += 1

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self._shape,
            dtype=np.float32)

    def _get_image_goal_at(self, position, rotation):
        obs = self._sim.get_observations_at(
            position=position, rotation=rotation)
        goal = []
        if 'rgb' in self._channel2uuid.keys():
            goal.append(obs[self._channel2uuid['rgb']].astype(
                self.observation_space.dtype))

        return np.concatenate(goal, axis=2)

    def _get_episode_image_goal(self, episode: Episode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        self._current_goal_views = []
        view = []

        if isinstance(episode, ImageGoalNavEpisode):
            view = episode.goals[0].views[0]
            if len(episode.goals[0].views) > 0 and self._sampling_type == 'random':
                # pick a random view for each episode run
                rng = np.random.RandomState()
                view = episode.goals[0].views[
                    rng.randint(low=0,
                                high=len(episode.goals[0].views))]
        else:
            if self._sampling_type == 'random':
                rng = np.random.RandomState()
                angle = rng.uniform(0, 2 * np.pi)
                view = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            else:
                # to be sure that the rotation is the same for
                # the same episode_id
                # since the task is using pointnav Dataset.
                seed = abs(hash(episode.episode_id)) % (2 ** 32)
                rng = np.random.RandomState(seed)
                angle = rng.uniform(0, 2 * np.pi)
                view = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

        source_rotation = np.array(view, dtype=np.float32).tolist()
        goal_observation = self._get_image_goal_at(
            goal_position.tolist(), source_rotation)
        self._current_goal_views.append(source_rotation)

        return goal_observation

    def get_goal_views(self):
        return self._current_goal_views

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        if episode.scene_id != self._current_scene_id:
            self._current_scene_id = episode.scene_id

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor(name="PoseSensor")
class PoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame
    defined by the episode, i.e. the axis it faces along and the origin is
    defined by its state at t=0. Additionally contains the time-step of
    the episode.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
            to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents
            position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0
        pose = np.array(
            [-agent_position_xyz[2], agent_position_xyz[0],
                agent_heading, ep_time],
            dtype=np.float32
        )

        return pose
