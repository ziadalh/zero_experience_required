from typing import Any

import numpy as np

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import (
    Simulator,
)
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    angle_between_quaternions,
)
from habitat.tasks.nav.nav import (
    DistanceToGoal,
    Success,
)
from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()

COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON
MAP_THICKNESS_SCALAR: int = 128


@registry.register_measure
class ViewAngle(Measure):
    r"""The angle between the agent pose and the goal view when stopping
    """

    cls_uuid: str = "view_angle"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self._metric = 180.
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        if ep_success:
            goal_sensor = task.sensor_suite.sensors[
                task._config.GOAL_SENSOR_UUID
            ]
            if hasattr(goal_sensor, 'get_goal_views'):
                goal_views = [
                    quaternion_from_coeff(v)
                    for v in goal_sensor.get_goal_views()
                ]
                agent_view = self._sim.get_agent_state().rotation
                dist_to_view = [
                    angle_between_quaternions(agent_view, qk)
                    for qk in goal_views
                ]
                dist_to_view = np.abs(np.array(dist_to_view)).min()
                self._metric = np.rad2deg(dist_to_view)
            else:
                self._metric = 180.0
        else:
            self._metric = 180.0


@registry.register_measure
class DistanceToView(Measure):
    r"""
    """

    cls_uuid: str = "distance_to_view"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        succ_d = getattr(task._config, "SUCCESS_DISTANCE", 0)
        dist_to_goal = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        dist_to_view = np.pi
        goal_sensor = task.sensor_suite.sensors[
            task._config.GOAL_SENSOR_UUID
        ]
        if dist_to_goal <= succ_d and hasattr(goal_sensor, 'get_goal_views'):
            goal_views = [
                quaternion_from_coeff(v)
                for v in goal_sensor.get_goal_views()
            ]
            agent_view = self._sim.get_agent_state().rotation
            dist_to_view = [
                angle_between_quaternions(agent_view, qk)
                for qk in goal_views
            ]
            dist_to_view = np.abs(np.array(dist_to_view)).min()

        self._metric = dist_to_goal + dist_to_view


@registry.register_measure
class ViewMatch(Measure):
    r"""
    """

    cls_uuid: str = "view_match"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._view_weight = getattr(self._config, "VIEW_WEIGHT", 0.5)
        self._angle_threshold = np.deg2rad(self._config.ANGLE_THRESHOLD)
        assert self._view_weight >= 0 and self._view_weight <= 1., "VIEW_WEIGHT has to be in [0, 1]"
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._metric = 0.
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        succ_d = getattr(task._config, "SUCCESS_DISTANCE", 0.)
        dist_to_goal = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if dist_to_goal <= succ_d:
            goal_sensor = task.sensor_suite.sensors[
                task._config.GOAL_SENSOR_UUID
            ]
            self._metric = 1 - self._view_weight
            if hasattr(goal_sensor, 'get_goal_views'):
                goal_views = [
                    quaternion_from_coeff(v)
                    for v in goal_sensor.get_goal_views()
                ]
                agent_view = self._sim.get_agent_state().rotation
                dist_to_view = [
                    angle_between_quaternions(agent_view, qk)
                    for qk in goal_views
                ]
                dist_to_view = np.abs(np.array(dist_to_view)).min()
                if dist_to_view <= self._angle_threshold:
                    self._metric += self._view_weight

            else:
                self._metric = 1.0
        else:
            self._metric = 0.0

        self._metric = ep_success * self._metric
