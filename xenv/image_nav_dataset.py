import os
from typing import List, Optional

import json
import attr

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    ShortestPathPoint,
)
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)


@attr.s(auto_attribs=True, kw_only=True)
class ImageGoal(NavigationGoal):
    r"""Image goal provides information about an image that is target for
    navigation.

    Args:
        views: valid views to sample for image goal
    """
    views: Optional[List[List[float]]] = None


@attr.s(auto_attribs=True, kw_only=True)
class ImageGoalNavEpisode(NavigationEpisode):
    r"""ImageGoal Navigation Episode
    """
    goals: List[ImageGoal] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_dataset(name="ImageNav-v2")
class ImageNavDatasetV2(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Image Navigation
    dataset."""
    episodes: List[ImageGoalNavEpisode]

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        self.config = config

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = ImageGoalNavEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = ImageGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)


@registry.register_task(name="ImageNav-v2")
class ImageNavigationTask(NavigationTask):
    r"""An Image Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
