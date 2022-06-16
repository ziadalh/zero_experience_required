from typing import Dict, List, Optional
import numpy as np
from PIL import Image

from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import colorize_ids
from habitat.utils.visualizations.utils import draw_collision
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

cv2 = try_cv2_import()


def draw_top_down_map(info, heading, output_size):
    """
    Creates an image of top down map with the agent
    info is from the output of env.step()
    """
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0]))
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def obs_semantic2rgb(semantic_obs):
    """
    Creates a 3D array based on the semantic ids in the observation
    """
    return colorize_ids(semantic_obs.astype(np.uint8))


def obs_depth2rgb(depth_obs):
    """
    Creates a 3d array for a depth image from depth observation
    """
    depth_map = depth_obs.squeeze() * 255.0
    if not isinstance(depth_map, np.ndarray):
        depth_map = depth_map.cpu().numpy()

    depth_map = depth_map.astype(np.uint8)
    depth_map = np.stack([depth_map for _ in range(3)], axis=2)
    return depth_map


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    for uuid in ["imagegoalv2"]:
        if uuid in observation:
            rgb = observation[uuid]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            if len(rgb.shape) == 2 or rgb.shape[2] == 1:
                rgb = np.stack([rgb.squeeze() for _ in range(3)], axis=2)

            if rgb.shape[0] != egocentric_view[0].shape[0]:
                imsize = egocentric_view[0].shape[:2]
                rgb = np.array(Image.fromarray(rgb).resize(imsize))

            egocentric_view.append(rgb)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    # if "collisions" in info and info["collisions"]["is_collision"]:
    #     egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], egocentric_view.shape[0]
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    if "floor_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["floor_map"], egocentric_view.shape[0]
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    return frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
    prefix="",
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"{prefix}episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"{prefix}episode{episode_id}", checkpoint_idx, images, fps=fps
        )
