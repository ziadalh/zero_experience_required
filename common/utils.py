from typing import Dict, List, Optional

import glob
import os
from collections import defaultdict

import numpy as np
import torch

from habitat_sim.utils.common import d3_40_colors_rgb


def np_to_dtype(observations: List[Dict], dtype: Optional[np.dtype] = np.float):
    r"""Cast all numpy arrays in obsercations to dtype
    """
    for obs in observations:
        for sensor in obs:
            if isinstance(obs[sensor], np.ndarray):
                obs[sensor] = obs[sensor].astype(dtype)
                if np.any(np.isnan(obs[sensor])):
                    print(f"Getting NANs from sensor {sensor}")

    return observations


def _to_tensor(v, dtype=torch.float) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.to(dtype=dtype)
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v).to(dtype=dtype)
    else:
        return torch.tensor(v, dtype=dtype)


@torch.no_grad()
def batch_obs(
    observations: List[Dict],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)
        if len(batch[sensor].shape) == 1:
            batch[sensor] = batch[sensor].unsqueeze(-1)

    return batch


def convert_semantics_to_rgb(semantics):
    r"""Converts semantic IDs to RGB images.
    """
    semantics = semantics.long() % 40
    mapping_rgb = torch.from_numpy(d3_40_colors_rgb).to(semantics.device)
    semantics_r = torch.take(mapping_rgb[:, 0], semantics)
    semantics_g = torch.take(mapping_rgb[:, 1], semantics)
    semantics_b = torch.take(mapping_rgb[:, 2], semantics)
    semantics_rgb = torch.stack([semantics_r, semantics_g, semantics_b], -1)

    return semantics_rgb


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*[0-9]*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None
