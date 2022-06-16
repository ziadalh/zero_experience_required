import torch

from habitat import logger


def init_model_from_file(model, init_file, state_key=None, module_prefix=None):
    device = model.device
    model_state = torch.load(init_file, map_location=device)
    if state_key is not None:
        model_state = model_state[state_key]

    init_model_from_state(model, model_state, module_prefix)


def init_model_from_state(
        model, model_state,
        module_prefix=None, exclude_prefix=None, skip_if_not_found=False,
        verbose=False):
    if exclude_prefix is not None:
        matched_prefix = [
            k[len(module_prefix):]
            for k, _ in model_state.items()
            if k.startswith(module_prefix) and not k.startswith(exclude_prefix)
        ]
    else:
        matched_prefix = [
            k[len(module_prefix):]
            for k, _ in model_state.items()
            if k.startswith(module_prefix)
        ]

    logger.info(f"model init prefix [{module_prefix}] / "
                f"excluding [{exclude_prefix}] matches "
                f"{len(matched_prefix)} modules")
    if verbose:
        logger.info(f"all matches: {matched_prefix}")

    if len(matched_prefix) == 0:
        if not skip_if_not_found:
            raise ValueError("can not find any matching keys with prefix "
                             f"[{module_prefix}]!")
        else:
            logger.info("can not find any matching keys with prefix "
                        f"[{module_prefix}]! skipping module initialization")
            return

    if exclude_prefix is not None:
        model.load_state_dict({
            k[len(module_prefix):]: v
            for k, v in model_state.items()
            if k.startswith(module_prefix) and not k.startswith(exclude_prefix)
        }, strict=False)
    else:
        model.load_state_dict({
            k[len(module_prefix):]: v
            for k, v in model_state.items()
            if k.startswith(module_prefix)
        })


def model_param_info(model, prefix=''):
    n_param = sum(param.numel() for param in model.parameters())
    n_tr_param = sum(
        param.numel()
        for param in model.parameters() if param.requires_grad)
    logger.info(f"{prefix} parameters: {n_param} ({n_param / 1e6:.3f} M), "
                f"trainable: {n_tr_param} "
                f"({(100. * n_tr_param/n_param):0.2f}%)")
