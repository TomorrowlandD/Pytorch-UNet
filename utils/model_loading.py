from collections.abc import Mapping

import torch


ATTENTION_MISSING_PREFIX = 'bottleneck_attention.'


def load_model_state(model, state_dict: Mapping, load_mode: str = 'strict'):
    """Load a U-Net state dict without silently accepting unrelated mismatches.

    ``backbone`` mode is intended only for initializing an attention-augmented
    U-Net from an older baseline checkpoint. It permits missing parameters from
    the new bottleneck attention branch and rejects every other mismatch.
    """
    if load_mode not in {'strict', 'backbone'}:
        raise ValueError(f'Unsupported load mode: {load_mode}')
    if not isinstance(state_dict, Mapping):
        raise TypeError('Checkpoint must contain a state-dict mapping')

    # Copy before removing metadata so callers can safely reuse their mapping.
    model_state = dict(state_dict)
    mask_values = model_state.pop('mask_values', [0, 1])

    if load_mode == 'strict':
        model.load_state_dict(model_state, strict=True)
        return mask_values

    incompatible = model.load_state_dict(model_state, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(ATTENTION_MISSING_PREFIX)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            'Checkpoint mismatch outside the attention branch: '
            f'missing={invalid_missing}, '
            f'unexpected={list(incompatible.unexpected_keys)}'
        )

    if getattr(model, 'attention_type', 'none') == 'none' and incompatible.missing_keys:
        raise RuntimeError(
            'Backbone loading reported missing attention keys for a model with '
            'attention disabled'
        )

    return mask_values


def load_checkpoint(model, checkpoint_path, map_location, load_mode: str = 'strict'):
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    return load_model_state(model, state_dict, load_mode=load_mode)
