# SPDX-License-Identifier: Apache-2.0
"""Storage placement only; preserve the model, gradient and Adam math."""
import torch
from torch.autograd.graph import save_on_cpu
from torch.utils.checkpoint import checkpoint

from h3.optimizer import FP32MasterAdamW


def checkpoint_with_cpu_inputs(module, *args):
    # Outer hooks move checkpoint's saved inputs; the inner checkpoint hooks
    # discard/recompute activations instead of offloading every frozen weight.
    with save_on_cpu(pin_memory=True):
        return checkpoint(module.__call__, *args, use_reentrant=False)


def enable_checkpoint_input_offload(model):
    model.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_with_cpu_inputs)


def move_fp32_optimizer_state(optimizer, *, to_cpu):
    """Park idle critic slots during G; restore them before the next C Adam.

    Synchronous, dtype-preserving copies do not consume RNG or touch live model
    parameters, gradients, counters or the CPU Adam step scalar. Empty/lazy
    states stay empty. Checkpoint state_dict may serialize the parked CPU slots;
    FP32MasterAdamW.load_state_dict restores them to each parameter's device.
    """
    if not isinstance(optimizer, FP32MasterAdamW):
        raise TypeError('State placement requires the native FP32-master AdamW')
    moved = 0
    for parameter, state in optimizer.state.items():
        if not state:
            continue
        if set(state) != {'step', 'master_param', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Unexpected FP32-master Adam state slots')
        step = state['step']
        if step.device.type != 'cpu' or step.dtype != torch.float32 or step.ndim != 0:
            raise ValueError('Adam step must remain a CPU FP32 scalar')
        target = torch.device('cpu') if to_cpu else parameter.device
        for key in ('master_param', 'exp_avg', 'exp_avg_sq'):
            value = state[key]
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError('Adam master/moments must retain FP32 and parameter shape')
            if value.device not in (torch.device('cpu'), parameter.device):
                raise ValueError('Adam state is on an unexpected device')
            if value.device != target:
                state[key] = value.to(device=target)
                moved += value.numel() * value.element_size()
    return moved
