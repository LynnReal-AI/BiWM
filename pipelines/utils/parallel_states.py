import datetime
import os

import torch.distributed as dist

# Long video training requires a longer timeout (2 hours)
_SP_DEADLINE = datetime.timedelta(hours=2)


class CommState:

    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_state = CommState()
_SEQ_PARALLEL_FLAG = False


def setup_sequence_parallel_state(sequence_parallel_size):
    global _SEQ_PARALLEL_FLAG
    if sequence_parallel_size > 1:
        _SEQ_PARALLEL_FLAG = True
        setup_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_state.sp_size = 1
        nccl_state.global_rank = int(os.getenv("RANK", "0"))
        nccl_state.rank_within_group = 0
        nccl_state.group_id = int(os.getenv("RANK", "0"))


def fetch_sequence_parallel_state():
    return _SEQ_PARALLEL_FLAG


def setup_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size)
    nccl_state.sp_size = sequence_parallel_size
    nccl_state.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks, timeout=_SP_DEADLINE)
        if rank in ranks:
            nccl_state.group = group
            nccl_state.rank_within_group = rank - i * sequence_parallel_size
            nccl_state.group_id = i


def teardown_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()
