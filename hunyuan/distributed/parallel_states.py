# get_parallel_state() adapter over biWM's nccl_info SP group (shared with Wan2.2).


class _ParallelState:
    """Thin adapter over biWM's nccl_info (properties fetch dynamically)."""

    @property
    def sp_enabled(self):
        from pipelines.utils.parallel_states import get_sequence_parallel_state
        return get_sequence_parallel_state()

    @property
    def sp(self):
        from pipelines.utils.parallel_states import nccl_info
        return nccl_info.sp_size

    @property
    def sp_rank(self):
        from pipelines.utils.parallel_states import nccl_info
        return nccl_info.rank_within_group

    @property
    def sp_group(self):
        from pipelines.utils.parallel_states import nccl_info
        return nccl_info.group


_STATE = _ParallelState()


def get_parallel_state():
    return _STATE
