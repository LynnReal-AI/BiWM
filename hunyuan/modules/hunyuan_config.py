# stub: HunyuanVideoConfig exists only so model.py can read `_fsdp_shard_conditions` at import time.
# The slimmed version does no FSDP, so an empty list suffices.


class HunyuanVideoConfig:
    def __init__(self):
        self._fsdp_shard_conditions = []
