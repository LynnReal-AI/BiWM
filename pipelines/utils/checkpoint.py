import json
import os

from safetensors.torch import save_file
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from pipelines.utils.logging_ import primary_print


def store_checkpoint(transformer, rank, output_dir, step):
    primary_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        # Try to save config (if the model has a config attribute)
        try:
            model_for_config = transformer
            if hasattr(transformer, '_fsdp_wrapped_module'):
                model_for_config = transformer._fsdp_wrapped_module
            if hasattr(model_for_config, '_checkpoint_wrapped_module'):
                model_for_config = model_for_config._checkpoint_wrapped_module

            if hasattr(model_for_config, 'config'):
                config_dict = dict(model_for_config.config)
                if "dtype" in config_dict:
                    del config_dict["dtype"]
                # Convert enum / non-serializable types to strings for JSON
                for key, value in list(config_dict.items()):
                    if hasattr(value, 'value'):
                        config_dict[key] = value.value
                    elif hasattr(value, 'name'):
                        config_dict[key] = value.name
                    elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        config_dict[key] = str(value)
                config_path = os.path.join(save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=4)
            else:
                config_path = os.path.join(save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump({"model_type": "ltx2", "step": step}, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
        del cpu_state
    primary_print(f"--> checkpoint saved at step {step}")
