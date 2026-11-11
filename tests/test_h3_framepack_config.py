"""Fixed recipe metadata must not silently misdescribe H3 Stage2 execution."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipelines.h3.train_stage2 import main, validate_recipe


class FramePackConfiguration(unittest.TestCase):
    def config(self, name='stage2_pilot.json'):
        path = Path(__file__).resolve().parents[1] / 'h3' / 'configs' / name
        return json.loads(path.read_text())

    def test_shipped_pilot_and_full_recipes_are_supported(self):
        validate_recipe(self.config())
        validate_recipe(self.config('stage2_full.json'))

    def test_unsupported_scientific_and_execution_changes_are_rejected(self):
        changes = [
            ('objective', 'shift', 12), ('objective', 'teacher_forcing', False),
            ('objective', 'first_chunk_latents', 5), ('objective', 'later_chunk_latents', 4),
            ('train', 'SP', 2), ('train', 'micro_batch', 2),
            ('train', 'effective_batch', 16), ('train', 'attention_backend', 'auto'),
            ('train', 'train_compressor', False), ('train', 'allowed_world_sizes', [2]),
            ('optimizer', 'warmup', 100), ('optimizer', 'schedule', 'cosine'),
            ('validation', 'CFG', 5), ('validation', 'schedule_points', 1),
            ('data', 'qwen_skip_reference_videos', False),
        ]
        for section, field, value in changes:
            with self.subTest(section=section, field=field, value=value):
                config = self.config()
                config[section][field] = value
                with self.assertRaises(ValueError):
                    validate_recipe(config)

    def test_supported_runtime_hyperparameters_remain_configurable(self):
        config = self.config()
        config['optimizer'].update(lr=2e-5, betas=[.8, .99], clip=2.)
        config['train'].update(lora_rank=16, lora_alpha=16, max_steps=20,
                               allowed_world_sizes=[4])
        config['validation']['schedule_points'] = 60
        validate_recipe(config)

    def test_incorrect_shift_fails_in_real_entry_before_cuda_initialization(self):
        config = self.config()
        config['objective']['shift'] = 12
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps(config))
            with patch.dict(os.environ, BIWM_RUN=directory), patch(
                    'pipelines.h3.train_stage2.torch.cuda.set_device') as set_device:
                with self.assertRaisesRegex(ValueError, 'objective.shift'):
                    main()
                set_device.assert_not_called()


if __name__ == '__main__':
    unittest.main()
