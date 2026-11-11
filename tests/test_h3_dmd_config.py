"""Configuration provenance gates; these fixtures do not exercise H3 training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipelines.h3.configure_stage3 import continuation_contract, digest, expand, prepare, write_config


class DMDConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [0, 2, 4, 6, 8, 10, 12, 13]
        def save(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return path
        self.save = save
        merge = save('merged/MERGE_COMPLETE.json', {'complete': True})
        phase2 = save('phase2/executed_config.json', {'initialization_path': str(merge.parent)})
        checkpoint = save('phase2/checkpoint_001010/COMPLETE.json', {'step': 1010}).parent
        selection = save('selection.json', dict(checkpoint=str(checkpoint), model_step=1010,
            source_config_sha256=digest(phase2), source_weight='Live', accepted_for_DMD=True,
            reviewed_cases=self.cases))
        bundle = save('bundle/COMPLETE.json', {'complete': True}).parent
        media = save('media.json', dict(all_passed=True, cases=8))
        teacher = save('teacher/COMPLETE.json', dict(cases=8, frames=158, source_step=1010,
            G_updates=0, C_updates=0))
        review = save('review.json', dict(accepted_for_finite_DMD_pilot=True,
            source_phase2_step=1010, reviewed_cases=self.cases, teacher_CFG=1,
            source_media_verification_sha256=digest(media),
            source_teacher_complete_sha256=digest(teacher)))
        train, validation = save('train.jsonl', []), save('validation.jsonl', [])
        self.config = dict(source=dict(phase2_checkpoint=str(checkpoint), selection=str(selection),
            score_bundle=str(bundle), teacher_quality_review=str(review),
            teacher_media_verification=str(media), teacher_complete=str(teacher)),
            data=dict(train_manifest=str(train), validation_manifest=str(validation),
                      release_root=str(self.root)),
            validation=dict(indices=self.cases, baseline_root=str(self.root/'run')),
            train=dict(max_generator_steps=2, max_critic_steps=15), dmd=dict(real_guidance_scale=1))

    def test_pins_reviewed_source_without_mutating_template(self):
        before = copy.deepcopy(self.config)
        result = prepare(self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(result['source']['selection_sha256'], digest(self.config['source']['selection']))
        self.assertEqual(result['source']['teacher_quality_review_sha256'], digest(self.config['source']['teacher_quality_review']))
        teacher_only = {key: copy.deepcopy(value) for key, value in self.config.items()
                        if key not in ('train', 'dmd')}
        self.assertIn('score_bundle_sha256', prepare(teacher_only)['source'])

    def test_rejects_changed_source(self):
        selected = json.loads(Path(self.config['source']['selection']).read_text())
        selected['source_config_sha256'] = 'changed'
        self.save('selection.json', selected)
        with self.assertRaisesRegex(ValueError, 'Phase2 source'):
            prepare(self.config)

    def test_rejects_wrong_teacher_CFG(self):
        changed = copy.deepcopy(self.config)
        changed['dmd']['real_guidance_scale'] = 5
        with self.assertRaisesRegex(ValueError, 'Teacher review'):
            prepare(changed)

    def test_rejects_tampered_teacher_marker(self):
        path = Path(self.config['source']['teacher_complete'])
        marker = json.loads(path.read_text())
        marker['source_step'] = 260
        path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, 'Teacher review'):
            prepare(self.config)

    def test_rejects_implicit_budget_extension(self):
        self.config['train'].update(max_generator_steps=3, max_critic_steps=20)
        with self.assertRaisesRegex(ValueError, 'budget exceeds'):
            prepare(self.config)

    def test_unset_paths_and_in_place_config_changes_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unset path'):
            expand('${BIWM_TEST_MISSING_PATH_731}')
        path = self.root/'run/config.json'
        write_config({'value': 1}, path)
        before = path.read_bytes()
        write_config({'value': 1}, path)
        with self.assertRaisesRegex(ValueError, 'Existing run'):
            write_config({'value': 2}, path)
        self.assertEqual(path.read_bytes(), before)

    def test_continuation_allows_reviewed_budget_metadata_only(self):
        before = copy.deepcopy(self.config)
        changed = copy.deepcopy(before)
        changed['train'].update(initial_generator_step=2, max_generator_steps=4, max_critic_steps=25)
        changed.update(status='Reviewed continuation', continuation='/complete/checkpoint_000002',
                       checkpoint_steps=[3, 4], validation_steps=[3, 4])
        changed['source'].update(teacher_quality_review='/new/review.json',
                                 teacher_quality_review_sha256='new-reviewed-proof')
        self.assertEqual(continuation_contract(before), continuation_contract(changed))
        self.assertEqual(before, self.config)
        for section, key, new in [('source', 'score_bundle_sha256', 'changed-weights'),
                                  ('dmd', 'real_guidance_scale', 5),
                                  ('data', 'train_manifest', '/different/data.jsonl')]:
            changed = copy.deepcopy(before)
            changed[section][key] = new
            self.assertNotEqual(continuation_contract(before), continuation_contract(changed))


if __name__ == '__main__':
    unittest.main()
