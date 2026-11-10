"""CPU contracts for H3 extraction; these do not certify 33B GPU training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from h3.data import CameraDataset, validate_sample
from h3.distributed import clip_sharded_grad_norm
from h3.flow import flow_matching_loss, sample_video
from h3.h3_fa3_policy import require_h3_fa3_config
from h3.layout import build_layout, patchify, row_timesteps, unpatchify
from h3.model import BiWMH3Transformer
from h3.optimizer import FP32MasterAdamW
from h3.torch_prope import logd4_relative_viewmats, prope_qkv


def sample():
    return dict(target_latents=torch.randn(24, 7, 2, 2),
                anchor_latents=torch.randn(24, 1, 2, 2),
                prompt_embeds=torch.randn(3, 5120),
                text_token_tags=torch.tensor([0, 1, 1]),
                camera_viewmats=torch.eye(4).repeat(7, 1, 1),
                camera_K=torch.eye(3).repeat(7, 1, 1))


def tiny_model():
    model = BiWMH3Transformer(
        num_attention_heads=1, attention_head_dim=128, hidden_size=128,
        num_layers=2, num_refiner_layers=1, ffn_dim=256,
        time_embed_hidden_dim=128, time_embed_dim=64)
    model.install_camera_processors()
    model.set_attention_backend("native")  # CPU numerical check only, no production launch.
    return model


class H3ContractTests(unittest.TestCase):
    def test_patch_order_and_round_trip(self):
        values = torch.arange(2 * 3 * 7 * 4 * 6).reshape(2, 3, 7, 4, 6)
        rows = patchify(values)
        expected = values[0, :, 0, :2, :2].reshape(-1)
        torch.testing.assert_close(rows[0, 0], expected)
        torch.testing.assert_close(unpatchify(rows, 7, 4, 6), values)

    def test_native_clock_and_anchor_camera_alignment(self):
        layout, rows = build_layout(torch.tensor([0, 1]), 7, 4, 6)
        positions = layout["position_ids"]
        self.assertEqual(positions.dtype, torch.float64)
        torch.testing.assert_close(positions[2:2 + rows], positions[2 + rows:2 + 2 * rows])
        times = positions[2 + rows:, 0].reshape(7, rows)[:, 0]
        expected = torch.tensor([0, 1, 5, 9, 13, 17, 18], dtype=torch.float64) * (5 / 3) + 2
        torch.testing.assert_close(times, expected)
        self.assertEqual(len(layout["audio_indices"]), 0)
        self.assertEqual(len(layout["video_indices"]), 8 * rows)

    def test_qwen_target_and_anchor_time_tags(self):
        layout, rows = build_layout(torch.tensor([0, 1]), 7, 4, 6)
        times, inverse = row_timesteps(layout, torch.tensor(.25), rows)
        expanded = times[inverse]
        torch.testing.assert_close(expanded[:2], torch.full((2,), .25))
        torch.testing.assert_close(expanded[2:2 + rows], torch.full((rows,), .999))
        torch.testing.assert_close(expanded[2 + rows:], torch.full((7 * rows,), .25))

    def test_camera_suffix_backward_and_inverse(self):
        torch.manual_seed(1)
        q = torch.randn(1, 2, 3, 128, requires_grad=True)
        k, v = torch.randn_like(q), torch.randn_like(q)
        views = torch.eye(4).repeat(1, 3, 1, 1)
        views[0, :, 0, 3] = torch.tensor([0., 1., 2.])
        intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
        pq, pk, pv, restore = prope_qkv(q, k, v, viewmats=views, Ks=intrinsics)
        for before, after in zip((q, k, v), (pq, pk, pv)):
            torch.testing.assert_close(before[..., :96], after[..., :96], rtol=0, atol=0)
        torch.testing.assert_close(restore(pv), v, rtol=1e-5, atol=1e-5)
        pq.square().mean().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        expected = torch.log1p(torch.tensor([0., 1., 2.])) / 4
        torch.testing.assert_close(logd4_relative_viewmats(views)[0, :, 0, 3], expected)

    def test_no_silent_non_fa3(self):
        self.assertEqual(require_h3_fa3_config("_flash_3"), "_flash_3")
        for invalid in ("auto", "default", "native", "flash"):
            with self.assertRaises(RuntimeError):
                require_h3_fa3_config(invalid)

    def test_cache_rejects_hand_and_bad_poses(self):
        values = validate_sample(sample())
        with tempfile.TemporaryDirectory() as folder:
            save_file(values, str(Path(folder) / "sample.safetensors"))
            (Path(folder) / "manifest.jsonl").write_text(json.dumps(dict(path="sample.safetensors")))
            self.assertEqual(len(CameraDataset(Path(folder) / "manifest.jsonl")), 1)
            validate_sample(CameraDataset(Path(folder) / "manifest.jsonl")[0])
        with self.assertRaises(ValueError):
            validate_sample(dict(values, hand_latents=values["target_latents"]))
        values["camera_viewmats"][0, 0, 3] = 1
        with self.assertRaises(ValueError):
            validate_sample(values)

    def test_fp32_master_resume_preserves_small_bf16_updates(self):
        p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([p], lr=1e-4, weight_decay=0.)
        for _ in range(3):
            p.grad = torch.ones_like(p)
            optimizer.step()
        self.assertLess(float(optimizer.state[p]["master_param"]), 1.)
        other = torch.nn.Parameter(p.detach().clone())
        restored = FP32MasterAdamW([other])
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        p.grad = torch.ones_like(p)
        other.grad = torch.ones_like(other)
        optimizer.step()
        restored.step()
        torch.testing.assert_close(p, other, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[p]["master_param"], restored.state[other]["master_param"])
        self.assertEqual(restored.state[other]["exp_avg"].dtype, torch.float32)

    def test_mixed_dtype_global_clip(self):
        params = [torch.nn.Parameter(torch.zeros(1, dtype=dtype)) for dtype in (torch.float32, torch.bfloat16)]
        params[0].grad = torch.tensor([3.])
        params[1].grad = torch.tensor([4.], dtype=torch.bfloat16)
        norm = clip_sharded_grad_norm(params, max_norm=1.)
        self.assertEqual(float(norm), 5.)
        self.assertAlmostEqual(float(params[0].grad), .6, places=5)
        self.assertAlmostEqual(float(params[1].grad), .8, places=2)

    def test_tiny_camera_model_backward_and_checkpointing(self):
        torch.manual_seed(2)
        values = sample()
        model = tiny_model()
        model.train()
        plain = copy.deepcopy(model)
        model.enable_gradient_checkpointing()
        torch.manual_seed(3)
        loss = flow_matching_loss(model, values)
        loss.backward()
        torch.manual_seed(3)
        expected = flow_matching_loss(plain, values)
        expected.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, expected)
        camera_grad = model.transformer_blocks[0].attn.to_q.weight.grad
        self.assertIsNotNone(camera_grad)
        torch.testing.assert_close(camera_grad, plain.transformer_blocks[0].attn.to_q.weight.grad)
        self.assertGreater(float(camera_grad.abs().sum()), 0)

    def test_sampling_never_uses_target_values(self):
        model = tiny_model().eval()
        values = sample()
        with torch.no_grad():
            first = sample_video(model, values, steps=3, seed=7)
            values["target_latents"].fill_(123.)
            second = sample_video(model, values, steps=3, seed=7)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_controls_are_explicit_per_forward(self):
        model = tiny_model()
        values = sample()
        second = copy.deepcopy(values)
        second["camera_viewmats"][1:, 0, 3] = 1.
        model.enable_gradient_checkpointing()
        loss1 = flow_matching_loss(model, values)
        loss2 = flow_matching_loss(model, second)
        (loss1 + loss2).backward()
        self.assertTrue(torch.isfinite(model.transformer_blocks[0].attn.to_v.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
