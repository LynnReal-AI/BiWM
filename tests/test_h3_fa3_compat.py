"""CPU regression for the forward-only wrapper; actual FA3 needs its GPU gate."""
import unittest
from unittest.mock import patch

import torch
from diffusers.models import attention_dispatch as dispatch

from h3.fa3_autograd import install_fa3_autograd


def native_reference(q, k, v, **kwargs):
    query, key, value = [item.transpose(1, 2) for item in (q, k, v)]
    scores = query @ key.transpose(-1, -2) / query.shape[-1] ** .5
    return (scores.softmax(-1) @ value).transpose(1, 2), scores.logsumexp(-1)


@torch.library.custom_op("biwm_fa3_regression::forward_only", mutates_args=())
def forward_only(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = native_reference(q, k, v)
    return out, lse.transpose(1, 2).contiguous()


class FA3CompatibilityTests(unittest.TestCase):
    def test_forward_only_reproducer_then_native_backward(self):
        torch.manual_seed(1)
        values = [torch.randn(1, 7, 2, 8, requires_grad=True) for _ in range(3)]
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            forward_only(*values)[0].sum().backward()
        with patch.object(dispatch, "_wrapped_flash_attn_3", forward_only), \
                patch.object(dispatch, "flash_attn_3_func", native_reference):
            shim = install_fa3_autograd()
            self.assertIs(install_fa3_autograd(), shim)
            self.assertIs(shim.original_custom_op, forward_only)
            output, lse = shim(*values)
            with torch.no_grad():
                expected, expected_lse = forward_only(*values)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
            output.square().mean().backward()
            for value in values:
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_missing_native_fa3_does_not_fallback(self):
        with patch.object(dispatch, "_wrapped_flash_attn_3", forward_only), \
                patch.object(dispatch, "flash_attn_3_func", None):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                install_fa3_autograd()
            self.assertIs(dispatch._wrapped_flash_attn_3, forward_only)


if __name__ == "__main__":
    unittest.main()
