# SPDX-License-Identifier: Apache-2.0
"""BiWM H3 second phase: clean FramePack history + noisy next chunk, FM only."""
import math
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
from h3.model import BiWMH3Transformer, CameraControl, get_dtype
from h3.layout import build_layout, patchify
from pipelines.wan.history_compress_framepack import FramePackPastCompressor


class FramePackH3(BiWMH3Transformer):
    def install_framepack(self):
        hidden, channels = self.proj_in.weight.shape[0], 24
        base = torch.nn.Conv3d(channels, hidden, (1, 2, 2), stride=(1, 2, 2))
        base.weight.data.copy_(self.proj_in.weight.detach().cpu().reshape(hidden, channels, 1, 2, 2))
        base.bias.data.copy_(self.proj_in.bias.detach().cpu())
        self.history_compressor = FramePackPastCompressor(base, hidden)

    def forward(self, sample, noisy, anchor, target_t, start):
        """History is GT during teacher forcing; all visible rows attend bidirectionally."""
        clean = sample['target_latents']
        _, total, height, width = clean.shape
        count = noisy.shape[2]
        device = noisy.device
        tags = sample['text_token_tags']
        text_count = len(tags)
        base, rows = build_layout(tags, count, height, width)
        # Always call the compressor, even for the first chunk, for FSDP order.
        history = clean[:, :start].float().unsqueeze(0) if start else anchor
        memory, bounds = self.history_compressor(history)
        if not start:
            memory, bounds = memory[:, :0], bounds[:, :, :0]
        memory_count = memory.shape[1]
        target_start = text_count + rows + memory_count
        sequence = target_start + count * rows
        positions = torch.zeros(sequence, 3, device=device, dtype=torch.float64)
        positions[:text_count + rows] = base['position_ids'][:text_count + rows]
        positions[target_start:] = base['position_ids'][text_count + rows:]
        spans = torch.tensor([1, 4, 4, 4, 4], device=device, dtype=torch.float64)
        spans = spans[torch.arange(total, device=device) % 5] * (5 / 3)
        times = torch.cat((spans.new_zeros(1), spans.cumsum(0)[:-1])) + text_count
        positions[target_start:, 0] = times[start:start + count].repeat_interleave(rows)
        memory_indices = torch.arange(text_count + rows, target_start, device=device)
        midpoint = bounds[0].mean(-1).floor().double()
        memory_frames = midpoint[0].long()
        if memory_count:
            positions[memory_indices, 0] = times[memory_frames]
            for axis, size in enumerate((height, width), start=1):
                ratio = size / math.sqrt(height * width)
                positions[memory_indices, axis] = ((1 - ratio) / 2 + ratio * midpoint[axis] / (size // 2)) * 32
        text_indices = torch.arange(text_count, device=device)
        anchor_indices = torch.arange(text_count, text_count + rows, device=device)
        target_indices = torch.arange(target_start, sequence, device=device)
        video_indices = torch.cat((anchor_indices, target_indices))
        token_tags = torch.cat((tags, torch.zeros(sequence - text_count, dtype=torch.long, device=device)))
        row_times = target_t.expand(sequence).clone()
        row_times[anchor_indices] = .999
        row_times[memory_indices] = 1.
        timesteps, inverse = torch.unique(row_times, sorted=True, return_inverse=True)
        frame_ids = torch.cat((torch.zeros(rows, dtype=torch.long, device=device),
                               torch.arange(start, start + count, device=device).repeat_interleave(rows),
                               memory_frames))
        camera_indices = torch.cat((video_indices, memory_indices))
        control = CameraControl(camera_indices,
                                sample['camera_viewmats'].unsqueeze(0).index_select(1, frame_ids),
                                sample['camera_K'].unsqueeze(0).index_select(1, frame_ids))
        video = self.proj_in(torch.cat((patchify(anchor), patchify(noisy)), 1).to(get_dtype(self.proj_in)))
        text = self.token_refiner(self.context_embedder(sample['prompt_embeds'].unsqueeze(0).to(get_dtype(self.context_embedder))))
        packed = text.new_zeros(1, sequence, text.shape[-1])
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, memory_indices, memory.to(text.dtype))
        temb = self.time_embedder(self.time_proj(timesteps).to(get_dtype(self.time_embedder)))
        adaln = inverse * MINIMAX_H3_MODALITY_NUM + token_tags
        rotary = self.rope(positions)
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed = self._gradient_checkpointing_func(block, packed, temb, adaln, rotary, control)
            else:
                packed = block(packed, temb, adaln, rotary, control)
        packed = self.norm_out(packed, temb, inverse).to(get_dtype(self.proj_out))
        return self.proj_out(packed).index_select(1, target_indices)


def loss(model, sample):
    frames = sample['target_latents'].shape[1]
    chunks = 1 + (frames - 7) // 5
    chunk = int(torch.randint(chunks, (), device=sample['target_latents'].device))
    start, count = (0, 7) if chunk == 0 else (7 + (chunk - 1) * 5, 5)
    clean = sample['target_latents'][:, start:start + count].float().unsqueeze(0)
    anchor = sample['anchor_latents'].float().unsqueeze(0)
    anchor = .999 * anchor + .001 * torch.randn_like(anchor)
    noise = torch.randn_like(clean)
    u = torch.rand((), device=clean.device)
    sigma = 3 * u / (1 + 2 * u)
    velocity = model(sample, (1 - sigma) * clean + sigma * noise, anchor, 1 - sigma, start)
    return torch.nn.functional.mse_loss(velocity.float(), patchify(clean - noise))
