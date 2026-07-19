import functools
import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange


class LTXRopeType(Enum):
    """RoPE 类型枚举"""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> torch.Tensor:
    """应用 RoPE（默认使用 Split 模式）"""
    return apply_split_rotary_emb(input_tensor, *freqs_cis)


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """
    应用 Split RoPE (旋转位置编码) 到输入张量

    RoPE 核心公式 (2D 旋转):
        [x', y'] = [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]

    Args:
        input_tensor: Q 或 K 张量, shape [batch_size, num_heads, num_tokens, head_dim]
                      例如: [1, 32, 25515, 128]
        cos_freqs: cos(position × frequency), shape [batch_size, num_heads, num_tokens, head_dim//2]
                   例如: [1, 32, 25515, 64]
        sin_freqs: sin(position × frequency), shape [batch_size, num_heads, num_tokens, head_dim//2]
                   例如: [1, 32, 25515, 64]

    Returns:
        旋转后的张量, shape 与 input_tensor 相同
    """
    needs_reshape = False
    # print(input_tensor.shape, cos_freqs.shape, sin_freqs.shape)

    # 如果 input 维度不匹配，先 reshape 成 [batch_size, num_heads, num_tokens, head_dim]
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # 将 head_dim 拆分成 (2, head_dim//2)，即把相邻的两个元素配对
    # [batch_size, num_heads, num_tokens, head_dim] -> [batch_size, num_heads, num_tokens, 2, head_dim//2]
    # 例如: [1, 32, 25515, 128] -> [1, 32, 25515, 2, 64]
    # 这样 [..., 0, :] 是所有偶数位置 (x), [..., 1, :] 是所有奇数位置 (y)
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)

    # 分离出 x 和 y (用于旋转公式)
    first_half_input = split_input[..., :1, :]   # x: [batch, heads, tokens, 1, 64]
    second_half_input = split_input[..., 1:, :]  # y: [batch, heads, tokens, 1, 64]

    # 第一步: output = input * cos
    # cos_freqs 需要 unsqueeze(-2) 从 [b,h,t,64] -> [b,h,t,1,64] 以便广播
    output = split_input * cos_freqs.unsqueeze(-2)  # [b, h, t, 2, 64]

    first_half_output = output[..., :1, :]   # x·cos: [b, h, t, 1, 64]
    second_half_output = output[..., 1:, :]  # y·cos: [b, h, t, 1, 64]

    # 第二步: 应用旋转公式
    # x' = x·cos - y·sin  (addcmul_ 是 inplace 的 a += b * c)
    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)  # x·cos - y·sin
    # y' = y·cos + x·sin
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)   # y·cos + x·sin

    # 合并回原来的 shape: [b, h, t, 2, 64] -> [b, h, t, 128]
    output = rearrange(output, "... d r -> ... (d r)")

    # 如果之前做了 reshape，恢复原始 shape
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta # 10000.0
    start = 1 # 1
    end = theta # 10000.0
    n_elem = 2 * positional_embedding_max_pos_count # 3 * 2 = 6
    # print("n_elem:",n_elem)
    # print("start:",start)
    # print("end:",end)
    # print("theta:",theta)
    # print("positional_embedding_max_pos_count:",positional_embedding_max_pos_count)
    # print("positional_embedding_theta:",positional_embedding_theta)
    # print("inner_dim:",inner_dim)
    # print("positional_embedding_max_pos_count:",positional_embedding_max_pos_count)
    # print("positional_embedding_theta:",positional_embedding_theta)
    # print("inner_dim:",inner_dim)
    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem, ## 这个是取多少个样本，应该是4096/2/3  得到每个维度需要的频率数
            dtype=torch.float32,
        )
    )
    # print("indices:",indices)
    # print("indices.shape:",indices.shape)
    # print("indices max:",indices.max())
    # print("indices min:",indices.min())
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2 ## 输出结果也是单调的区间

    return indices


def get_fractional_positions(
    indices_grid: torch.Tensor,
    max_pos: list[int],
    normalize: bool = True,
) -> torch.Tensor:
    """
    获取位置坐标，可选是否归一化。

    Args:
        indices_grid: [batch, n_dims, num_tokens] 绝对位置索引
        max_pos: [n_dims] 每个维度的最大位置值，用于归一化
        normalize: 是否归一化到 [0, 1] 范围
            - True (默认): position / max_pos，用于标准推理
            - False: 直接使用绝对位置，用于流式生成实验

    Returns:
        fractional_positions: [batch, num_tokens, n_dims] 位置坐标
    """
    n_pos_dims = indices_grid.shape[1]
    # indices_grid: [1, 3, 25515]
    # max_pos: [20, 2048, 2048]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )

    if normalize:
        # 归一化到 [0, 1] 范围 (标准模式)
        fractional_positions = torch.stack(
            [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
            dim=-1,
        )
    else:
        # 直接使用绝对位置 (流式实验模式)
        # 注意：使用未归一化位置训练的模型才能正常工作
        fractional_positions = torch.stack(
            [indices_grid[:, i].float() for i in range(n_pos_dims)],
            dim=-1,
        )

    return fractional_positions


def generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
    use_middle_indices_grid: bool,
    normalize_positions: bool = True,
    time_yarn_config: dict = None,
) -> torch.Tensor:
    """
    生成 RoPE 频率。

    Args:
        indices: 频率基底 [D]
        indices_grid: 位置网格
        max_pos: 最大位置值
        use_middle_indices_grid: 是否使用 patch 中点
        normalize_positions: 是否归一化位置到 [0, 1]
        time_yarn_config: YaRN 配置 dict，None 时走原始路径。包含:
            - scale: 外推倍率
            - train_frac_pos_max: 训练数据最大 frac_pos
            - beta_fast: 高频边界 (训练范围内旋转圈数)
            - beta_slow: 低频边界
            - extrapolation_factor: 外推权重
    """
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(
        indices_grid, max_pos, normalize=normalize_positions
    )
    indices = indices.to(device=fractional_positions.device)

    if time_yarn_config is not None and time_yarn_config.get('scale', 1.0) > 1.0:
        # ===== YaRN for time dimension only (参考 wan/modules/yarn_rope_21.py) =====
        # 空间维度完全不变
        s = time_yarn_config['scale']
        train_frac = time_yarn_config.get('train_frac_pos_max', 0.33)
        beta_fast = time_yarn_config.get('beta_fast', 2.0)
        beta_slow = time_yarn_config.get('beta_slow', 0.1)
        ext_factor = time_yarn_config.get('extrapolation_factor', 1.0)

        # 每个频率在训练 frac_pos 范围内的实际旋转圈数
        rotations = indices * train_frac / math.pi

        # ramp: 0 (低频,旋转<beta_slow) → 1 (高频,旋转>beta_fast)
        ramp = ((rotations - beta_slow) / (beta_fast - beta_slow + 1e-6)).clamp(0, 1)
        mask = ramp * ext_factor

        # 频率缩放: 高频→1.0 (不变), 低频→1/scale (插值)
        scale_per_freq = mask * 1.0 + (1.0 - mask) * (1.0 / s)
        time_indices = indices * scale_per_freq

        # 注意: mscale 不乘进频率角度！它是 attention 幅度补偿，
        # 在 split_freqs_cis 的 cos/sin 阶段应用（通过 time_yarn_config 传递）

        angle_factors = fractional_positions.unsqueeze(-1) * 2 - 1  # [B, N, 3, 1]
        time_freqs = time_indices * angle_factors[:, :, 0:1, :]      # [B, N, 1, D]
        spatial_freqs = indices * angle_factors[:, :, 1:, :]           # [B, N, 2, D]
        freqs = torch.cat([time_freqs, spatial_freqs], dim=2)          # [B, N, 3, D]
        freqs = freqs.transpose(-1, -2).flatten(2)
    else:
        ## 原始行为，完全不变
        freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)

    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    # 输入: [batch_size, num_tokens, freq_dim]
    #   - batch_size = 1
    #   - num_tokens = 25515 (= 63帧 × 15高 × 27宽)
    #   - freq_dim = 2048 = num_heads × (head_dim // 2) = 32 × 64
    batch_size = cos_freq.shape[0]   # 1
    num_tokens = cos_freq.shape[1]   # 25515

    # 拆分成多头: [batch_size, num_tokens, freq_dim] -> [batch_size, num_tokens, num_heads, head_dim//2]
    #   - num_heads = 32
    #   - head_dim // 2 = 64 (RoPE 只作用于每个头一半的维度)
    cos_freq = cos_freq.reshape(batch_size, num_tokens, num_attention_heads, -1)  # [1, 25515, 32, 64]
    sin_freq = sin_freq.reshape(batch_size, num_tokens, num_attention_heads, -1)  # [1, 25515, 32, 64]

    # 交换维度以匹配 attention 的 [batch_size, num_heads, num_tokens, head_dim] 格式
    # [batch_size, num_tokens, num_heads, head_dim//2] -> [batch_size, num_heads, num_tokens, head_dim//2]
    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # [1, 32, 25515, 64]
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # [1, 32, 25515, 64]
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
    normalize_positions: bool = True,
    time_yarn_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos/sin 频率。

    Args:
        normalize_positions: 是否归一化位置到 [0, 1]
        time_yarn_config: YaRN 配置 dict，None 时原始行为
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(
        indices, indices_grid, max_pos, use_middle_indices_grid,
        normalize_positions=normalize_positions,
        time_yarn_config=time_yarn_config,
    )
    #     freqs[token] = [
    #     T维度的682个频率值,
    #     H维度的682个频率值,
    #     W维度的682个频率值
    # ]
    # = 682 × 3 = 2046 维
    # print("freqs:",freqs.shape)## 1, 25515, 2046
    # 使用 Split RoPE（官方 LTX-2 默认）
    expected_freqs = dim // 2  # 4096/2 = 2048
    current_freqs = freqs.shape[-1]  # 2046
    pad_size = expected_freqs - current_freqs
    cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
