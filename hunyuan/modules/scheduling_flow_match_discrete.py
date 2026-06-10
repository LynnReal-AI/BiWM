# Copied+slimmed from minWM/HY15 (imports rewritten to hunyuan.* local dependencies; see the hunyuan/ module documentation)
# Modified from diffusers; Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
        n_tokens: Optional[int] = None,
        flux_base_token=256.0,
        flux_max_token=4096.0,
        flux_shift_factor=1.0,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        # Training sigma table: fixed 1000-step schedule for add_noise / get_sigma
        train_sigmas = torch.linspace(1, 0, num_train_timesteps + 1)[:-1]  # [1000]
        if shift != 1.0:
            train_sigmas = self.sd3_time_shift(train_sigmas)
        self.train_sigmas = train_sigmas  # indexed by integer 0..num_train_timesteps-1

        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler", "cm"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device=None,
        n_tokens: Optional[int] = None,
        **kwargs,
    ):
        num_inference_steps = num_inference_steps or self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        if self.config.use_flux_shift:
            if n_tokens is None:
                n_tokens = self.config.n_tokens
            mu = self.get_lin_function(
                x1=self.config.flux_base_token, y1=self.config.flux_base_shift,
                x2=self.config.flux_max_token, y2=self.config.flux_max_shift,
            )(n_tokens)
            sigmas = self.flux_time_shift(mu, self.config.flux_shift_factor, sigmas)
        else:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas.to(device=device)
        self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(
            device=device, dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    @staticmethod
    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        n_tokens: Optional[int] = None,
        return_dict: bool = True,
        solver: Optional[str] = None,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps to step() is not supported. "
                "Pass one of scheduler.timesteps as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        # Allow runtime solver override; fall back to config
        active_solver = solver if solver is not None else self.config.solver

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if active_solver == "euler":
            prev_sample = sample + model_output.float() * dt
        elif active_solver == "cm":
            # 1. Predict x_0: x̂_0 = x_t - σ_t * v_θ(x_t, t)
            sigma_t = self.sigmas[self.step_index]
            x_0_hat = sample - sigma_t * model_output.float()
            # 2. Re-noise: x_{t-1} = (1 - σ_{t-1}) * x̂_0 + σ_{t-1} * ε
            sigma_next = self.sigmas[self.step_index + 1]
            noise = torch.randn(x_0_hat.shape, dtype=x_0_hat.dtype,
                                device=x_0_hat.device, generator=generator)
            prev_sample = (1 - sigma_next) * x_0_hat + sigma_next * noise

        else:
            raise ValueError(
                f"Solver {active_solver} not supported."
            )

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def get_sigma(self, indices: torch.Tensor) -> torch.Tensor:
        """Return sigma for integer training indices (0..num_train_timesteps-1)."""
        self.train_sigmas = self.train_sigmas.to(indices.device)
        indices = indices.clamp(0, len(self.train_sigmas) - 1)
        return self.train_sigmas[indices]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching: x_t = (1 - sigma) * x_0 + sigma * noise."""
        sigma = self.get_sigma(timestep)
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(original_samples)

    def pred_noise_to_pred_video(
        self,
        pred_noise: torch.Tensor,
        noise_input_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """x_0 = x_t - sigma * v."""
        sigma = self.get_sigma(timestep)
        while sigma.dim() < pred_noise.dim():
            sigma = sigma.unsqueeze(-1)
        return noise_input_latent - sigma * pred_noise

    def __len__(self):
        return self.config.num_train_timesteps
