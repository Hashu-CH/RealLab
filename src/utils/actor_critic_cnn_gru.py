"""CNN actor-critics ported from WheeledLab for RealLab inference.

For racing: [(1,40,80) camera, (vel, ang, a_t-1) proprioceptive data].

The classes here mirror
`WheeledLab/source/wheeledlab_rl/wheeledlab_rl/utils/actor_critic_cnn_gru.py`
verbatim apart from the three import lines, which point at the vendored
rsl_rl slice in `utils.rsl_rl_compat` instead of an installed `rsl_rl`
package. Submodule layout and parameter names are unchanged, so the
checkpoints in `src/models/racing_cnn.pt` and
`src/models/racing_cnn_rnn.pt` load via `load_state_dict` without
remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.rsl_rl_compat.actor_critic import ActorCritic
from utils.rsl_rl_compat.memory import Memory
from utils.rsl_rl_compat.utils import resolve_nn_activation


def _build_cnn(
    image_shape: tuple[int, int, int],
    channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    out_dim: int,
    activation: str,
) -> nn.Module:
    """Conv stack -> flatten -> linear projection to `out_dim`."""
    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError(
            "cnn_channels, cnn_kernel_sizes, cnn_strides must have equal length; "
            f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
        )

    channel_in = image_shape[0]
    conv_layers: list[nn.Module] = []
    for channel_out, k, s in zip(channels, kernel_sizes, strides):
        conv_layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=k, stride=s))
        conv_layers.append(resolve_nn_activation(activation))
        channel_in = channel_out
    conv_layers.append(nn.Flatten())
    conv = nn.Sequential(*conv_layers)

    with torch.no_grad():
        flat_dim = conv(torch.zeros(1, *image_shape)).shape[-1]

    return nn.Sequential(conv, nn.Linear(flat_dim, out_dim), resolve_nn_activation(activation))


class ActorCriticCNN(ActorCritic):
    """CNN + MLP actor-critic, no recurrence."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (1, 40, 80),
        cnn_channels: list[int] = (16, 32),
        cnn_kernel_sizes: list[int] = (5, 3),
        cnn_strides: list[int] = (2, 2),
        cnn_out_dim: int = 64,
        actor_hidden_dims: list[int] = (64, 64),
        critic_hidden_dims: list[int] = (64, 64),
        activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        image_shape_t = tuple(image_shape)
        n_img = int(image_shape_t[0] * image_shape_t[1] * image_shape_t[2])
        if n_img > num_actor_obs:
            raise ValueError(
                f"ActorCriticCNN: image_shape numel ({n_img}) exceeds "
                f"num_actor_obs ({num_actor_obs}). Obs layout likely changed — "
                f"update image_shape in the policy cfg."
            )
        if n_img > num_critic_obs:
            raise ValueError(
                f"ActorCriticCNN: image_shape numel ({n_img}) exceeds "
                f"num_critic_obs ({num_critic_obs})."
            )

        num_proprio_a = num_actor_obs - n_img
        num_proprio_c = num_critic_obs - n_img
        enc_dim_a = cnn_out_dim + num_proprio_a
        enc_dim_c = cnn_out_dim + num_proprio_c

        super().__init__(
            num_actor_obs=enc_dim_a,
            num_critic_obs=enc_dim_c,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        self.image_shape = image_shape_t
        self._n_img = n_img

        self.cnn = _build_cnn(
            self.image_shape,
            list(cnn_channels),
            list(cnn_kernel_sizes),
            list(cnn_strides),
            cnn_out_dim,
            activation,
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        leading, D = obs.shape[:-1], obs.shape[-1]
        flat = obs.reshape(-1, D)
        img = flat[:, : self._n_img].reshape(-1, *self.image_shape)
        proprio = flat[:, self._n_img:]
        feat = self.cnn(img)
        enc = torch.cat([feat, proprio], dim=-1)
        return enc.reshape(*leading, -1)

    def act(self, observations, **kwargs):
        return super().act(self._encode(observations))

    def act_inference(self, observations):
        return super().act_inference(self._encode(observations))

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(self._encode(critic_observations))


class ActorCriticCNNGRU(ActorCritic):
    """CNN encoder -> GRU memory -> MLP heads."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (1, 40, 80),
        cnn_channels: list[int] = (16, 32),
        cnn_kernel_sizes: list[int] = (5, 3),
        cnn_strides: list[int] = (2, 2),
        cnn_out_dim: int = 64,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        actor_hidden_dims: list[int] = (64, 64),
        critic_hidden_dims: list[int] = (64, 64),
        activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        self.image_shape = tuple(image_shape)
        self._n_img = int(self.image_shape[0] * self.image_shape[1] * self.image_shape[2])

        if self._n_img > num_actor_obs:
            raise ValueError(
                f"ActorCriticCNNGRU: image_shape numel ({self._n_img}) exceeds "
                f"num_actor_obs ({num_actor_obs}). Obs layout likely changed — "
                f"update image_shape in the policy cfg."
            )
        if self._n_img > num_critic_obs:
            raise ValueError(
                f"ActorCriticCNNGRU: image_shape numel ({self._n_img}) exceeds "
                f"num_critic_obs ({num_critic_obs})."
            )

        self.cnn = _build_cnn(
            self.image_shape,
            list(cnn_channels),
            list(cnn_kernel_sizes),
            list(cnn_strides),
            cnn_out_dim,
            activation,
        )

        num_proprio_a = num_actor_obs - self._n_img
        num_proprio_c = num_critic_obs - self._n_img
        enc_dim_a = cnn_out_dim + num_proprio_a
        enc_dim_c = cnn_out_dim + num_proprio_c

        self.memory_a = Memory(
            enc_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )
        self.memory_c = Memory(
            enc_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        leading, D = obs.shape[:-1], obs.shape[-1]
        flat = obs.reshape(-1, D)
        img = flat[:, : self._n_img].reshape(-1, *self.image_shape)
        proprio = flat[:, self._n_img:]
        feat = self.cnn(img)
        enc = torch.cat([feat, proprio], dim=-1)
        return enc.reshape(*leading, -1)

    def act(self, observations, masks=None, hidden_states=None):
        enc = self._encode(observations)
        hid = self.memory_a(enc, masks, hidden_states)
        return super().act(hid.squeeze(0))

    def act_inference(self, observations):
        enc = self._encode(observations)
        hid = self.memory_a(enc)
        return super().act_inference(hid.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        enc = self._encode(critic_observations)
        hid = self.memory_c(enc, masks, hidden_states)
        return super().evaluate(hid.squeeze(0))

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
