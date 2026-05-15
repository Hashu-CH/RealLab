# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Vendored from rsl_rl v2.3.3 (rsl_rl/networks/memory.py). Parameter
# names are unchanged so state_dicts trained against upstream load here.

from __future__ import annotations

import torch
import torch.nn as nn

from utils.rsl_rl_compat.utils import unpad_trajectories


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:
                if isinstance(self.hidden_states, tuple):
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:
                if isinstance(self.hidden_states, tuple):
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()
