"""Minimal vendored slice of rsl_rl v2.3.3.

Contains only the symbols needed at inference time by `ActorCriticCNN`
and `ActorCriticCNNGRU` so the trained checkpoints in `src/models/` can
load without taking a dependency on the full `rsl_rl` package.

Source: https://github.com/leggedrobotics/rsl_rl (v2.3.3, BSD-3-Clause).
Parameter names match the upstream classes so `load_state_dict` succeeds.
"""
