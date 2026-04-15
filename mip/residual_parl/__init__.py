"""Residual PARL (Policy-Agnostic RL) module.

PARL trains the actor via Best-of-N + Q-gradient refinement + BC distillation,
never differentiating through the critic. This avoids gradient issues with
tanh squashing and action clipping that can destabilize residual SAC.
"""

from mip.residual_parl.parl_agent import ResidualPARLAgent, PARLConfig
