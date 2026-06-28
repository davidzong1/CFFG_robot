"""LLM-driven supervisor that adjusts reward weights during RL training.

The supervisor runs as a background daemon thread inside the training
process. It periodically:

  1. Reads TensorBoard scalars from ``log_dir``.
  2. Samples frames from the most recent training videos.
  3. Asks an LLM to diagnose and propose a reward-weight patch.
  4. Validates the patch (schema + bounds + cooldown + killswitch).
  5. Writes the patch atomically and mutates the live ``RewardManager``.

Use ``SupervisorCallbacks`` to wire the supervisor into non-RL frameworks.
"""

from .config import SupervisorConfig
from .interfaces import SupervisorCallbacks
from .objective import TrainingObjective
from .supervisor import Supervisor

__all__ = ["Supervisor", "SupervisorCallbacks", "SupervisorConfig", "TrainingObjective"]
