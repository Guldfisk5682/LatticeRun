"""Serial on-policy distillation records, loss, and workflows."""

from .artifacts import RolloutRecord, read_valid_rollouts, save_teacher_artifact
from .loss import (
    backward_chunked_forward_kl,
    causal_completion_view,
    chunked_forward_kl_from_hidden,
    dense_forward_kl_from_hidden,
)

__all__ = [
    "RolloutRecord",
    "backward_chunked_forward_kl",
    "causal_completion_view",
    "chunked_forward_kl_from_hidden",
    "dense_forward_kl_from_hidden",
    "read_valid_rollouts",
    "save_teacher_artifact",
]
