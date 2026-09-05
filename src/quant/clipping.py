"""Clipping-policy public surface.

The numerical kernels currently remain co-located in ``core`` because they
share the reference fake-quantizer and are still compact. This module owns the
stable clipping API so implementations can move without affecting callers.
"""

from .core import (
    choose_awq_clip_ratio,
    choose_weight_mse_clip_ratio,
    normalized_output_error,
)

__all__ = [
    "choose_awq_clip_ratio",
    "choose_weight_mse_clip_ratio",
    "normalized_output_error",
]
