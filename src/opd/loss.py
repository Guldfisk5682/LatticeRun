"""Exact full-vocabulary forward-KL with bounded vocabulary memory."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_completion_view(
    hidden_states: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align hidden[t] with target token[t+1] and its completion mask."""

    if hidden_states.shape[:-1] != completion_mask.shape:
        raise ValueError(
            "completion mask must match hidden state batch/sequence dimensions"
        )
    return hidden_states[:, :-1, :], completion_mask[:, 1:].bool()


def _forward_kl_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student), using every vocabulary entry."""

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)


def dense_forward_kl_from_hidden(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation used to verify token chunking equivalence."""

    student_view, active = causal_completion_view(student_hidden, completion_mask)
    teacher_view, teacher_active = causal_completion_view(
        teacher_hidden, completion_mask
    )
    if not torch.equal(active, teacher_active):
        raise RuntimeError("student/teacher completion masks diverged")
    student_logits = F.linear(student_view, student_lm_head_weight)
    teacher_logits = F.linear(teacher_view, teacher_lm_head_weight)
    per_token = _forward_kl_per_token(student_logits, teacher_logits)
    if not active.any():
        raise ValueError("completion mask has no supervised tokens")
    return per_token.masked_select(active).mean()


def chunked_forward_kl_from_hidden(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Mathematically exact token-chunked full-vocabulary forward KL.

    Each chunk still materializes the complete vocabulary. Chunk numerators are
    summed and divided once by the total active-token count, avoiding the
    unequal-chunk weighting bug caused by averaging chunk means.
    """

    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    student_view, active = causal_completion_view(student_hidden, completion_mask)
    teacher_view, _ = causal_completion_view(teacher_hidden, completion_mask)
    flat_student = student_view.reshape(-1, student_view.shape[-1])
    flat_teacher = teacher_view.reshape(-1, teacher_view.shape[-1])
    flat_active = active.reshape(-1)
    active_count = flat_active.sum()
    if int(active_count.item()) == 0:
        raise ValueError("completion mask has no supervised tokens")
    numerator = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_student.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, flat_student.shape[0])
        chunk_mask = flat_active[start:stop]
        if not chunk_mask.any():
            continue
        student_logits = F.linear(flat_student[start:stop], student_lm_head_weight)
        with torch.no_grad():
            teacher_logits = F.linear(flat_teacher[start:stop], teacher_lm_head_weight)
        per_token = _forward_kl_per_token(student_logits, teacher_logits)
        numerator = numerator + per_token.masked_select(chunk_mask).sum()
    return numerator / active_count


def backward_chunked_forward_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    token_chunk_size: int = 256,
    gradient_denominator: int | None = None,
) -> float:
    """Cap vocabulary memory while traversing the student backbone once.

    Each vocabulary chunk is differentiated only to its student-hidden slice.
    Those slice gradients are assembled into one hidden-state gradient, then a
    single backward call traverses the expensive transformer graph. When
    ``gradient_denominator`` is provided, gradients are normalized by the
    accumulation group's global valid-token count while the returned scalar
    remains this trajectory's token mean for diagnostics.
    """

    student_view, active = causal_completion_view(student_hidden, completion_mask)
    teacher_view, _ = causal_completion_view(teacher_hidden, completion_mask)
    flat_student = student_view.reshape(-1, student_view.shape[-1])
    flat_teacher = teacher_view.reshape(-1, teacher_view.shape[-1])
    flat_active = active.reshape(-1)
    active_count = int(flat_active.sum().item())
    if active_count == 0:
        raise ValueError("completion mask has no supervised tokens")
    denominator = active_count if gradient_denominator is None else gradient_denominator
    if denominator <= 0:
        raise ValueError("gradient_denominator must be positive")
    flat_hidden_gradient = torch.zeros_like(flat_student)
    loss_sum = 0.0
    for start in range(0, flat_student.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, flat_student.shape[0])
        chunk_mask = flat_active[start:stop]
        if not chunk_mask.any():
            continue
        student_chunk = flat_student[start:stop]
        student_logits = F.linear(student_chunk, student_lm_head_weight)
        with torch.no_grad():
            teacher_logits = F.linear(flat_teacher[start:stop], teacher_lm_head_weight)
        chunk_sum = (
            _forward_kl_per_token(student_logits, teacher_logits)
            .masked_select(chunk_mask)
            .sum()
        )
        loss_sum += float(chunk_sum.detach().item())
        loss = chunk_sum / denominator
        chunk_gradient = torch.autograd.grad(loss, student_chunk)[0]
        flat_hidden_gradient[start:stop].copy_(chunk_gradient)
        del student_logits, teacher_logits, chunk_sum, loss, chunk_gradient
    hidden_gradient = torch.zeros_like(student_hidden)
    hidden_gradient[:, :-1, :].copy_(flat_hidden_gradient.reshape_as(student_view))
    student_hidden.backward(hidden_gradient)
    return loss_sum / active_count
