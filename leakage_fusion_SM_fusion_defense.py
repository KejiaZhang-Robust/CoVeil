#!/usr/bin/env python3
"""CoVeil edge-side fusion: privacy-aware synchronized-token selection."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable

import torch

from leakage_fusion_core import apply_repetition_penalty, run_dataset


def _mask_to_ids(scores: torch.Tensor, ids: Iterable[int] | None) -> torch.Tensor:
    if not ids:
        return scores
    valid = [int(token_id) for token_id in ids if 0 <= int(token_id) < scores.shape[-1]]
    if not valid:
        return scores
    masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
    index = torch.tensor(valid, dtype=torch.long, device=scores.device)
    masked[index] = scores[index]
    return masked


def _unique_sorted(ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(ids.long())
    return unique[torch.argsort(scores[unique].float(), descending=True)]


def select_synchronized_token(
    z_l: torch.Tensor,
    z_s: torch.Tensor,
    alpha: float,
    privacy_weight: float,
    candidate_topk: int,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
    answer_complete: bool,
    answer_phase: bool,
    label_token_ids,
    eos_token_id: int | None,
) -> Dict[str, Any]:
    """Minimize the released edge-side utility-privacy objective over candidates."""
    fused = float(alpha) * z_l.float() + (1.0 - float(alpha)) * z_s.float()
    fused_select = apply_repetition_penalty(fused, generated_ids, repetition_penalty)
    large_select = apply_repetition_penalty(z_l.float(), generated_ids, repetition_penalty)
    small_select = apply_repetition_penalty(z_s.float(), generated_ids, repetition_penalty)

    if eos_token_id is not None and not answer_complete and int(eos_token_id) < fused.shape[-1]:
        for scores in (fused_select, large_select, small_select):
            scores[int(eos_token_id)] = torch.finfo(scores.dtype).min

    valid_action_ids = [
        int(token_id)
        for token_id in label_token_ids
        if 0 <= int(token_id) < z_l.shape[-1]
    ]
    action_ids = valid_action_ids if answer_phase and valid_action_ids else None
    fused_action = _mask_to_ids(fused_select, action_ids)
    large_action = _mask_to_ids(large_select, action_ids)
    small_action = _mask_to_ids(small_select, action_ids)

    vocab_size = int(z_l.shape[-1])
    if action_ids:
        allowed = torch.tensor(action_ids, dtype=torch.long, device=z_l.device)
        k = max(1, min(int(candidate_topk), int(allowed.numel())))
        candidate_ids = allowed[torch.topk(large_action[allowed], k=k).indices]
    else:
        k = max(1, min(int(candidate_topk), vocab_size))
        candidate_ids = torch.topk(large_action, k=k).indices
    fused_top1 = int(torch.argmax(fused_action).item())
    large_top1 = int(torch.argmax(large_action).item())
    extras = torch.tensor([fused_top1, large_top1], device=z_l.device)
    candidate_ids = _unique_sorted(torch.cat([candidate_ids, extras]), large_action)

    edge_weight = 1.0 - float(alpha)
    reference_logits = float(alpha) * large_action.float()
    reference_logits = reference_logits.clone()
    reference_logits[candidate_ids] += edge_weight * small_action[candidate_ids].float()
    reference_winner = int(torch.argmax(reference_logits).item())
    if not torch.any(candidate_ids == reference_winner):
        candidate_ids = _unique_sorted(
            torch.cat([candidate_ids, torch.tensor([reference_winner], device=z_l.device)]),
            large_action,
        )
        reference_logits = float(alpha) * large_action.float()
        reference_logits = reference_logits.clone()
        reference_logits[candidate_ids] += edge_weight * small_action[candidate_ids].float()
        reference_winner = int(torch.argmax(reference_logits).item())

    winner_mask = (candidate_ids == reference_winner).float()
    utility_gain = edge_weight * small_action[candidate_ids].float() * (
        winner_mask - 1.0 / float(candidate_ids.numel())
    )
    utility_cost = -utility_gain
    small_values = small_action[candidate_ids].float()
    large_values = large_action[candidate_ids].float()
    privacy_cost = (small_values - small_values.mean()) - (large_values - large_values.mean())
    objective = utility_cost + float(privacy_weight) * privacy_cost
    selected_id = int(candidate_ids[torch.argmin(objective)].item())

    return {
        "next_token_id": selected_id,
        "observed_scores": z_l.detach(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--large_model", required=True, help="Model path or Hugging Face identifier.")
    parser.add_argument("--small_model", required=True, help="Model path or Hugging Face identifier.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--privacy_weight", type=float, default=1.0)
    parser.add_argument("--candidate_topk", type=int, default=50)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--record_steps", type=int, default=160)
    parser.add_argument("--trace_topk", type=int, default=100)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if args.privacy_weight < 0 or args.candidate_topk <= 0:
        raise ValueError("privacy_weight must be non-negative and candidate_topk must be positive.")
    run_dataset(args, selector=select_synchronized_token, mode="edge")


if __name__ == "__main__":
    main()
