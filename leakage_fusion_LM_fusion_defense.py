#!/usr/bin/env python3
"""CoVeil cloud-side fusion: privacy-aware sparse SLM-logit upload."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from leakage_fusion_core import apply_repetition_penalty, run_dataset


def select_upload_support(
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
    """Select all upload positions with a negative first-order objective change."""
    del answer_phase, label_token_ids
    vocab_size = int(z_l.shape[-1])
    k = max(1, min(int(candidate_topk), vocab_size))
    candidate_ids = torch.topk(z_s.float(), k=k).indices

    alpha_value = float(alpha)
    edge_weight = 1.0 - alpha_value
    zero_upload = alpha_value * z_l.float()
    full_candidate_upload = zero_upload.clone()
    full_candidate_upload[candidate_ids] += edge_weight * z_s.float()[candidate_ids]
    reference_winner = int(torch.argmax(full_candidate_upload).item())

    winner_mask = (candidate_ids == reference_winner).float()
    utility_gradient = edge_weight * z_s.float()[candidate_ids] * (winner_mask - 1.0 / k)
    privacy_gradient = float(privacy_weight) * torch.abs(
        z_s.float()[candidate_ids] - z_l.float()[candidate_ids]
    )
    total_gradient = -utility_gradient + privacy_gradient
    selected_ids = candidate_ids[total_gradient < 0]

    fused_logits = zero_upload.clone()
    if selected_ids.numel() > 0:
        fused_logits[selected_ids] += edge_weight * z_s.float()[selected_ids]
    fused_logits = apply_repetition_penalty(fused_logits, generated_ids, repetition_penalty)
    if eos_token_id is not None and not answer_complete and int(eos_token_id) < vocab_size:
        fused_logits[int(eos_token_id)] = torch.finfo(fused_logits.dtype).min

    return {
        "next_token_id": int(torch.argmax(fused_logits).item()),
        "observed_scores": z_s.detach(),
        "observed_ids": selected_ids.detach(),
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
    run_dataset(args, selector=select_upload_support, mode="cloud")


if __name__ == "__main__":
    main()
