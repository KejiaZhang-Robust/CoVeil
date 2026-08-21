"""Shared utilities for the compact CoVeil release."""

from .model_loader import load_models, load_tokenizer
from .prompts import build_large_prompt, build_small_prompt, render_chat_prompt
from .runtime import (
    apply_repetition_penalty,
    answer_pattern,
    answer_prefix_pattern,
    extract_answer,
    load_jsonl,
    option_labels,
    resolve_label_token_ids,
    run_dataset,
    safe_decode,
)

__all__ = [
    "apply_repetition_penalty",
    "answer_pattern",
    "answer_prefix_pattern",
    "build_large_prompt",
    "build_small_prompt",
    "extract_answer",
    "load_jsonl",
    "load_models",
    "load_tokenizer",
    "option_labels",
    "render_chat_prompt",
    "resolve_label_token_ids",
    "run_dataset",
    "safe_decode",
]
