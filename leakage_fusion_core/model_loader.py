"""Model loading with the shared-vocabulary invariant used by fusion."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _model_vocab_size(model: Any) -> int:
    configured = getattr(getattr(model, "config", None), "vocab_size", None)
    if configured is not None:
        return int(configured)
    return int(model.get_output_embeddings().weight.shape[0])


def _load_model(model_name: str, device: str):
    kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def load_models(large_model_name: str, small_model_name: str, device: str = "auto"):
    """Load a model pair and require an identical token-id mapping."""
    large_tokenizer = load_tokenizer(large_model_name)
    small_tokenizer = load_tokenizer(small_model_name)
    if large_tokenizer.get_vocab() != small_tokenizer.get_vocab():
        raise ValueError(
            "Collaborative decoding requires identical tokenizer vocabularies. "
            "Configure a large/small model pair with the same token-id mapping."
        )

    large_model = _load_model(large_model_name, device)
    small_model = _load_model(small_model_name, device)
    vocab_size = len(large_tokenizer)
    if _model_vocab_size(large_model) < vocab_size or _model_vocab_size(small_model) < vocab_size:
        raise ValueError("A model output vocabulary is smaller than the shared tokenizer vocabulary.")

    if not torch.cuda.is_available() and device == "auto":
        print("Warning: CUDA is unavailable; large-model inference may be slow.")
    return large_model, small_model, large_tokenizer
