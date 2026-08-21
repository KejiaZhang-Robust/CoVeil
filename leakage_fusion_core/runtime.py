"""Minimal shared runtime for the two released CoVeil fusion paths."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm

from .model_loader import load_models
from .prompts import build_large_prompt, build_small_prompt

Selector = Callable[..., Dict[str, Any]]


def load_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def option_labels(options: Any) -> List[str]:
    if isinstance(options, dict):
        labels = sorted(str(key).strip().upper() for key in options)
    elif isinstance(options, list):
        labels = [chr(ord("A") + index) for index in range(len(options))]
    else:
        labels = []
    return labels or ["A", "B", "C", "D"]


def answer_pattern(labels: Sequence[str]) -> re.Pattern[str]:
    choices = "|".join(re.escape(label) for label in labels)
    return re.compile(rf"Answer\s*:\s*(?:<\s*)?({choices})(?:\s*>|\b)", re.IGNORECASE)


def answer_prefix_pattern() -> re.Pattern[str]:
    return re.compile(r"Answer\s*:\s*(?:<\s*)?$", re.IGNORECASE)


def extract_answer(text: str, labels: Sequence[str]) -> tuple[str, str]:
    match = answer_pattern(labels).search(text)
    answer = match.group(1).upper() if match else "INVALID"
    reason_match = re.search(
        r"(?:Brief\s+Reasoning|Reason)\s*:\s*(.+?)(?=Answer\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return answer, reason


def resolve_label_token_ids(tokenizer, labels: Sequence[str]) -> List[int]:
    ids = set()
    for label in labels:
        for form in (label, f" {label}", f"<{label}>"):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[-1]))
    return sorted(ids)


def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return "<DECODING_ERROR>"


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or generated_ids.numel() == 0:
        return logits
    adjusted = logits.clone().float()
    for token_id in torch.unique(generated_ids).tolist():
        if int(token_id) >= adjusted.shape[-1]:
            continue
        if adjusted[int(token_id)] > 0:
            adjusted[int(token_id)] /= penalty
        else:
            adjusted[int(token_id)] *= penalty
    return adjusted


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _top_trace(scores: torch.Tensor, tokenizer, trace_topk: int, allowed_ids=None) -> Dict[str, Any]:
    if allowed_ids is None:
        k = min(max(1, trace_topk), int(scores.numel()))
        values, ids = torch.topk(scores.float(), k=k)
    else:
        ids = allowed_ids.long()
        if ids.numel() == 0:
            return {"ids": [], "tokens": [], "scores": []}
        order = torch.argsort(scores[ids].float(), descending=True)
        ids = ids[order][:trace_topk]
        values = scores[ids].float()
    id_list = [int(value) for value in ids.detach().cpu().tolist()]
    return {
        "ids": id_list,
        "tokens": [safe_decode(tokenizer, token_id) for token_id in id_list],
        "scores": [float(value) for value in values.detach().cpu().tolist()],
    }


def _generate_sample(
    sample: Dict[str, Any],
    large_model,
    small_model,
    tokenizer,
    selector: Selector,
    mode: str,
    alpha: float,
    privacy_weight: float,
    candidate_topk: int,
    max_new_tokens: int,
    record_steps: int,
    trace_topk: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    labels = option_labels(sample.get("options"))
    brief_reasoning = str(sample.get("_id", "")).lower().startswith("cosmos_")
    large_prompt = build_large_prompt(
        str(sample.get("public_query", "")),
        sample.get("options"),
        brief_reasoning=brief_reasoning,
    )
    small_prompt = build_small_prompt(
        str(sample.get("private_context", "")),
        str(sample.get("public_query", "")),
        sample.get("options"),
        brief_reasoning=brief_reasoning,
    )

    large_device = _model_input_device(large_model)
    small_device = _model_input_device(small_model)
    large_input_ids = tokenizer.encode(large_prompt, return_tensors="pt").to(large_device)
    small_input_ids = tokenizer.encode(small_prompt, return_tensors="pt").to(small_device)
    generated_ids = large_input_ids.clone()
    completed_pattern = answer_pattern(labels)
    prefix_pattern = answer_prefix_pattern()
    label_token_ids = resolve_label_token_ids(tokenizer, labels)
    trace = {
        "fusion_side": mode,
        "record_steps": int(record_steps),
        "topk": int(trace_topk),
        "steps": [],
    }

    for step in range(max_new_tokens):
        with torch.no_grad():
            large_logits = large_model(generated_ids).logits[0, -1, : len(tokenizer)]
        new_tokens = generated_ids[:, large_input_ids.shape[1] :].to(small_device)
        small_full_ids = torch.cat([small_input_ids, new_tokens], dim=1)
        with torch.no_grad():
            small_logits = small_model(small_full_ids).logits[0, -1, : len(tokenizer)]
        small_logits = small_logits.to(large_logits.device)

        current_text = tokenizer.decode(
            generated_ids[0, large_input_ids.shape[1] :], skip_special_tokens=True
        )
        selection = selector(
            z_l=large_logits,
            z_s=small_logits,
            alpha=alpha,
            privacy_weight=privacy_weight,
            candidate_topk=candidate_topk,
            generated_ids=generated_ids,
            repetition_penalty=repetition_penalty,
            answer_complete=bool(completed_pattern.search(current_text)),
            answer_phase=bool(prefix_pattern.search(current_text)),
            label_token_ids=label_token_ids,
            eos_token_id=tokenizer.eos_token_id,
        )

        if step < record_steps:
            observed_scores = selection["observed_scores"]
            observed_ids = selection.get("observed_ids")
            step_trace = _top_trace(observed_scores, tokenizer, trace_topk, observed_ids)
            step_trace["step"] = step
            trace["steps"].append(step_trace)

        next_token_id = int(selection["next_token_id"])
        if next_token_id < 0 or next_token_id >= len(tokenizer):
            break
        next_token = torch.tensor([[next_token_id]], device=generated_ids.device, dtype=torch.long)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        generated_text = tokenizer.decode(
            generated_ids[0, large_input_ids.shape[1] :], skip_special_tokens=True
        )
        if completed_pattern.search(generated_text):
            break
        if tokenizer.eos_token_id is not None and next_token_id == int(tokenizer.eos_token_id):
            break

    generated_text = tokenizer.decode(
        generated_ids[0, large_input_ids.shape[1] :], skip_special_tokens=True
    )
    answer, reason = extract_answer(generated_text, labels)
    gold = str(sample.get("answer_idx", "")).strip().upper()
    result = dict(sample)
    result.update(
        {
            "fusion_side": mode,
            "fusion_answer": answer,
            "fusion_reason": reason,
            "fusion_full_response": generated_text,
            "is_correct": answer == gold,
            "alpha": float(alpha),
            "privacy_weight": float(privacy_weight),
            "candidate_topk": int(candidate_topk),
            "leakage_trace": trace,
        }
    )
    return result


def _sample_id(row: Dict[str, Any]) -> str:
    return str(row.get("_id", row.get("id", row.get("sample_id", ""))))


def _existing_ids(output_file: str) -> set[str]:
    if not os.path.exists(output_file):
        return set()
    return {_sample_id(row) for row in load_jsonl(output_file)}


def _accuracy(rows: Iterable[Dict[str, Any]]) -> float:
    values = [1.0 if row.get("is_correct") else 0.0 for row in rows]
    return sum(values) / len(values) if values else 0.0


def run_dataset(args, selector: Selector, mode: str) -> None:
    """Run one released fusion path and write resumable JSONL results."""
    dataset = load_jsonl(args.input_file, args.max_samples)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = _existing_ids(args.output_file) if args.resume else set()
    pending = [row for row in dataset if _sample_id(row) not in processed]

    large_model, small_model, tokenizer = load_models(
        args.large_model, args.small_model, args.device
    )
    file_mode = "a" if processed else "w"
    with open(output_path, file_mode, encoding="utf-8") as handle:
        for sample in tqdm(pending, desc=f"{mode}-side fusion", unit="sample"):
            result = _generate_sample(
                sample=sample,
                large_model=large_model,
                small_model=small_model,
                tokenizer=tokenizer,
                selector=selector,
                mode=mode,
                alpha=args.alpha,
                privacy_weight=args.privacy_weight,
                candidate_topk=args.candidate_topk,
                max_new_tokens=args.max_new_tokens,
                record_steps=args.record_steps,
                trace_topk=args.trace_topk,
                repetition_penalty=args.repetition_penalty,
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    results = load_jsonl(args.output_file)
    summary = {
        "fusion_side": mode,
        "num_samples": len(results),
        "accuracy": _accuracy(results),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
