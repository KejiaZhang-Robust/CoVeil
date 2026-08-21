#!/usr/bin/env python3
"""Reconstruct private context from saved exposure traces and report ROUGE-1."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from leakage_fusion_core import load_jsonl, load_tokenizer, render_chat_prompt

WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def words(text: str) -> List[str]:
    return [match.group(0).lower() for match in WORD.finditer(text)]


def rouge1_scores(prediction: str, reference: str) -> Dict[str, float]:
    prediction_counts = Counter(words(prediction))
    reference_counts = Counter(words(reference))
    overlap = sum((prediction_counts & reference_counts).values())
    prediction_total = sum(prediction_counts.values())
    reference_total = sum(reference_counts.values())
    precision = overlap / prediction_total if prediction_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": recall, "f1": f1}


def trace_evidence(trace: Dict[str, Any], topk_per_step: int) -> tuple[List[str], List[str]]:
    step_lines: List[str] = []
    global_tokens: List[str] = []
    seen = set()
    for step in trace.get("steps", []):
        tokens = [str(token).strip() for token in step.get("tokens", [])[:topk_per_step]]
        tokens = [token for token in tokens if token]
        step_lines.append(f"Step {int(step.get('step', len(step_lines)))}: {' | '.join(tokens)}")
        for token in tokens:
            if token not in seen:
                seen.add(token)
                global_tokens.append(token)
    return step_lines, global_tokens


def build_prompt(row: Dict[str, Any], topk_per_step: int) -> str:
    trace = row.get("leakage_trace") or {}
    step_lines, global_tokens = trace_evidence(trace, topk_per_step)
    fusion_side = str(trace.get("fusion_side", row.get("fusion_side", "unknown")))
    return f"""You are auditing privacy leakage in {fusion_side}-side cloud-edge collaborative decoding.

Public question:
{row.get('public_query', '')}

Cloud-observed token evidence across decoding steps:
{chr(10).join(step_lines) if step_lines else '<none>'}

Unique observed tokens:
{' | '.join(global_tokens) if global_tokens else '<none>'}

Reconstruct the most likely hidden private context. Output only the reconstructed passage, without analysis or meta-commentary.

Private context:"""


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    rendered = render_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer.encode(rendered, return_tensors="pt").to(model_device(model))
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--attacker_model", required=True, help="Model path or Hugging Face identifier.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--topk_per_step", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl, args.max_samples)
    tokenizer = load_tokenizer(args.attacker_model)
    model_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}
    model_kwargs["device_map"] = "auto" if args.device == "auto" else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.attacker_model, **model_kwargs)
    model.eval()

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    recalls: List[float] = []
    f1_values: List[float] = []
    with open(output, "w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Private-context inversion", unit="sample"):
            reconstruction = generate(model, tokenizer, build_prompt(row, args.topk_per_step), args.max_new_tokens)
            scores = rouge1_scores(reconstruction, str(row.get("private_context", "")))
            recalls.append(scores["recall"])
            f1_values.append(scores["f1"])
            result = {
                "id": row.get("_id", row.get("id", row.get("sample_id"))),
                "reconstructed_context": reconstruction,
                "ROUGE-1 Recall": scores["recall"],
                "ROUGE-1 F1": scores["f1"],
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "num_samples": len(rows),
        "ROUGE-1 Recall": safe_mean(recalls),
        "ROUGE-1 F1": safe_mean(f1_values),
    }
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
