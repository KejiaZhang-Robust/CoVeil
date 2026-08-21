"""Dataset-independent prompts for multiple-choice collaborative decoding."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _normalise_options(options: Any) -> Dict[str, str]:
    if isinstance(options, dict):
        normalised = {str(key).strip().upper(): str(value) for key, value in options.items()}
        return dict(sorted(normalised.items()))
    if isinstance(options, list):
        return {chr(ord("A") + index): str(value) for index, value in enumerate(options)}
    raise ValueError("Each sample must provide options as an object or list.")


def _format_options(options: Any) -> Tuple[str, List[str]]:
    normalised = _normalise_options(options)
    labels = list(normalised)
    block = "\n".join(f"({label}) {normalised[label]}" for label in labels)
    return block, labels


def build_large_prompt(public_query: str, options: Any, brief_reasoning: bool = False) -> str:
    options_block, labels = _format_options(options)
    reasoning_instruction = (
        "Brief Reasoning: <one short sentence>"
        if brief_reasoning
        else "Reason: <1-2 sentences>"
    )
    return (
        f"Question: {public_query}\n"
        f"Options:\n{options_block}\n\n"
        "Return in the format:\n"
        f"{reasoning_instruction}\n"
        f"Answer: <{'|'.join(labels)}>\n"
    )


def build_small_prompt(
    private_context: str,
    public_query: str,
    options: Any,
    brief_reasoning: bool = False,
) -> str:
    prefix = f"{private_context}\n" if private_context else ""
    return prefix + build_large_prompt(public_query, options, brief_reasoning=brief_reasoning)


def render_chat_prompt(tokenizer, prompt: str, system_prompt: str | None = None) -> str:
    """Use a model's chat template when available, otherwise keep plain text."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered if isinstance(rendered, str) and rendered else prompt
    except Exception:
        return prompt
