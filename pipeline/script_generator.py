"""
pipeline/script_generator.py
==============================
Generates lecture script – OpenRouter API (fast, works on low RAM).
Local model is attempted only if loaded.
"""

import re
import time
import logging
from typing import Optional

import torch

logger = logging.getLogger("edugenie.script")

SYSTEM_PROMPT = (
    "You are an expert university lecturer. "
    "Produce well-structured, accurate, and detailed educational explanations "
    "with equations, examples, and code where appropriate."
)


def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _generate_via_openrouter(
    topic: str,
    api_key: str,
    model_name: str,
    max_tokens: int = 900,
) -> Optional[str]:
    import httpx

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Write a detailed university-level lecture script about: {topic}\n\n"
                "Structure it with clear headings (##), bullet points, equations "
                "where relevant, and practical examples. Aim for about 800 words."
            ),
        },
    ]

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://eduvara.app",
                "X-Title": "EduGenie",
            },
            json={"model": model_name, "messages": messages, "max_tokens": max_tokens},
            timeout=90,
        )
        if resp.is_success:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("Script generated (OpenRouter '%s'): %d words", model_name, len(text.split()))
            return text
        else:
            logger.error("OpenRouter error %d: %s", resp.status_code, resp.text[:300])
            return None
    except Exception as exc:
        logger.error("OpenRouter request failed: %s", exc)
        return None


def _generate_template_script(topic: str) -> str:
    logger.warning("Using template script fallback for topic '%s'", topic)
    return f"""## Introduction to {topic}

Welcome to this lesson on **{topic}**.
In this session we will break down the core concepts, explore real-world
applications, and build a solid understanding from the ground up.

## What is {topic}?

{topic} is a fundamental concept that plays a central role in its field.
It provides the theoretical and practical foundation for a wide range of
applications across engineering, science, and industry.

## Core Principles

Understanding {topic} starts with its foundational principles:

1. **Definition** — A precise statement of what {topic} means.
2. **Mechanism** — How {topic} operates step by step.
3. **Properties** — The key characteristics that make {topic} useful.

## Practical Applications

{topic} appears in many real-world scenarios:
- Engineering and technology
- Scientific research
- Industry and automation

## Summary

In this lesson we covered the definition, core principles, and practical
examples of {topic}.
"""


def _generate_local(
    topic: str,
    tokenizer,
    model,
    device,
    use_sampling: bool = True,
    max_new_tokens: int = 900,
) -> str:
    """Local inference – only used if model is loaded (rare on low RAM)."""
    prompt = build_prompt(topic)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.15,
            do_sample=use_sampling,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    generated = re.sub(r"<\|im_(start|end)\|>.*", "", generated, flags=re.DOTALL).strip()
    generated = re.sub(r"\n{3,}", "\n\n", generated).strip()
    return generated


def generate_script(
    topic: str,
    tokenizer=None,
    model=None,
    device=None,
    use_sampling: bool = True,
    max_new_tokens: int = 900,
) -> str:
    """
    Generate script using OpenRouter API (fast, works on low RAM).
    Falls back to local model only if both tokenizer and model are not None.
    """
    # Try local model if fully loaded
    if tokenizer is not None and model is not None:
        try:
            return _generate_local(topic, tokenizer, model, device, use_sampling, max_new_tokens)
        except Exception as exc:
            logger.error("Local model failed: %s", exc)

    # Use OpenRouter API
    try:
        from config import settings
        api_key = getattr(settings, "OPENROUTER_API_KEY", "")
        if api_key:
            model_name = getattr(settings, "OPENROUTER_SCRIPT_MODEL", "microsoft/phi-3-mini-128k-instruct:free")
            result = _generate_via_openrouter(topic, api_key, model_name, max_new_tokens)
            if result:
                return result
    except Exception as exc:
        logger.warning("OpenRouter error: %s", exc)

    # Ultimate fallback
    return _generate_template_script(topic)