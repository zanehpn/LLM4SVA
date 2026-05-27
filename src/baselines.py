"""
Baseline SVA generation using GPT-4o-mini as proxy.

Paper baselines: CodeV-SVA-14B, GPT-5, DeepSeek-R1, AssertLLM.
All are unavailable locally. We use GPT-4o-mini as the accessible proxy.

SCALE-BLOCKED: CodeV-SVA-14B requires 14B-parameter model inference (GPU).
"""

import os
import json
from typing import Optional
from openai import OpenAI

# SCALE-BLOCKED components:
SCALE_BLOCKED = [
    "CodeV-SVA-14B (requires 28GB GPU memory, ~4× A100)",
    "GPT-5 (not publicly available)",
    "DeepSeek-R1 (requires large GPU cluster)",
    "AssertLLM (proprietary, requires license)",
    "TemporalSVA trained model (requires curriculum fine-tuning, RLVF, 4× A100)",
]

_SYSTEM_PROMPT = """You are an expert in SystemVerilog Assertions (SVA).
Generate syntactically correct, meaningful SVA assertions from natural language specifications.
Output ONLY the SVA code, no explanations. Use assert property syntax with @(posedge clk)."""

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it before running baseline experiments."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def generate_sva_gpt4o_mini(
    nl_spec: str,
    system_prompt: str = _SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> dict:
    """
    Generate SVA from NL spec using GPT-4o-mini.

    Returns:
        {model, nl_spec, generated_sva, usage}
    """
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate an SVA assertion for:\n{nl_spec}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generated = response.choices[0].message.content.strip()
    return {
        "model": "gpt-4o-mini",
        "nl_spec": nl_spec,
        "generated_sva": generated,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
