"""LLM layer: an OpenAI-compatible client, an offline MockLLM, and the prompt templates."""

from .client import LLMClient, MockLLM, OpenAIClient, get_client
from .prompts import cloze_prompt, generation_prompt, judge_prompt, rewrite_prompt

__all__ = [
    "LLMClient",
    "MockLLM",
    "OpenAIClient",
    "cloze_prompt",
    "generation_prompt",
    "get_client",
    "judge_prompt",
    "rewrite_prompt",
]
