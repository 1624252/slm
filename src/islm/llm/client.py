"""LLM clients.

`OpenAIClient` talks to any OpenAI-compatible endpoint (set base_url for other providers or a
local server). `MockLLM` is a deterministic offline stand-in that builds spec-compliant stories
and canned judgements from the prompt, so the whole pipeline and eval run without a network or
API key.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from ..config import LLMConfig


class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str: ...


class OpenAIClient:
    """Thin wrapper over the OpenAI-compatible chat completions API."""

    def __init__(self, model: str, config: LLMConfig | None = None):
        from openai import OpenAI  # lazy: keep openai optional

        cfg = config or LLMConfig.from_env()
        if not cfg.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(never commit it), or use MockLLM for offline runs."
            )
        self._client = OpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout
        )
        self.model = model

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# --- Offline mock ---------------------------------------------------------------------------


def _field_list(text: str, label: str) -> list[str]:
    m = re.search(rf"^{label}:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return []
    return [w.strip() for w in re.split(r"[,\n]", m.group(1)) if w.strip()]


def _first_known(candidates: list[str], known: set[str]) -> str:
    for c in candidates:
        if c in known:
            return c
    return ""


def _sentence(words: list[str]) -> str:
    words = [w for w in words if w]
    if not words:
        return ""
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


class MockLLM:
    """Deterministic stand-in for a teacher/judge model (offline tests and smoke runs)."""

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        task = ""
        m = re.search(r"TASK:\s*(\w+)", system)
        if m:
            task = m.group(1).upper()
        if task == "JUDGE":
            return self._judge()
        if task == "CLOZE":
            return self._cloze(user)
        return self._story(system + "\n" + user)

    def _story(self, text: str) -> str:
        targets = [t.lower() for t in _field_list(text, "TARGET_WORDS")]
        known = {w.lower() for w in _field_list(text, "KNOWN_WORDS")}
        m = re.search(r"at least (\d+)", text)
        min_recurrence = int(m.group(1)) if m else 3
        lm = re.search(r"^Language:\s*(\w+)", text, flags=re.MULTILINE)
        language = lm.group(1).lower() if lm else "en"
        if language == "en":
            return self._english_story(known, targets, min_recurrence)
        return self._generic_story(known, targets, min_recurrence, language)

    def _english_story(self, known: set[str], targets: list[str], min_recurrence: int) -> str:
        art = "the" if "the" in known else ("a" if "a" in known else "")
        subj = _first_known(["cat", "dog", "bird", "mouse", "friend", "man", "woman"], known)
        subj = subj or ("it" if "it" in known else "")
        verb = _first_known(["see", "find", "look", "play", "run", "like", "want"], known)
        # "is" lemmatizes to "be", so it is valid whenever the known set contains either.
        is_ = "is" if ("is" in known or "be" in known) else ""
        adj = _first_known(["big", "small", "happy", "funny", "new", "good", "kind"], known)

        sentences = [_sentence([art, subj, is_, adj])]  # known-only opener
        for t in targets:  # introduce each target as the single new word
            sentences.append(_sentence([art, subj, verb, t]))
        for t in targets:  # recurrence: repeat with known-only frames
            for _ in range(min_recurrence):
                sentences.append(_sentence([art, t, is_, adj]))
        return "\n".join(s for s in sentences if s)

    def _generic_story(
        self, known: set[str], targets: list[str], min_recurrence: int, language: str
    ) -> str:
        # Language-agnostic filler: two known words per sentence, separated so a segmenter
        # (jieba / MeCab) won't merge them into an unintended word. Not good prose — it just
        # exercises the validators offline.
        cjk = language in ("zh", "ja")
        sep, end = ("、", "。") if cjk else (" ", ".")
        fillers = sorted(known)[:2] or ["x"]

        def sent(parts: list[str]) -> str:
            parts = [p for p in parts if p]
            return sep.join(parts) + end if parts else ""

        sentences = [sent(fillers)]
        for t in targets:
            sentences.append(sent([fillers[0], t]))
        for t in targets:
            for _ in range(min_recurrence):
                sentences.append(sent([t, fillers[-1]]))
        return "\n".join(s for s in sentences if s)

    def _judge(self) -> str:
        # Full marks on whatever the current rubric dimensions are (kept in sync automatically).
        from .prompts import JUDGE_DIMENSIONS

        scores = {dim: 2 for dim in JUDGE_DIMENSIONS}
        scores["rationale"] = "mock judge: full marks"
        return json.dumps(scores)

    def _cloze(self, user: str) -> str:
        # Cannot truly infer offline; return a placeholder so the metric is reported, not gating.
        return "something"


def get_client(model: str | None = None, *, mock: bool = False) -> LLMClient:
    """Return a MockLLM (offline) or an OpenAIClient for the given model."""
    if mock:
        return MockLLM()
    cfg = LLMConfig.from_env()
    return OpenAIClient(model or cfg.teacher_model, cfg)
