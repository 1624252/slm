"""Story generators: the "model under test".

A generator turns a scenario into a story. Swap in an OpenAI-compatible API model, a local
fine-tuned Hugging Face checkpoint (optionally with a LoRA adapter), or the offline mock — the
eval harness doesn't change. This is what makes "once we have a trained model, just run the eval"
a one-liner.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..datagen.generate import generate_story, rewrite_story
from ..datagen.scenarios import Scenario
from ..llm.client import LLMClient, get_client
from ..llm.prompts import generation_prompt
from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer

StoryGenerator = Callable[[Scenario], str]


def api_generator(
    model: str | None = None,
    *,
    client: LLMClient | None = None,
    temperature: float = 0.0,
    mock: bool = False,
) -> StoryGenerator:
    """Generator backed by an OpenAI-compatible client (or the mock). Temperature 0 for repro."""
    client = client or get_client(model, mock=mock)
    return lambda scenario: generate_story(scenario, client, temperature=temperature)


class HFGenerator:
    """Local Hugging Face causal LM, optionally with a PEFT/LoRA adapter — for evaluating a
    fine-tuned checkpoint. Heavy deps (torch/transformers/peft) are imported lazily.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | None = None,
        chat_kwargs: dict | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_kwargs: dict = {}
        if device_map:  # only when accelerate/GPU is available; CPU loads fine without it
            load_kwargs = {"device_map": device_map, "torch_dtype": "auto"}
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # e.g. {"enable_thinking": False} for Qwen3 to suppress <think> blocks.
        self.chat_kwargs = chat_kwargs or {}

    def __call__(self, scenario: Scenario) -> str:
        system, user = generation_prompt(scenario)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_kwargs,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature or None,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def client_rewriter(client: LLMClient) -> Callable[[Scenario, str, list[str]], str]:
    """A rewrite function for the inference-time guard, backed by an LLM client."""
    return lambda scenario, story, failures: rewrite_story(scenario, story, failures, client)


def guarded(
    generate: StoryGenerator,
    rewrite: Callable[[Scenario, str, list[str]], str],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    max_retries: int = 3,
) -> StoryGenerator:
    """Wrap a generator with the inference-time validate-and-rewrite guard (PRD 9/10).

    Returns the first spec-passing story, or the last attempt if none pass. Evaluate the raw model
    to measure the model itself; evaluate the guarded generator to measure the deployed system.
    """
    analyzers: dict[str, object] = {}

    def run(scenario: Scenario) -> str:
        lem = analyzers.setdefault(scenario.language, get_analyzer(scenario.language))
        known, target = scenario.known_set(), scenario.target_set()
        story = generate(scenario)
        report = validate_story(story, known, target, lem, thresholds)
        tries = 0
        while not report.hard_pass and tries < max_retries:
            story = rewrite(scenario, story, report.failures())
            report = validate_story(story, known, target, lem, thresholds)
            tries += 1
        return story

    return run
