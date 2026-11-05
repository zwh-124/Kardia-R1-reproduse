"""Rubric-as-judge LLM wrapper.

The rubric reward (Eq. 12) requires an external judge LLM ``LLM_rubric``
(Qwen3-8B in the paper) that scores a candidate response along five criteria.
This module provides a small, swappable interface so the same reward logic can
be backed by:

* a local ``transformers`` model,
* any OpenAI-compatible endpoint (vLLM, SGLang, ...), or
* a deterministic :class:`MockJudge` for unit tests and offline demos.
"""

from __future__ import annotations

import abc
from typing import Dict, Optional

from .data import UserProfile
from .prompts import build_rubric_prompt, parse_rubric_output


class RubricJudge(abc.ABC):
    """Abstract rubric-as-judge that maps a (profile, context, response) triple
    to five dimension scores."""

    dimensions = ("relevance", "fluency", "empathy", "persona_consistency", "safety")

    def prompt(self, profile: UserProfile, context: str, response: str) -> str:
        return build_rubric_prompt(profile, context, response)

    @abc.abstractmethod
    def score(
        self, profile: UserProfile, context: str, response: str
    ) -> Dict[str, float]:
        """Return ``{dimension: score}`` for the five rubric dimensions."""

    def __call__(self, profile, context, response):
        return self.score(profile, context, response)


class MockJudge(RubricJudge):
    """Deterministic, model-free judge for tests and offline demos.

    It applies simple, interpretable heuristics so the pipeline is runnable
    without downloading a model: length and keyword-based signals for each of
    the five dimensions.  **This is only a placeholder**; production training
    must use a real judge LLM (Qwen3-8B) as in the paper.
    """

    _EMPATHY_WORDS = (
        "understand", "feel", "sound", "sorry", "here for you", "that must",
        "i can only imagine", "valid", "okay to", "ok to",
    )
    _UNSAFE_WORDS = ("kill", "hurt yourself", "suicide", "self-harm")

    def score(self, profile, context, response):
        r = (response or "").strip()
        c = (context or "").strip()
        low = r.lower()

        relevance = 3.0
        if len(r) >= 8:
            relevance += 1.0
        ctx_words = set(c.lower().split())
        overlap = sum(1 for w in ctx_words if w in low)
        if overlap >= 2:
            relevance += 1.0

        fluency = 3.0
        if r and r[-1] in ".!?":
            fluency += 1.0
        if len(r.split()) >= 5:
            fluency += 1.0

        empathy = 2.0
        for w in self._EMPATHY_WORDS:
            if w in low:
                empathy += 1.0
                break
        if len(r) >= 12:
            empathy += 1.0

        persona = 2.0
        if any(w in low for w in ("you", "your")):
            persona += 2.0
        if profile and profile.user_id.lower() in low:
            persona += 1.0

        safety = 5.0
        for w in self._UNSAFE_WORDS:
            if w in low:
                safety -= 3.0
                break

        def clamp(x):
            return max(1.0, min(5.0, float(x)))

        return {
            "relevance": clamp(relevance),
            "fluency": clamp(fluency),
            "empathy": clamp(empathy),
            "persona_consistency": clamp(persona),
            "safety": clamp(safety),
        }


class TransformersJudge(RubricJudge):
    """Rubric judge backed by a local ``transformers`` model (e.g. Qwen3-8B)."""

    def __init__(self, model_name_or_path: str, **gen_kwargs):
        self.model_name_or_path = model_name_or_path
        self.gen_kwargs = gen_kwargs
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype="auto",
                device_map="auto",
            )

    def score(self, profile, context, response):
        self._load()
        prompt = self.prompt(profile, context, response)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, **self.gen_kwargs)
        decoded = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_rubric_output(decoded)


class OpenAICompatibleJudge(RubricJudge):
    """Rubric judge backed by any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        **gen_kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.gen_kwargs = gen_kwargs

    def score(self, profile, context, response):
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        prompt = self.prompt(profile, context, response)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            **self.gen_kwargs,
        )
        return parse_rubric_output(resp.choices[0].message.content)


def make_judge(
    backend: str = "mock", model_name_or_path: Optional[str] = None, **kwargs
) -> RubricJudge:
    """Factory to build a judge from a backend name.

    ``backend`` is one of ``"mock"``, ``"transformers"``, or ``"openai"``.
    """
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockJudge()
    if backend == "transformers":
        return TransformersJudge(model_name_or_path or "Qwen/Qwen3-8B", **kwargs)
    if backend == "openai":
        return OpenAICompatibleJudge(model_name_or_path or "Qwen3-8B", **kwargs)
    raise ValueError(f"Unknown judge backend: {backend}")
