"""
Multi-provider LLM generation for PACE experiments.

Provides a unified ``generate()`` that routes to OpenAI, Anthropic, or
HuggingFace/Novita depending on the model registry entry.  All functions
return a standardized response dict compatible with LiC's model_openai
contract.

Usage::

    from pace.providers import generate, generate_json

    # Works exactly like LiC's model_openai.generate, but routes to the
    # correct provider based on the model name.
    resp = generate(messages, model="claude-4.6-sonnet", return_metadata=True)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from pace.models import ModelConfig, get_model, MODEL_REGISTRY_ALL


# ──────────────────────────────────────────────
#  Client pool (lazy-initialized singletons)
# ──────────────────────────────────────────────

_clients: dict[str, Any] = {}


def _get_openai_client(api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None):
    cache_key = f"openai:{api_key_env}:{base_url or 'default'}"
    if cache_key not in _clients:
        from openai import OpenAI
        kwargs: dict[str, Any] = {}
        key = os.environ.get(api_key_env, "")
        if not key:
            raise EnvironmentError(f"Missing env var {api_key_env}")
        kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        _clients[cache_key] = OpenAI(**kwargs)
    return _clients[cache_key]


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("Missing env var ANTHROPIC_API_KEY")
        _clients["anthropic"] = anthropic.Anthropic(api_key=key)
    return _clients["anthropic"]


def _client_for_model(cfg: ModelConfig):
    """Return the right API client for a given model config."""
    if cfg.provider == "openai":
        return _get_openai_client("OPENAI_API_KEY")
    elif cfg.provider == "hf_novita":
        return _get_openai_client("HF_TOKEN", base_url="https://router.huggingface.co/v1")
    elif cfg.provider == "anthropic":
        return _get_anthropic_client()
    elif cfg.provider == "google":
        raise NotImplementedError(
            "Google provider not yet implemented. Add GOOGLE_API_KEY and google-genai package."
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")


# ──────────────────────────────────────────────
#  Message formatting helpers
# ──────────────────────────────────────────────

def _format_messages(messages: list[dict], variables: dict | None = None) -> list[dict]:
    """Replace [[VAR]] placeholders in the last user message."""
    if not variables:
        return messages
    messages = [dict(m) for m in messages]
    last_user = [m for m in messages if m["role"] == "user"]
    if not last_user:
        return messages
    msg = last_user[-1]
    for k, v in variables.items():
        msg["content"] = msg["content"].replace(f"[[{k}]]", str(v))
    return messages


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split out a leading system message (needed for Anthropic)."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def _handle_reasoning_model(messages: list[dict], model_id: str) -> list[dict]:
    """Reasoning models (o1/o3/R1) don't support system role — merge into first user msg."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = dict(rest[0])
        rest[0]["content"] = f"System Message: {sys_content}\n{rest[0]['content']}"
    else:
        rest.insert(0, {"role": "user", "content": sys_content})
    return rest


# ──────────────────────────────────────────────
#  Cost estimation (best-effort)
# ──────────────────────────────────────────────

_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.00015, 0.0006),
    "gpt-4o":       (0.0025, 0.01),
    "gpt-4.1":      (0.002, 0.008),
    "o3":           (0.01, 0.04),
    "gpt-5.4":      (0.003, 0.012),
    "gpt-5.4-mini": (0.0002, 0.0008),
    "gpt-5.4-pro":  (0.01, 0.04),
    "gpt-5.4-nano": (0.0001, 0.0004),
}


def _estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD cost estimate.  Returns 0.0 for unknown pricing."""
    if model_name in _COST_PER_1K:
        inp, out = _COST_PER_1K[model_name]
        return (prompt_tokens / 1000) * inp + (completion_tokens / 1000) * out
    return 0.0


# ──────────────────────────────────────────────
#  Extract logprobs from OpenAI-style responses
# ──────────────────────────────────────────────

def _extract_logprobs(response_dict: dict) -> list[list[float]]:
    """
    Pull per-token top-k logprobs from an OpenAI-compatible response dict.
    Returns list of lists: [[logp1, logp2, ...], ...] per generated token.
    """
    result = []
    try:
        logprobs_data = response_dict["choices"][0].get("logprobs")
        if not logprobs_data:
            return result
        for token_info in logprobs_data.get("content", []):
            top = token_info.get("top_logprobs", [])
            if top:
                result.append([entry["logprob"] for entry in top])
    except (KeyError, IndexError, TypeError):
        pass
    return result


# ──────────────────────────────────────────────
#  OpenAI-compatible generation (OpenAI + HF/Novita)
# ──────────────────────────────────────────────

def _generate_openai_compat(
    client,
    messages: list[dict],
    cfg: ModelConfig,
    *,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    is_json: bool = False,
    logprobs: bool = True,
    top_logprobs: int = 5,
    timeout: int = 120,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Generate via any OpenAI-compatible API."""

    if cfg.is_reasoning:
        messages = _handle_reasoning_model(messages, cfg.api_id)

    kwargs: dict[str, Any] = {}
    if is_json:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    elif cfg.max_output_tokens:
        kwargs["max_completion_tokens"] = cfg.max_output_tokens

    request_logprobs = logprobs and cfg.supports_logprobs
    if request_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs

    if not cfg.is_reasoning:
        kwargs["temperature"] = temperature

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=cfg.api_id,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Failed after {max_retries} retries: {last_err}") from last_err

    resp = response.to_dict() if hasattr(response, "to_dict") else response.model_dump()
    usage = resp.get("usage", {})
    text = resp["choices"][0]["message"]["content"] or ""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    token_top_logprobs = _extract_logprobs(resp) if request_logprobs else []

    return {
        "message": text,
        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_usd": _estimate_cost(cfg.name, prompt_tokens, completion_tokens),
        "token_top_logprobs": token_top_logprobs,
        "model": cfg.name,
        "provider": cfg.provider,
    }


# ──────────────────────────────────────────────
#  Anthropic Messages API generation
# ──────────────────────────────────────────────

def _generate_anthropic(
    client,
    messages: list[dict],
    cfg: ModelConfig,
    *,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    is_json: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Generate via Anthropic Messages API."""

    system_text, conv_messages = _split_system(messages)

    kwargs: dict[str, Any] = {
        "model": cfg.api_id,
        "messages": conv_messages,
        "max_tokens": max_tokens or cfg.max_output_tokens or 4096,
        "temperature": temperature,
    }
    if system_text:
        kwargs["system"] = system_text

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Anthropic failed after {max_retries} retries: {last_err}") from last_err

    text = response.content[0].text if response.content else ""
    prompt_tokens = getattr(response.usage, "input_tokens", 0)
    completion_tokens = getattr(response.usage, "output_tokens", 0)

    return {
        "message": text,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_usd": 0.0,
        "token_top_logprobs": [],  # Anthropic doesn't support logprobs
        "model": cfg.name,
        "provider": cfg.provider,
    }


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def generate(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    *,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    is_json: bool = False,
    return_metadata: bool = False,
    logprobs: bool = True,
    top_logprobs: int = 5,
    timeout: int = 120,
    max_retries: int = 3,
    variables: dict | None = None,
) -> dict[str, Any] | str:
    """
    Unified generation across all providers.

    Drop-in compatible with LiC's ``model_openai.generate``.
    Logprobs are requested by default and returned in ``token_top_logprobs``
    (empty list if the provider/model doesn't support them).
    """
    messages = _format_messages(messages, variables)

    cfg = get_model(model)
    client = _client_for_model(cfg)

    if cfg.provider == "anthropic":
        result = _generate_anthropic(
            client, messages, cfg,
            temperature=temperature, max_tokens=max_tokens,
            is_json=is_json, timeout=timeout, max_retries=max_retries,
        )
    else:
        result = _generate_openai_compat(
            client, messages, cfg,
            temperature=temperature, max_tokens=max_tokens,
            is_json=is_json, logprobs=logprobs, top_logprobs=top_logprobs,
            timeout=timeout, max_retries=max_retries,
        )

    if not return_metadata:
        return result["message"]
    return result


def generate_json(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    **kwargs,
) -> dict[str, Any]:
    """Generate and parse JSON response. Compatible with LiC's generate_json."""
    kwargs["is_json"] = True
    kwargs["return_metadata"] = True
    result = generate(messages, model, **kwargs)
    if isinstance(result, str):
        result = {"message": result, "total_usd": 0.0}
    try:
        result["message"] = json.loads(result["message"])
    except (json.JSONDecodeError, TypeError):
        pass
    return result
