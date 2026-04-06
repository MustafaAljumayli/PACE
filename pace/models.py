"""
Model registry for PACE multi-provider experiments.

Each model has a canonical short name (used in CLI, logs, tables) and full
provider-specific config (API model ID, provider, capabilities).

Providers:
  - openai:     Standard OpenAI API (OPENAI_API_KEY)
  - anthropic:  Anthropic Messages API (ANTHROPIC_API_KEY)
  - hf_novita:  HuggingFace Inference Router → Novita (HF_TOKEN)
  - google:     Google GenAI / Vertex (GOOGLE_API_KEY) — optional
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["openai", "anthropic", "hf_novita", "google"]


@dataclass(frozen=True)
class ModelConfig:
    name: str                     # Short canonical name for tables/logs
    api_id: str                   # Provider-specific model identifier
    provider: Provider
    supports_logprobs: bool = True
    is_reasoning: bool = False    # o-series, DeepSeek-R1, etc.
    is_paper_model: bool = False  # Was in the original LiC paper
    max_output_tokens: int = 4096
    env_key: str = ""             # Env var for API key (auto-detected if empty)
    cost_tier: str = "mid"        # "cheap", "mid", "expensive"
    notes: str = ""

    @property
    def env_var(self) -> str:
        if self.env_key:
            return self.env_key
        return {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "hf_novita": "HF_TOKEN",
            "google": "GOOGLE_API_KEY",
        }[self.provider]


# ──────────────────────────────────────────────
#  Paper models (for cross-study comparison)
# ──────────────────────────────────────────────

PAPER_MODELS: dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini", api_id="gpt-4o-mini",
        provider="openai", is_paper_model=True, cost_tier="cheap",
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o", api_id="gpt-4o",
        provider="openai", is_paper_model=True, cost_tier="mid",
    ),
    "gpt-4.1": ModelConfig(
        name="gpt-4.1", api_id="gpt-4.1",
        provider="openai", is_paper_model=True, cost_tier="mid",
    ),
    "o3": ModelConfig(
        name="o3", api_id="o3",
        provider="openai", is_paper_model=True, is_reasoning=True,
        supports_logprobs=False, max_output_tokens=16000, cost_tier="expensive",
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku", api_id="claude-3-haiku-20240307",
        provider="anthropic", is_paper_model=True,
        supports_logprobs=False, cost_tier="cheap",
    ),
    "claude-3.7-sonnet": ModelConfig(
        name="claude-3.7-sonnet", api_id="claude-sonnet-4-20250514",
        provider="anthropic", is_paper_model=True,
        supports_logprobs=False, cost_tier="mid",
        notes="Claude 3.7 Sonnet deprecated; using Claude Sonnet 4 (May 2025) as closest equivalent",
    ),
    "llama-4-scout": ModelConfig(
        name="llama-4-scout", api_id="meta-llama/Llama-4-Scout-17B-16E-Instruct:novita",
        provider="hf_novita", is_paper_model=True,
        supports_logprobs=False, cost_tier="cheap",
        notes="Novita provider does not return logprobs for Llama models",
    ),
    "deepseek-r1": ModelConfig(
        name="deepseek-r1", api_id="deepseek-ai/DeepSeek-R1:novita",
        provider="hf_novita", is_paper_model=True, is_reasoning=True,
        supports_logprobs=False, max_output_tokens=16000, cost_tier="mid",
    ),
}

# Optional paper models (need Google API key)
PAPER_MODELS_OPTIONAL: dict[str, ModelConfig] = {
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash", api_id="gemini-2.5-flash",
        provider="google", is_paper_model=True, cost_tier="cheap",
        notes="Requires GOOGLE_API_KEY",
    ),
    "gemini-2.5-pro": ModelConfig(
        name="gemini-2.5-pro", api_id="gemini-2.5-pro",
        provider="google", is_paper_model=True, cost_tier="expensive",
        notes="Requires GOOGLE_API_KEY",
    ),
}

# ──────────────────────────────────────────────
#  2026 models (our contribution)
# ──────────────────────────────────────────────

NEW_MODELS: dict[str, ModelConfig] = {
    # OpenAI GPT-5.4 family
    "gpt-5.4": ModelConfig(
        name="gpt-5.4", api_id="gpt-5.4",
        provider="openai", cost_tier="mid",
    ),
    "gpt-5.4-mini": ModelConfig(
        name="gpt-5.4-mini", api_id="gpt-5.4-mini",
        provider="openai", cost_tier="cheap",
    ),
    "gpt-5.4-pro": ModelConfig(
        name="gpt-5.4-pro", api_id="gpt-5.4-pro",
        provider="openai", cost_tier="expensive",
    ),
    "gpt-5.4-nano": ModelConfig(
        name="gpt-5.4-nano", api_id="gpt-5.4-nano",
        provider="openai", cost_tier="cheap",
    ),

    # Anthropic Claude 4.x family
    "claude-4.6-sonnet": ModelConfig(
        name="claude-4.6-sonnet", api_id="claude-sonnet-4-6",
        provider="anthropic", supports_logprobs=False, cost_tier="mid",
    ),
    "claude-opus-4.6": ModelConfig(
        name="claude-opus-4.6", api_id="claude-opus-4-6",
        provider="anthropic", supports_logprobs=False, cost_tier="expensive",
    ),
    "claude-haiku-4.5": ModelConfig(
        name="claude-haiku-4.5", api_id="claude-haiku-4-5-20251001",
        provider="anthropic", supports_logprobs=False, cost_tier="cheap",
    ),

    # Qwen 3.5 MoE family (via HF → Novita)
    "qwen-3.5-122b": ModelConfig(
        name="qwen-3.5-122b", api_id="Qwen/Qwen3.5-122B-A10B:novita",
        provider="hf_novita", cost_tier="mid",
    ),
    "qwen-3.5-35b": ModelConfig(
        name="qwen-3.5-35b", api_id="Qwen/Qwen3.5-35B-A3B:novita",
        provider="hf_novita", cost_tier="cheap",
    ),

    # DeepSeek updated family (via HF → Novita)
    "deepseek-r1-0528": ModelConfig(
        name="deepseek-r1-0528", api_id="deepseek-ai/DeepSeek-R1-0528:novita",
        provider="hf_novita", is_reasoning=True,
        supports_logprobs=False, max_output_tokens=16000, cost_tier="mid",
    ),
    "deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2", api_id="deepseek-ai/DeepSeek-V3-0324:novita",
        provider="hf_novita", supports_logprobs=False, cost_tier="mid",
        notes="Novita provider does not return logprobs for DeepSeek models",
    ),
}


# ──────────────────────────────────────────────
#  Unified registry
# ──────────────────────────────────────────────

# Models that need special access (not currently available via configured providers)
UNAVAILABLE_MODELS: dict[str, ModelConfig] = {
    "llama-4-maverick": ModelConfig(
        name="llama-4-maverick",
        api_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        provider="hf_novita", cost_tier="mid",
        notes="NOT available on HF/Novita router. Needs dedicated endpoint.",
    ),
}

MODEL_REGISTRY: dict[str, ModelConfig] = {**PAPER_MODELS, **NEW_MODELS}
MODEL_REGISTRY_ALL: dict[str, ModelConfig] = {
    **PAPER_MODELS, **PAPER_MODELS_OPTIONAL, **NEW_MODELS, **UNAVAILABLE_MODELS,
}


def get_model(name: str) -> ModelConfig:
    """Look up a model by its canonical short name."""
    if name in MODEL_REGISTRY_ALL:
        return MODEL_REGISTRY_ALL[name]
    # Fallback: try matching by api_id for users who pass the raw provider ID
    for cfg in MODEL_REGISTRY_ALL.values():
        if cfg.api_id == name:
            return cfg
    raise KeyError(
        f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY_ALL.keys())}"
    )


def available_models(include_optional: bool = False) -> list[str]:
    """Return sorted list of model names that have the required API key set."""
    import os
    registry = MODEL_REGISTRY_ALL if include_optional else MODEL_REGISTRY
    return sorted(
        name for name, cfg in registry.items()
        if os.environ.get(cfg.env_var)
    )


def models_by_provider(provider: Provider) -> list[str]:
    """List model names for a specific provider."""
    return sorted(
        name for name, cfg in MODEL_REGISTRY_ALL.items()
        if cfg.provider == provider
    )


def print_registry(verbose: bool = False) -> None:
    """Print a human-readable summary of all registered models."""
    import os

    print(f"\n{'Model':<25} {'Provider':<12} {'Logprobs':<10} {'Reasoning':<10} {'Paper':<7} {'Key?':<5}")
    print("─" * 79)
    for name, cfg in sorted(MODEL_REGISTRY_ALL.items()):
        has_key = "✓" if os.environ.get(cfg.env_var) else "✗"
        print(
            f"{name:<25} {cfg.provider:<12} "
            f"{'yes' if cfg.supports_logprobs else 'no':<10} "
            f"{'yes' if cfg.is_reasoning else 'no':<10} "
            f"{'yes' if cfg.is_paper_model else 'no':<7} "
            f"{has_key:<5}"
        )
    if verbose:
        avail = available_models(include_optional=True)
        print(f"\n{len(avail)} models available with current API keys: {avail}")
