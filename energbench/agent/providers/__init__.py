"""LLM Provider implementations for energBench."""

from .anthropic_provider import AnthropicProvider
from .base_provider import BaseProvider
from .deepinfra_provider import DeepInfraProvider
from .openai_provider import OpenAIProvider

from energbench.agent.schema import (
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    # Base classes
    "BaseProvider",
    "Message",
    "ProviderResponse",
    "ToolCall",
    "ToolDefinition",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepInfraProvider",
]


def get_provider(
    provider_name: str,
    model: str | None = None,
    **kwargs,
) -> BaseProvider:
    """Factory function to get a provider by name.

    Args:
        provider_name: Name of the provider ("openai", "anthropic", "deepinfra").
        model: Optional model identifier. If not provided, uses provider default.
        **kwargs: Additional provider configuration.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "deepinfra": DeepInfraProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(providers.keys())}"
        )

    provider_class = providers[provider_name]

    if model:
        return provider_class(model=model, **kwargs)
    return provider_class(**kwargs)
