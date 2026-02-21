from typing import Any

from energbench.agent.schema import (
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)

from .anthropic_provider import AnthropicProvider
from .base_provider import BaseProvider
from .deepinfra_provider import DeepInfraProvider
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider

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
    "GoogleProvider",
]


def get_provider(
    provider_name: str,
    model: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Factory function to get a provider by name.

    Args:
        provider_name: Name of the provider ("openai", "anthropic", "google", "deepinfra").
        model: Optional model identifier. If not provided, uses provider default.
        **kwargs: Additional provider configuration.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers: dict[str, type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
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
