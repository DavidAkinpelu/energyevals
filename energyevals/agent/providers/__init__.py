from typing import Any

from energyevals.agent.schema import (
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from energyevals.core.types import ProviderName

from .anthropic_provider import AnthropicProvider
from .base_provider import BaseProvider
from .deepinfra_provider import DeepInfraProvider
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "Message",
    "ProviderName",
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
    providers: dict[ProviderName, type[BaseProvider]] = {
        ProviderName.OPENAI: OpenAIProvider,
        ProviderName.ANTHROPIC: AnthropicProvider,
        ProviderName.GOOGLE: GoogleProvider,
        ProviderName.DEEPINFRA: DeepInfraProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(providers.keys())}"
        )

    provider_class = providers[provider_name]  # type: ignore[index]

    if model:
        return provider_class(model=model, **kwargs)
    return provider_class(**kwargs)
