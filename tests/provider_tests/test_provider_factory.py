import pytest

from energbench.agent.providers import get_provider
from energbench.agent.providers.anthropic_provider import AnthropicProvider
from energbench.agent.providers.deepinfra_provider import DeepInfraProvider
from energbench.agent.providers.google_provider import GoogleProvider
from energbench.agent.providers.openai_provider import OpenAIProvider


class TestProviderFactory:
    """Tests for provider factory function."""

    def test_get_openai_provider(self, monkeypatch):
        """Test getting OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        provider = get_provider("openai", model="gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_get_anthropic_provider(self, monkeypatch):
        """Test getting Anthropic provider."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        provider = get_provider("anthropic", model="claude-sonnet-4-20250514")
        assert isinstance(provider, AnthropicProvider)

    def test_get_google_provider(self, monkeypatch):
        """Test getting Google provider."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
        provider = get_provider("google", model="gemini-2.0-flash")
        assert isinstance(provider, GoogleProvider)

    def test_get_deepinfra_provider(self, monkeypatch):
        """Test getting DeepInfra provider."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "test_key")
        provider = get_provider("deepinfra", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert isinstance(provider, DeepInfraProvider)

    def test_get_invalid_provider(self):
        """Test getting invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")
