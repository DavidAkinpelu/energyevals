"""Unit tests for MCP Client URL support."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from energbench.mcp import MCPClient, MCPServerConfig, get_default_mcp_servers


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_config_with_command(self):
        """Test creating config with command."""
        config = MCPServerConfig(
            name="test-server",
            command="test-command",
            description="Test server",
        )
        assert config.name == "test-server"
        assert config.command == "test-command"
        assert config.url is None

    def test_config_with_url(self):
        """Test creating config with URL."""
        config = MCPServerConfig(
            name="test-server",
            url="https://example.com/sse",
            description="Test server",
        )
        assert config.name == "test-server"
        assert config.url == "https://example.com/sse"
        assert config.command is None

    def test_config_validation_fails_without_command_or_url(self):
        """Test that config validation fails without command or URL."""
        with pytest.raises(ValueError, match="must have either 'command' or 'url'"):
            MCPServerConfig(name="test-server", description="Test server")

    def test_config_with_both_command_and_url(self):
        """Test creating config with both command and URL (allowed, URL takes precedence)."""
        config = MCPServerConfig(
            name="test-server",
            command="test-command",
            url="https://example.com/sse",
            description="Test server",
        )
        assert config.command == "test-command"
        assert config.url == "https://example.com/sse"


class TestGetDefaultMCPServers:
    """Tests for get_default_mcp_servers function."""

    def test_default_local_servers(self):
        """Test default configuration uses local servers."""
        with patch.dict(os.environ, {}, clear=True):
            servers = get_default_mcp_servers()
            
            assert len(servers) == 2
            
            rag_server = next(s for s in servers if s.name == "energy-rag")
            assert rag_server.command == "energy-rag-server"
            assert rag_server.url is None
            assert "local" in rag_server.description.lower()
            
            db_server = next(s for s in servers if s.name == "energy-database")
            assert db_server.command == "energy-database-server"
            assert db_server.url is None
            assert "local" in db_server.description.lower()

    def test_remote_servers_from_env_vars(self):
        """Test configuration uses URLs from environment variables."""
        env_vars = {
            "RAG_SERVER_URL": "https://rag-server.com/sse",
            "DATABASE_SERVER_URL": "https://db-server.com/sse",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            servers = get_default_mcp_servers()
            
            assert len(servers) == 2
            
            rag_server = next(s for s in servers if s.name == "energy-rag")
            assert rag_server.url == "https://rag-server.com/sse"
            assert rag_server.command is None
            assert "remote" in rag_server.description.lower()
            
            db_server = next(s for s in servers if s.name == "energy-database")
            assert db_server.url == "https://db-server.com/sse"
            assert db_server.command is None
            assert "remote" in db_server.description.lower()

    def test_mixed_local_and_remote_servers(self):
        """Test configuration with one local and one remote server."""
        env_vars = {
            "RAG_SERVER_URL": "https://rag-server.com/sse",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            servers = get_default_mcp_servers()
            
            assert len(servers) == 2
            
            rag_server = next(s for s in servers if s.name == "energy-rag")
            assert rag_server.url == "https://rag-server.com/sse"
            assert "remote" in rag_server.description.lower()
            
            db_server = next(s for s in servers if s.name == "energy-database")
            assert db_server.command == "energy-database-server"
            assert "local" in db_server.description.lower()


class TestMCPClientConnection:
    """Tests for MCPClient connection logic."""

    @pytest.mark.asyncio
    async def test_connect_routes_to_stdio(self):
        """Test that servers with command use stdio connection."""
        config = MCPServerConfig(
            name="test-server",
            command="test-command",
        )
        
        client = MCPClient([config])
        
        # Mock the _connect_stdio method
        client._connect_stdio = AsyncMock()
        client._connect_sse = AsyncMock()
        
        # Mock session and tools
        mock_session = MagicMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = []
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        client._sessions["test-server"] = mock_session
        
        await client.connect()
        
        # Verify stdio was called, not SSE
        client._connect_stdio.assert_called_once()
        client._connect_sse.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_routes_to_sse(self):
        """Test that servers with URL use SSE connection."""
        config = MCPServerConfig(
            name="test-server",
            url="https://example.com/sse",
        )
        
        client = MCPClient([config])
        
        # Mock the connection methods
        client._connect_stdio = AsyncMock()
        client._connect_sse = AsyncMock()
        
        # Mock session and tools
        mock_session = MagicMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = []
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        client._sessions["test-server"] = mock_session
        
        await client.connect()
        
        # Verify SSE was called, not stdio
        client._connect_sse.assert_called_once()
        client._connect_stdio.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_connections(self):
        """Test that disconnect cleans up sessions and tools."""
        client = MCPClient([])

        # Mock a session
        mock_session = MagicMock()

        # Mock exit stack
        mock_exit_stack = MagicMock()
        mock_exit_stack.aclose = AsyncMock()

        client._sessions["test-server"] = mock_session
        client._tools["test-tool"] = {"name": "test", "description": "test", "parameters": {}}
        client._exit_stack = mock_exit_stack
        client._connected = True

        await client.disconnect()

        # Verify cleanup
        assert len(client._sessions) == 0
        assert len(client._tools) == 0
        assert client._exit_stack is None
        assert not client._connected
        mock_exit_stack.aclose.assert_called_once()


class TestMCPClientProperties:
    """Tests for MCPClient properties."""

    def test_is_connected_false_initially(self):
        """Test that is_connected is False initially."""
        client = MCPClient([])
        assert not client.is_connected

    def test_is_connected_true_after_connect(self):
        """Test that is_connected is True after successful connection."""
        client = MCPClient([])
        client._connected = True
        client._sessions["test"] = MagicMock()
        assert client.is_connected

    def test_is_connected_false_without_sessions(self):
        """Test that is_connected is False even if _connected is True but no sessions."""
        client = MCPClient([])
        client._connected = True
        assert not client.is_connected

