"""Tests for stdio MCP server tool definitions."""

from stackchan_mcp.stdio_server import create_server


def test_create_server():
    """Server creation succeeds with correct name."""
    server = create_server()
    assert server is not None
    assert server.name == "stackchan-mcp"
