"""Tests for gateway module."""

import pytest

from stackchan_mcp.gateway import Gateway, get_gateway


def test_get_gateway_singleton():
    """get_gateway returns the same instance."""
    # Reset singleton for test isolation
    import stackchan_mcp.gateway as gw_mod
    gw_mod._gateway = None

    g1 = get_gateway()
    g2 = get_gateway()
    assert g1 is g2

    # Cleanup
    gw_mod._gateway = None


@pytest.mark.asyncio
async def test_gateway_start_stop():
    """Gateway can start and stop."""
    import os
    os.environ["WS_PORT"] = "0"  # Random port

    gw = Gateway()
    await gw.start()
    assert gw._running is True
    assert gw.esp32._server is not None

    await gw.stop()
    assert gw._running is False

    if "WS_PORT" in os.environ:
        del os.environ["WS_PORT"]
