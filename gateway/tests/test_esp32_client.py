"""Tests for ESP32 client connection management."""

import asyncio
import json

import pytest
import pytest_asyncio
import websockets

from stackchan_mcp.esp32_client import ESP32Manager


@pytest_asyncio.fixture
async def manager():
    """Create and start an ESP32Manager on a free port."""
    mgr = ESP32Manager()
    await mgr.start("127.0.0.1", 0)  # Port 0 = OS picks a free port

    # Get the actual port
    server = mgr._server
    port = server.sockets[0].getsockname()[1]
    mgr._test_port = port

    yield mgr
    await mgr.stop()


@pytest.mark.asyncio
async def test_manager_starts_and_stops():
    """Manager can start and stop cleanly."""
    mgr = ESP32Manager()
    await mgr.start("127.0.0.1", 0)
    assert mgr._server is not None
    await mgr.stop()
    assert mgr._server is None


@pytest.mark.asyncio
async def test_no_device_connected():
    """call_tool returns error when no device is connected."""
    mgr = ESP32Manager()
    result, error = await mgr.call_tool("self.robot.set_head_angles", {"yaw": 0, "pitch": 0})
    assert result is None
    assert error is not None
    assert "not connected" in error["message"].lower() or "No ESP32" in error["message"]


@pytest.mark.asyncio
async def test_get_status_disconnected():
    """get_status returns disconnected state."""
    mgr = ESP32Manager()
    status = mgr.get_status()
    assert status["connected"] is False
    assert status["device_id"] is None


@pytest.mark.asyncio
async def test_esp32_hello_handshake(manager):
    """ESP32 can connect and complete hello handshake."""
    port = manager._test_port

    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        # Send hello
        hello = {
            "type": "hello",
            "version": 1,
            "features": {"mcp": True},
            "transport": "websocket",
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
        await ws.send(json.dumps(hello))

        # Receive hello response
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        resp = json.loads(resp_raw)
        assert resp["type"] == "hello"
        assert resp["version"] == 1
        assert "session_id" in resp

        # Receive initialize request from gateway
        init_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        init_msg = json.loads(init_raw)
        assert init_msg["type"] == "mcp"
        assert init_msg["payload"]["method"] == "initialize"

        # Send initialize response
        init_resp = {
            "session_id": init_msg["session_id"],
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "id": init_msg["payload"]["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-device", "version": "1.0.0"},
                },
            },
        }
        await ws.send(json.dumps(init_resp))

        # Receive tools/list request
        tools_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        tools_msg = json.loads(tools_raw)
        assert tools_msg["type"] == "mcp"
        assert tools_msg["payload"]["method"] == "tools/list"

        # Send tools/list response
        tools_resp = {
            "session_id": tools_msg["session_id"],
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "id": tools_msg["payload"]["id"],
                "result": {
                    "tools": [
                        {
                            "name": "self.robot.set_head_angles",
                            "description": "Set head angles",
                            "inputSchema": {"type": "object"},
                        }
                    ],
                    "nextCursor": "",
                },
            },
        }
        await ws.send(json.dumps(tools_resp))

        # Wait for manager to process
        await asyncio.sleep(0.2)

        # Verify connection is established
        assert manager.device_connected is True
        status = manager.get_status()
        assert status["connected"] is True
        assert status["tools_count"] == 1


@pytest.mark.asyncio
async def test_esp32_tool_call_relay(manager):
    """Gateway relays tool calls to ESP32."""
    port = manager._test_port

    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        # Complete handshake
        await _complete_handshake(ws, tools=[
            {"name": "self.robot.set_head_angles", "description": "Set head", "inputSchema": {}}
        ])

        await asyncio.sleep(0.2)

        # Now call tool via manager
        call_task = asyncio.create_task(
            manager.call_tool("self.robot.set_head_angles", {"yaw": 45, "pitch": 10})
        )

        # ESP32 receives the request
        req_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        req_msg = json.loads(req_raw)
        assert req_msg["type"] == "mcp"
        assert req_msg["payload"]["method"] == "tools/call"
        assert req_msg["payload"]["params"]["name"] == "self.robot.set_head_angles"
        assert req_msg["payload"]["params"]["arguments"] == {"yaw": 45, "pitch": 10}

        # ESP32 sends response
        tool_resp = {
            "session_id": req_msg["session_id"],
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "id": req_msg["payload"]["id"],
                "result": {
                    "content": [{"type": "text", "text": "true"}],
                    "isError": False,
                },
            },
        }
        await ws.send(json.dumps(tool_resp))

        # Verify result
        result, error = await asyncio.wait_for(call_task, timeout=5.0)
        assert error is None
        assert result["content"][0]["text"] == "true"


@pytest.mark.asyncio
async def test_esp32_disconnect_handling(manager):
    """Manager handles ESP32 disconnection gracefully."""
    port = manager._test_port

    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        await _complete_handshake(ws)
        await asyncio.sleep(0.2)
        assert manager.device_connected is True

    # Connection closed
    await asyncio.sleep(0.2)
    assert manager.device_connected is False


@pytest.mark.asyncio
async def test_auth_rejection(manager):
    """Unauthorized connections are rejected."""
    import os
    port = manager._test_port

    # Set token to require auth
    os.environ["STACKCHAN_TOKEN"] = "test-secret-token"
    try:
        # Try connecting without auth — should fail
        with pytest.raises(Exception):
            async with websockets.connect(
                f"ws://127.0.0.1:{port}",
                additional_headers={"Authorization": "Bearer wrong-token"},
            ) as ws:
                await ws.recv()
    finally:
        del os.environ["STACKCHAN_TOKEN"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _complete_handshake(ws, tools=None):
    """Complete the full ESP32 handshake sequence."""
    if tools is None:
        tools = []

    # Send hello
    hello = {
        "type": "hello",
        "version": 1,
        "features": {"mcp": True},
        "transport": "websocket",
    }
    await ws.send(json.dumps(hello))

    # Receive hello response
    await asyncio.wait_for(ws.recv(), timeout=5.0)

    # Receive and respond to initialize
    init_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    init_msg = json.loads(init_raw)
    init_resp = {
        "session_id": init_msg["session_id"],
        "type": "mcp",
        "payload": {
            "jsonrpc": "2.0",
            "id": init_msg["payload"]["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-device", "version": "1.0.0"},
            },
        },
    }
    await ws.send(json.dumps(init_resp))

    # Receive and respond to tools/list
    tools_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    tools_msg = json.loads(tools_raw)
    tools_resp = {
        "session_id": tools_msg["session_id"],
        "type": "mcp",
        "payload": {
            "jsonrpc": "2.0",
            "id": tools_msg["payload"]["id"],
            "result": {"tools": tools, "nextCursor": ""},
        },
    }
    await ws.send(json.dumps(tools_resp))
