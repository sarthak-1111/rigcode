from __future__ import annotations

import asyncio
import socket

from rigcode.llm import probe


def _dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_explicit_url_and_auto_model(server):
    provider = {"base_url": server.base_url, "model": "auto"}
    info, notice = asyncio.run(probe.discover(provider))
    assert info.reachable and info.kind == "llama.cpp"
    assert provider["model"] == "mock-model"
    assert notice == "Using model mock-model"


def test_configured_model_is_kept(server):
    provider = {"base_url": server.base_url, "model": "my-model"}
    info, notice = asyncio.run(probe.discover(provider))
    assert info.reachable and provider["model"] == "my-model" and notice is None


def test_auto_url_finds_the_first_local_server(server, monkeypatch):
    monkeypatch.setattr(probe, "LOCAL_SERVERS", (("dead", _dead_url()), ("mock", server.base_url)))
    provider = {"base_url": "auto", "model": "auto"}
    info, notice = asyncio.run(probe.discover(provider))
    assert info.reachable and provider["base_url"] == server.base_url
    assert notice == f"Found mock at {server.base_url}, using model mock-model"


def test_auto_url_without_any_server(monkeypatch):
    monkeypatch.setattr(probe, "LOCAL_SERVERS", (("dead", _dead_url()),))
    provider = {"base_url": "auto", "model": "auto"}
    info, notice = asyncio.run(probe.discover(provider))
    assert not info.reachable and "no local model server found" in info.error
    assert provider["base_url"] == "auto"
