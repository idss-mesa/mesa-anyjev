"""The gateway backend against a stub HTTP server (AnyJev's own vLLM test pattern):
logprobs=20 always, no allowed_token_ids, missing labels floored and counted, no hidden
states, loopback only, breaker."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import pytest

from mesa_anyjev.backends.gateway import (
    MISSING_LOGPROB,
    CircuitBreaker,
    GatewayBackend,
    GatewayError,
    assert_loopback,
)


class FakeTok:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(text.strip()[0])]

    def decode(self, ids: list[int]) -> str:
        return chr(ids[0])

    def convert_ids_to_tokens(self, tid: int) -> str:
        return chr(tid)


class _Handler(BaseHTTPRequestHandler):
    bodies: list[dict[str, Any]] = []
    top: dict[str, float] = {"A": -0.1, "B": -2.5, "Y": -0.2, "N": -1.7}
    fail_400 = False

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health/liveliness" else 404)
        self.end_headers()
        self.wfile.write(b'"I\'m alive!"')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        _Handler.bodies.append(body)
        if _Handler.fail_400 or body.get("logprobs", 0) > 20:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "greater than max allowed: 20"}')
            return
        payload = {"choices": [{"text": "A", "logprobs": {"top_logprobs": [dict(_Handler.top)]}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Any:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.bodies.clear()
    _Handler.fail_400 = False
    yield srv
    srv.shutdown()


def _backend(srv: HTTPServer, **kw: Any) -> GatewayBackend:
    return GatewayBackend(
        f"http://127.0.0.1:{srv.server_port}",
        "carc-fast",
        "tok",
        None,
        "Canon/Model",
        "key",
        tokenizer_obj=FakeTok(),
        workers=1,
        retries=1,
        **kw,
    )


def test_request_shape_and_missing_labels(server: HTTPServer) -> None:
    be = _backend(server)
    lp = be.next_token_logprobs(["p1"], [[ord("A"), ord("B"), ord("Z")]])[0]
    body = _Handler.bodies[-1]
    assert (
        body["logprobs"] == 20 and "allowed_token_ids" not in body and body["model"] == "carc-fast"
    )
    assert body["max_tokens"] == 1 and body["temperature"] == 0.0
    assert np.allclose(lp, [-0.1, -2.5, MISSING_LOGPROB])
    assert be.missing_label_events == 1
    assert be.name == "Canon/Model" and be.served_model == "carc-fast"
    assert not hasattr(be, "hidden_states") and not hasattr(be, "n_layers")


def test_probe_and_cap(server: HTTPServer) -> None:
    be = _backend(server)
    probe = be.probe()
    assert probe.ok and probe.liveness and probe.labels_present and probe.top_k == 20
    with pytest.raises(GatewayError, match="rejected"):
        be._post({"model": "carc-fast", "prompt": "x", "max_tokens": 1, "logprobs": 26})


def test_breaker_opens_after_failures(server: HTTPServer) -> None:
    be = _backend(server, breaker=CircuitBreaker(failures=2, open_for=60))
    _Handler.fail_400 = True
    for _ in range(2):
        with pytest.raises(GatewayError):
            be._one("p", [ord("A")])
    assert be.breaker.is_open
    with pytest.raises(GatewayError, match="breaker"):
        be._one("p", [ord("A")])


def test_loopback_only() -> None:
    assert assert_loopback("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    with pytest.raises(GatewayError):
        assert_loopback("http://10.100.32.1:8000")
    with pytest.raises(GatewayError):
        assert_loopback("http://127.0.0.1:8000/v1")
