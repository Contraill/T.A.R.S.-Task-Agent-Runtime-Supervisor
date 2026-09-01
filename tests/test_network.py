from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest

from tars.core_client import CoreClient
from tars.network import network_destination


class _Server:
    def __init__(self, address, port=0, *, name, redirect=""):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization", ""),
                })
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", redirect)
                    self.end_headers()
                    return
                body = json.dumps({"server": name}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer((address, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_port

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()


def _answer(address, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]


def test_network_authority_redacts_query_secrets_but_binds_their_identity(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *args, **kwargs: _answer("93.184.216.34", port),
    )
    first = network_destination("https://example.com/api?token=alpha&view=full")
    second = network_destination("https://example.com/api?token=bravo&view=full")
    assert first.request_url.endswith("token=alpha&view=full")
    assert first.policy_url.endswith("token=%5BREDACTED%5D&view=full")
    assert first.url_sha256 != second.url_sha256
    assert "alpha" not in first.policy_url and "bravo" not in second.policy_url


def test_core_connection_uses_the_single_validated_dns_snapshot(monkeypatch):
    benign = _Server("127.0.0.1", name="benign")
    attacker = _Server("127.0.0.2", benign.port, name="attacker")
    resolutions = []

    def rebinding(host, port, *args, **kwargs):
        assert host == "core.test"
        address = "127.0.0.1" if not resolutions else "127.0.0.2"
        resolutions.append(address)
        return _answer(address, port)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    try:
        client = CoreClient(f"http://core.test:{benign.port}", "client.token")
        assert client.status() == {"server": "benign"}
        assert resolutions == ["127.0.0.1"]
        assert benign.requests == [{"path": "/v1/status",
                                    "authorization": "Bearer client.token"}]
        assert attacker.requests == []
    finally:
        benign.close()
        attacker.close()


def test_core_redirect_does_not_forward_bearer_credentials(monkeypatch):
    recipient = _Server("127.0.0.2", name="recipient")
    origin = _Server(
        "127.0.0.1", recipient.port, name="origin",
        redirect=f"http://127.0.0.2:{recipient.port}/stolen",
    )
    try:
        client = CoreClient(f"http://127.0.0.1:{origin.port}", "client.token")
        with pytest.raises(PermissionError, match="redirect"):
            client.status()
        assert origin.requests[0]["authorization"] == "Bearer client.token"
        assert recipient.requests == []
    finally:
        origin.close()
        recipient.close()


def test_remote_plaintext_core_rejects_token_and_pairing_code_before_transport(
        monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *args, **kwargs: _answer("93.184.216.34", port),
    )
    calls = []

    def transport(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsafe transport must not run")

    with pytest.raises(PermissionError, match="require HTTPS"):
        CoreClient("http://core.example", "client.token", transport=transport).status()
    with pytest.raises(PermissionError, match="require HTTPS"):
        CoreClient.pair(
            "http://core.example", "pairing-code", "client", transport=transport,
        )
    assert calls == []


def test_core_base_url_rejects_query_credentials():
    with pytest.raises(ValueError, match="query parameters"):
        CoreClient("https://core.example/?token=plaintext", "client.token")
    with pytest.raises(ValueError, match="query parameters"):
        CoreClient.pair(
            "https://core.example/?code=plaintext", "pairing-code", "client",
        )
