from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from tars.runtime_http import local_runtime_destination, request_json


@contextmanager
def _http_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_runtime_transport_requires_a_literal_loopback_peer():
    assert local_runtime_destination(
        "http://127.0.0.1:8080/v1/models").addresses == ("127.0.0.1",)
    assert local_runtime_destination(
        "http://[::1]:8080/v1/models").addresses == ("::1",)
    localhost = local_runtime_destination("http://localhost:8080/v1/models")
    assert localhost.addresses == ("127.0.0.1",)
    with pytest.raises(ValueError, match="literal loopback"):
        local_runtime_destination("https://8.8.8.8/v1/models")


def test_local_runtime_transport_does_not_follow_redirects():
    reached = threading.Event()

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *_):
            pass

    with _http_server(Target) as target:
        target_url = f"http://127.0.0.1:{target.server_port}/escaped"

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(307)
                self.send_header("Location", target_url)
                self.end_headers()

            def log_message(self, *_):
                pass

        with _http_server(Redirect) as redirect:
            with pytest.raises(RuntimeError, match="HTTP status 307"):
                request_json(
                    "GET",
                    f"http://127.0.0.1:{redirect.server_port}/v1/models",
                )
    assert not reached.is_set()
