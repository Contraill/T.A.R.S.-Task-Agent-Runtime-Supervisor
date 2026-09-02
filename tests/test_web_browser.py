from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import threading

import pytest

from tars import (approvals, browser_tools, network, policy, state_store, tool_core,
                  web_research)


class FakeBrowser:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.allowed_hosts = None

    def set_allowed_hosts(self, hosts):
        self.allowed_hosts = tuple(hosts)

    def call(self, operation, **kwargs):
        self.calls.append((operation, kwargs))
        if operation == "snapshot":
            return {"url": "https://example.com", "content": "Page", "refs": [{"ref": "e1"}]}
        return {"url": kwargs.get("url", "https://example.com"), "path": kwargs.get("path", "")}

    def close(self): self.closed = True


class FakeResearchHTTP:
    def __init__(self): self.calls = []
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return tool_core.ToolResult("http.request", "succeeded", {
            "content": json.dumps({"results": [{"url": "https://docs.example/source"}]})
        }, action_ids=("action",), evidence_ids=("evidence",))


@pytest.fixture
def web_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    return tmp_path


def _approve(request):
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision)
    broker.decide(pending.id, approve=True)
    return pending.id


def test_browser_uses_dedicated_profile_stable_actions_and_evidence(web_environment):
    driver = FakeBrowser()
    profile = web_environment / "tars-profile"
    downloads = web_environment / "downloads"
    browser = browser_tools.BrowserTools(
        profile=profile, downloads=downloads, driver=driver,
    )
    destination = network.network_destination("https://example.com")
    navigate_request = policy.ScopeRequest(
        "browser.navigate", "network", "https://example.com/",
        {"profile": "dedicated-tars", "allowed_origins": ["https://example.com:443"],
         "url_sha256": destination.url_sha256},
        allowed_hosts=("https://example.com:443",),
    )
    navigation = browser.navigate("https://example.com", approval_id=_approve(navigate_request))
    assert driver.allowed_hosts == ()
    snapshot = browser.action("snapshot")
    click_request = policy.ScopeRequest("browser.click", "write", "browser-session", {"ref": "e1"})
    click = browser.action("click", ref="e1", approval_id=_approve(click_request))
    assert navigation.succeeded and snapshot.data["refs"][0]["ref"] == "e1"
    assert click.succeeded and browser.status()["profile_policy"] == "dedicated-tars"
    assert browser.status()["network_contract"] == {
        "javascript": "enabled",
        "http": "exact-origin pinned-peer transport",
        "service_workers": "blocked",
        "websocket": "blocked until a pinned-peer transport exists",
        "webrtc": "blocked",
        "webtransport": "blocked",
    }
    assert snapshot.evidence_ids and profile.is_dir() and downloads.is_dir()
    browser.close()
    assert driver.closed


def test_browser_personal_profile_private_network_and_raw_evaluation_are_blocked(web_environment):
    with pytest.raises(PermissionError, match="opt-in"):
        browser_tools.BrowserTools(
            profile=web_environment / "personal", driver=FakeBrowser(), personal_profile=True,
        )
    browser = browser_tools.BrowserTools(
        profile=web_environment / "profile", downloads=web_environment / "downloads",
        driver=FakeBrowser(),
    )
    with pytest.raises(ValueError, match="denied"):
        browser.navigate("http://127.0.0.1/admin")
    with pytest.raises(PermissionError):
        browser.action("evaluate", script="document.cookie")


@pytest.mark.parametrize(
    "final_url", ("http://example.com/", "https://example.com:444/"),
)
def test_browser_navigation_cannot_change_authorized_origin(web_environment, final_url):
    class RedirectingBrowser(FakeBrowser):
        def call(self, operation, **kwargs):
            self.calls.append((operation, kwargs))
            return {"url": final_url}

    destination = network.network_destination("https://example.com")
    request = policy.ScopeRequest(
        "browser.navigate", "network", destination.policy_url,
        {"profile": "dedicated-tars",
         "allowed_origins": [destination.origin],
         "url_sha256": destination.url_sha256},
        allowed_hosts=(destination.origin,),
    )
    browser = browser_tools.BrowserTools(
        profile=web_environment / "profile", downloads=web_environment / "downloads",
        driver=RedirectingBrowser(),
    )
    with pytest.raises(PermissionError, match="authorized destination"):
        browser.navigate(destination.request_url, approval_id=_approve(request))
    assert browser.driver.allowed_hosts == ()


def test_browser_screenshot_is_forced_into_isolated_location(web_environment):
    driver = FakeBrowser()
    downloads = web_environment / "downloads"
    browser = browser_tools.BrowserTools(
        profile=web_environment / "profile", downloads=downloads, driver=driver,
    )
    request = policy.ScopeRequest(
        "browser.screenshot", "write", "browser-session",
        {"path": str(downloads / "personal.png")},
    )
    result = browser.action(
        "screenshot", path="../../personal.png", approval_id=_approve(request),
    )
    assert result.data["path"] == str(downloads / "personal.png")


def test_browser_output_root_replacement_cannot_redirect_screenshot(
        web_environment, monkeypatch):
    class OutputBrowser(FakeBrowser):
        def call(self, operation, **kwargs):
            if operation == "screenshot":
                Path(kwargs["path"]).write_bytes(b"image")
            return super().call(operation, **kwargs)

    downloads = web_environment / "downloads"
    outside = web_environment / "outside"
    outside.mkdir()
    browser = browser_tools.BrowserTools(
        profile=web_environment / "profile", downloads=downloads,
        driver=OutputBrowser(),
    )
    request = policy.ScopeRequest(
        "browser.screenshot", "write", "browser-session",
        {"path": str(downloads / "screen.png")},
    )
    approval = _approve(request)
    authorize = browser.runtime.authorize

    def swap_after_authorization(*args, **kwargs):
        actions = authorize(*args, **kwargs)
        downloads.rename(web_environment / "displaced-downloads")
        downloads.symlink_to(outside, target_is_directory=True)
        return actions

    monkeypatch.setattr(browser.runtime, "authorize", swap_after_authorization)
    result = browser.action("screenshot", path="screen.png", approval_id=approval)
    assert result.succeeded
    assert not (outside / "screen.png").exists()
    assert (web_environment / "displaced-downloads" / "screen.png").read_bytes() == b"image"


def test_live_browser_javascript_uses_bound_http_and_cannot_escape_origin(tmp_path):
    if importlib.util.find_spec("playwright") is None:
        pytest.skip("Playwright is not installed")

    class DeniedHandler(BaseHTTPRequestHandler):
        requests = []

        def _record(self):
            type(self).requests.append((self.command, self.path))
            self.send_response(204)
            self.end_headers()

        do_GET = do_POST = _record

        def log_message(self, *_):
            pass

    denied = ThreadingHTTPServer(("127.0.0.1", 0), DeniedHandler)
    denied_thread = threading.Thread(target=denied.serve_forever, daemon=True)
    denied_thread.start()
    denied_origin = f"http://127.0.0.1:{denied.server_port}"

    class AllowedHandler(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            type(self).requests.append((self.command, self.path))
            if self.path == "/":
                cross = json.dumps(denied_origin)
                body = f"""<!doctype html><body><div id='app'>initial</div><script>
                    const cross = {cross};
                    document.querySelector('#app').textContent = 'javascript-ready';
                    fetch('/api').then(r => r.text()).then(
                        value => document.body.dataset.api = value);
                    const worker = new Worker('/worker.js');
                    worker.onmessage = event => document.body.dataset.worker = event.data;
                    fetch(cross + '/fetch').catch(() => {{}});
                    const xhr = new XMLHttpRequest();
                    xhr.open('GET', cross + '/xhr'); xhr.send();
                    for (const tag of ['img', 'iframe', 'script']) {{
                        const node = document.createElement(tag);
                        node.src = cross + '/' + tag; document.body.appendChild(node);
                    }}
                    const formFrame = document.createElement('iframe');
                    formFrame.name = 'form-target'; document.body.appendChild(formFrame);
                    const form = document.createElement('form');
                    form.method = 'POST'; form.action = cross + '/form';
                    form.target = formFrame.name; document.body.appendChild(form); form.submit();
                    navigator.sendBeacon(cross + '/beacon', 'payload');
                    try {{
                        const events = new EventSource(cross + '/events');
                        setTimeout(() => events.close(), 100);
                    }} catch (_) {{}}
                    try {{ new WebSocket(cross.replace('http:', 'ws:') + '/socket'); }} catch (_) {{}}
                    window.open(cross + '/popup', 'blocked-popup');
                    navigator.serviceWorker.register('/sw.js').catch(() => {{}});
                </script>""".encode()
                content_type = "text/html"
            elif self.path == "/api":
                body, content_type = b"api-ready", "text/plain"
            elif self.path == "/worker.js":
                body, content_type = b"postMessage('worker-ready')", "text/javascript"
            elif self.path == "/sw.js":
                body, content_type = b"", "text/javascript"
            else:
                body, content_type = b"not-found", "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    allowed = ThreadingHTTPServer(("127.0.0.1", 0), AllowedHandler)
    allowed_thread = threading.Thread(target=allowed.serve_forever, daemon=True)
    allowed_thread.start()
    allowed_origin = f"http://127.0.0.1:{allowed.server_port}"
    profile = tmp_path / "profile"
    downloads = tmp_path / "downloads"
    profile.mkdir()
    downloads.mkdir()
    driver = None
    try:
        try:
            driver = browser_tools.PlaywrightDriver(
                profile, downloads, allow_loopback=True,
            )
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                pytest.skip("Playwright Chromium is not installed")
            raise
        driver.set_allowed_origins((allowed_origin,))
        driver.call("navigate", url=allowed_origin + "/")
        driver.page.wait_for_function(
            "document.body.dataset.api === 'api-ready' && "
            "document.body.dataset.worker === 'worker-ready'",
            timeout=10_000,
        )
        driver.page.wait_for_timeout(500)
        assert driver.page.locator("#app").inner_text() == "javascript-ready"
        assert driver.page.evaluate(
            "[typeof RTCPeerConnection, typeof WebSocket, typeof WebTransport]"
        ) == ["undefined", "undefined", "undefined"]
        assert ("GET", "/api") in AllowedHandler.requests
        assert ("GET", "/worker.js") in AllowedHandler.requests
        assert driver.context.service_workers == []
        assert not DeniedHandler.requests
    finally:
        if driver is not None:
            driver.close()
        allowed.shutdown()
        allowed.server_close()
        allowed_thread.join()
        denied.shutdown()
        denied.server_close()
        denied_thread.join()


def test_tavily_is_optional_and_secret_reference_is_not_exposed(monkeypatch, web_environment):
    missing = web_research.TavilyResearch(http=FakeResearchHTTP())
    result = missing.search("query")
    assert result.state == "unavailable" and "credential" in result.error
    monkeypatch.setenv("TAVILY_API_KEY", "raw-tavily-secret")
    http = FakeResearchHTTP()
    research = web_research.TavilyResearch(http=http)
    result = research.search("Arch maintenance", max_results=3)
    assert result.succeeded and result.evidence_ids == ("evidence",)
    body = json.loads(http.calls[0][2]["body"])
    assert body["api_key"] == "raw-tavily-secret"
    assert "raw-tavily-secret" not in str(result)
    assert http.calls[0][2]["sensitive_values"] == ("raw-tavily-secret",)
