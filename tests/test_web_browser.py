import json
import socket

import pytest

from tars import approvals, browser_tools, policy, state_store, tool_core, web_research


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
    navigate_request = policy.ScopeRequest(
        "browser.navigate", "network", "https://example.com/",
        {"profile": "dedicated-tars", "allowed_hosts": ["example.com"]},
        allowed_hosts=("example.com",),
    )
    navigation = browser.navigate("https://example.com", approval_id=_approve(navigate_request))
    assert driver.allowed_hosts == ()
    snapshot = browser.action("snapshot")
    click_request = policy.ScopeRequest("browser.click", "write", "browser-session", {"ref": "e1"})
    click = browser.action("click", ref="e1", approval_id=_approve(click_request))
    assert navigation.succeeded and snapshot.data["refs"][0]["ref"] == "e1"
    assert click.succeeded and browser.status()["profile_policy"] == "dedicated-tars"
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
