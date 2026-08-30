import socket

import pytest

from tars import approvals, evidence, http_tools, policy, state_store


class FakeHTTP:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return next(self.responses)


@pytest.fixture
def http_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    policy.add_rule("network", "allow", target="example.com")


def test_http_get_tracks_evidence_bounds_and_conditional_cache(http_environment):
    first = http_tools.HTTPResponse(
        "https://example.com/doc", 200,
        {"Content-Type": "text/plain", "ETag": "one"}, b"documentation", False,
    )
    second = http_tools.HTTPResponse(
        "https://example.com/doc", 304, {"ETag": "one"}, b"", False,
    )
    transport = FakeHTTP((first, second))
    tools = http_tools.HTTPTools(transport=transport, max_bytes=100)
    result = tools.get("https://example.com/doc")
    cached = tools.get("https://example.com/doc")
    assert result.succeeded and result.data["content"] == "documentation"
    assert result.evidence_ids and cached.data["content"] == "documentation"
    assert transport.calls[1][2]["headers"]["If-None-Match"] == "one"


def test_http_redirect_revalidates_and_denies_private_or_cross_host(http_environment):
    private = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/start", 302, {"Location": "http://127.0.0.1/private"}, b"", False,
    ),))
    with pytest.raises(ValueError, match="denied"):
        http_tools.HTTPTools(transport=private).get("https://example.com/start")
    cross = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/start", 302, {"Location": "https://other.example/path"}, b"", False,
    ),))
    with pytest.raises(PermissionError, match="cross-host"):
        http_tools.HTTPTools(transport=cross).get("https://example.com/start")


def test_http_state_change_requires_separate_write_policy(http_environment):
    transport = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/api", 204, {"Content-Type": "text/plain"}, b"", False,
    ),))
    with pytest.raises(PermissionError, match="approved authorization"):
        http_tools.HTTPTools(transport=transport).request(
            "POST", "https://example.com/api", body=b"change",
        )
    assert transport.calls == []


def test_http_response_size_and_binary_handling(http_environment):
    transport = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/file", 200,
        {"Content-Type": "application/octet-stream"}, b"abcde", True,
    ),))
    result = http_tools.HTTPTools(transport=transport, max_bytes=5).get(
        "https://example.com/file"
    )
    assert result.data["content"] == "" and result.data["truncated"]


def test_http_response_cookies_are_redacted(http_environment):
    transport = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/", 200,
        {"Content-Type": "text/plain", "Set-Cookie": "session=secret"}, b"ok", False,
    ),))
    result = http_tools.HTTPTools(transport=transport).get("https://example.com/")
    assert result.data["headers"]["Set-Cookie"] == "[REDACTED]"


def test_http_download_writes_bounded_verified_artifact(http_environment, tmp_path):
    output = tmp_path / "download.bin"
    transport = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/file", 200,
        {"Content-Type": "application/octet-stream"}, b"payload", False,
    ),))
    request = policy.ScopeRequest(
        "http.download", "write", str(output), {"url": "https://example.com/file"},
        allowed_paths=(str(tmp_path),),
    )
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    result = http_tools.HTTPTools(transport=transport).download(
        "https://example.com/file", output, allowed_paths=(str(tmp_path),),
        approval_ids={"output": pending.id},
    )
    assert result.succeeded and output.read_bytes() == b"payload"
    assert result.data["verified_bytes"] == 7 and len(result.data["sha256"]) == 64


def test_artifact_claim_verification_uses_retained_source_chunk(http_environment, tmp_path):
    transport = FakeHTTP((http_tools.HTTPResponse(
        "https://example.com/doc", 200, {"Content-Type": "text/plain"},
        b"Run full upgrades to preserve consistency.", False,
    ),))
    source = http_tools.HTTPTools(transport=transport).get("https://example.com/doc")
    artifact = tmp_path / "guide.md"
    artifact.write_text("Use full upgrades.")
    result = evidence.verify_artifact(artifact, [{
        "text": "Use full upgrades.", "evidence_id": source.evidence_ids[0],
        "supporting_text": "full upgrades",
    }])
    assert result["verified"] and evidence.load(result["evidence_id"]).result_ref == str(artifact)
