from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import stat
import tempfile

from .config import DATA_ROOT
from .network import network_destination, open_bound
from .policy import ScopeRequest
from .secure_paths import AnchoredRoot
from .tool_core import ToolResult, ToolRuntime


class PlaywrightDriver:
    def __init__(self, profile, downloads, *, headless=True,
                 max_resource_bytes=16_000_000):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed") from exc
        self._playwright = sync_playwright().start()
        self.context = self._playwright.chromium.launch_persistent_context(
            str(profile), headless=headless, accept_downloads=True,
            downloads_path=str(downloads), service_workers="block",
            java_script_enabled=False,
            args=[
                "--host-resolver-rules=MAP * ~NOTFOUND",
                "--disable-quic",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            ],
        )
        self.allowed_origins = set()
        self.max_resource_bytes = int(max_resource_bytes)
        self.peer_addresses = {}

        def route_request(route):
            url = route.request.url
            if not url.startswith(("http://", "https://")):
                if url.startswith(("about:", "blob:", "data:")):
                    route.continue_()
                else:
                    route.abort("blockedbyclient")
                return
            try:
                destination = network_destination(url, resolve_dns=True)
                if destination.origin not in self.allowed_origins:
                    raise PermissionError("browser request left its authorized origin set")
                request_headers = route.request.all_headers()
                with open_bound(
                    destination,
                    method=route.request.method,
                    headers=request_headers,
                    body=route.request.post_data_buffer,
                    timeout=30,
                ) as response:
                    body = response.read(self.max_resource_bytes + 1)
                    if len(body) > self.max_resource_bytes:
                        raise RuntimeError("browser resource exceeded size limit")
                    self.peer_addresses[destination.request_url] = response.peer_ip
                    headers = {
                        key: value for key, value in response.headers.items()
                        if key.casefold() not in {
                            "connection", "content-length", "transfer-encoding",
                        }
                    }
                    route.fulfill(status=response.status, headers=headers, body=body)
            except Exception:
                route.abort("blockedbyclient")
        self.context.route("**/*", route_request)
        self.context.route_web_socket(
            "**/*", lambda route: route.close(code=1008, reason="network authority required")
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

    def set_allowed_origins(self, origins):
        self.allowed_origins = set(origins)

    def call(self, operation, **kwargs):
        if operation == "navigate": return {"url": self.page.goto(kwargs["url"]).url}
        if operation == "snapshot":
            elements = self.page.locator("a,button,input,select,textarea")
            refs = []
            for index in range(min(elements.count(), 500)):
                element = elements.nth(index)
                ref = f"e{index + 1}"
                element.evaluate("(node, value) => node.setAttribute('data-tars-ref', value)", ref)
                refs.append({"ref": ref, "tag": element.evaluate("node => node.tagName.toLowerCase()"),
                             "text": element.inner_text() if element.is_visible() else ""})
            return {"url": self.page.url, "content": self.page.locator("body").inner_text(),
                    "refs": refs}
        if operation == "click": self.page.locator(f'[data-tars-ref="{kwargs["ref"]}"]').click(); return {"url": self.page.url}
        if operation == "type": self.page.locator(f'[data-tars-ref="{kwargs["ref"]}"]').fill(kwargs["text"]); return {"url": self.page.url}
        if operation == "select": self.page.locator(f'[data-tars-ref="{kwargs["ref"]}"]').select_option(kwargs["value"]); return {"url": self.page.url}
        if operation == "key": self.page.keyboard.press(kwargs["key"]); return {"url": self.page.url}
        if operation == "scroll": self.page.mouse.wheel(0, int(kwargs["amount"])); return {"url": self.page.url}
        if operation == "wait": self.page.wait_for_timeout(int(kwargs["milliseconds"])); return {"url": self.page.url}
        if operation == "screenshot": self.page.screenshot(path=kwargs["path"], full_page=kwargs.get("full_page", False)); return {"url": self.page.url, "path": kwargs["path"]}
        if operation == "download":
            with self.page.expect_download() as download:
                self.page.locator(f'[data-tars-ref="{kwargs["ref"]}"]').click()
            path = str(Path(kwargs["directory"]) / download.value.suggested_filename)
            download.value.save_as(path)
            return {"url": self.page.url, "path": path}
        if operation == "back": self.page.go_back(); return {"url": self.page.url}
        if operation == "reload": self.page.reload(); return {"url": self.page.url}
        if operation == "tabs": return {"tabs": [{"index": i, "url": page.url} for i, page in enumerate(self.context.pages)]}
        if operation == "new_tab": self.page = self.context.new_page(); return {"url": self.page.url}
        if operation == "close":
            closed_url = self.page.url
            self.page.close()
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            return {"closed_url": closed_url, "url": self.page.url}
        raise ValueError(f"unsupported browser operation: {operation}")

    def close(self):
        self.context.close()
        self._playwright.stop()


class BrowserTools:
    def __init__(self, *, profile=None, downloads=None, driver=None, runtime=None,
                 personal_profile=False, personal_opt_in=False,
                 max_output_bytes=16_000_000):
        if personal_profile and not personal_opt_in:
            raise PermissionError("personal browser profiles require explicit opt-in")
        self.profile = Path(profile or (DATA_ROOT / "browser" / "profile")).expanduser().resolve()
        self.downloads = Path(downloads or (DATA_ROOT / "browser" / "downloads")).expanduser().resolve()
        self.profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.profile.chmod(0o700)
        self.downloads.chmod(0o700)
        self._downloads_anchor = AnchoredRoot(self.downloads)
        self.runtime = runtime or ToolRuntime()
        self.driver = driver
        self._injected_driver = driver is not None
        self._live_driver_started = False
        self.personal_profile = personal_profile
        self.max_output_bytes = int(max_output_bytes)

    def status(self):
        available = self.driver is not None or importlib.util.find_spec("playwright") is not None
        support = ("mock-tested" if self._injected_driver else
                   "reference-tested" if self._live_driver_started else
                   "installed-unverified" if available else "unavailable")
        return {"available": available, "support": support,
                "profile": str(self.profile), "downloads": str(self.downloads),
                "profile_policy": "explicit-personal" if self.personal_profile else "dedicated-tars"}

    def _driver(self):
        if self.driver is None:
            self.driver = PlaywrightDriver(self.profile, self.downloads)
            self._live_driver_started = True
        return self.driver

    def navigate(self, url, *, allowed_hosts=(), approval_id=None, task_id=None, session_id=None):
        destination = network_destination(url, resolve_dns=True)
        origins = {destination.origin}
        for value in allowed_hosts:
            candidate = str(value)
            if "://" not in candidate:
                candidate = f"{destination.scheme}://{candidate}"
            origins.add(network_destination(candidate, resolve_dns=True).origin)
        destinations = tuple(sorted(origins))
        request = ScopeRequest(
            "browser.navigate", "network", destination.policy_url,
            {"profile": "dedicated-tars", "allowed_origins": list(destinations),
             "url_sha256": destination.url_sha256},
            task_id=task_id, session_id=session_id, allowed_hosts=destinations,
        )
        actions = self.runtime.authorize((("network", request),), {"network": approval_id})
        retain_hosts = approval_id is None
        if approval_id is not None:
            retain_hosts = self.runtime.broker.load(approval_id).scope in {"session", "persistent"}
        try:
            driver = self._driver()
            if hasattr(driver, "set_allowed_origins"):
                driver.set_allowed_origins(destinations)
            elif hasattr(driver, "set_allowed_hosts"):
                driver.set_allowed_hosts(destinations)
            data = driver.call("navigate", url=destination.request_url)
            final = network_destination(
                data.get("url", destination.request_url), resolve_dns=False,
            )
            if final.origin not in origins:
                raise PermissionError("browser navigation left the authorized destination set")
        except Exception as exc:
            if 'driver' in locals() and not retain_hosts:
                if hasattr(driver, "set_allowed_origins"):
                    driver.set_allowed_origins(())
                elif hasattr(driver, "set_allowed_hosts"):
                    driver.set_allowed_hosts(())
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        if not retain_hosts:
            if hasattr(driver, "set_allowed_origins"):
                driver.set_allowed_origins(())
            elif hasattr(driver, "set_allowed_hosts"):
                driver.set_allowed_hosts(())
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("browser", destination.policy_url, repr(data), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("browser.navigate", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def action(self, operation, *, approval_id=None, task_id=None, session_id=None, **kwargs):
        read_operations = {"snapshot", "tabs", "wait"}
        if operation == "evaluate":
            effect, elevated = "elevated", True
        else:
            effect, elevated = ("read", False) if operation in read_operations else ("write", False)
        requested_path = None
        if operation == "download":
            kwargs = dict(kwargs) | {"directory": str(self.downloads)}
        if operation == "screenshot" and "path" in kwargs:
            requested_path = self.downloads / Path(kwargs["path"]).name
            kwargs = dict(kwargs) | {"path": str(requested_path)}
        request = ScopeRequest(
            f"browser.{operation}", effect, "browser-session", kwargs,
            task_id=task_id, session_id=session_id, elevated=elevated,
        )
        actions = self.runtime.authorize((("action", request),), {"action": approval_id})
        try:
            if operation in {"download", "screenshot"}:
                with tempfile.TemporaryDirectory(prefix="tars-browser-output-") as stage:
                    execution = dict(kwargs)
                    if operation == "download":
                        execution["directory"] = stage
                    else:
                        execution["path"] = str(Path(stage) / requested_path.name)
                    data = self._driver().call(operation, **execution)
                    staged_value = data.get("path", "")
                    staged = Path(staged_value) if staged_value else None
                    if (staged is not None and staged.parent == Path(stage)
                            and staged.exists()):
                        fd = os.open(staged, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                        try:
                            if not stat.S_ISREG(os.fstat(fd).st_mode):
                                raise ValueError("browser output must be a regular file")
                            payload = bytearray()
                            while True:
                                chunk = os.read(
                                    fd, min(1024 * 1024,
                                            self.max_output_bytes + 1 - len(payload))
                                )
                                if not chunk:
                                    break
                                payload.extend(chunk)
                                if len(payload) > self.max_output_bytes:
                                    raise RuntimeError("browser output exceeded size limit")
                        finally:
                            os.close(fd)
                        name = requested_path.name if requested_path is not None else staged.name
                        if not name or name in {".", ".."}:
                            raise ValueError("browser output filename is invalid")
                        output = self.downloads / name
                        self._downloads_anchor.atomic_write((name,), bytes(payload))
                        data = dict(data) | {"path": str(output)}
                    elif requested_path is not None:
                        if not self._injected_driver:
                            raise RuntimeError(
                                "browser screenshot did not create an output file"
                            )
                        data = dict(data) | {"path": str(requested_path)}
                    elif not self._injected_driver:
                        raise RuntimeError("browser download did not create an output file")
            else:
                data = self._driver().call(operation, **kwargs)
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence_ids = ()
        if operation in {"snapshot", "screenshot"}:
            evidence = self.runtime.evidence(
                "browser", data.get("url", "browser-session"), repr(data),
                task_id=task_id, event_uuid=actions[0].event_uuid,
                result_ref=data.get("path", ""),
            )
            evidence_ids = (evidence.id,)
        return ToolResult(f"browser.{operation}", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=evidence_ids)

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
        self._downloads_anchor.close()
