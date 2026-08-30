from __future__ import annotations

from pathlib import Path
import importlib.util

from .config import DATA_ROOT
from .policy import ScopeRequest, normalize_network_target
from .tool_core import ToolResult, ToolRuntime


class PlaywrightDriver:
    def __init__(self, profile, downloads, *, headless=True):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed") from exc
        self._playwright = sync_playwright().start()
        self.context = self._playwright.chromium.launch_persistent_context(
            str(profile), headless=headless, accept_downloads=True,
            downloads_path=str(downloads),
        )
        self.allowed_hosts = set()
        def route_request(route):
            url = route.request.url
            if not url.startswith(("http://", "https://")):
                route.continue_()
                return
            try:
                _, host = normalize_network_target(url, resolve_dns=True)
            except ValueError:
                route.abort("blockedbyclient")
                return
            if self.allowed_hosts and host not in self.allowed_hosts:
                route.abort("blockedbyclient")
                return
            route.continue_()
        self.context.route("**/*", route_request)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

    def set_allowed_hosts(self, hosts):
        self.allowed_hosts = set(hosts)

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
                 personal_profile=False, personal_opt_in=False):
        if personal_profile and not personal_opt_in:
            raise PermissionError("personal browser profiles require explicit opt-in")
        self.profile = Path(profile or (DATA_ROOT / "browser" / "profile")).expanduser().resolve()
        self.downloads = Path(downloads or (DATA_ROOT / "browser" / "downloads")).expanduser().resolve()
        self.profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.profile.chmod(0o700)
        self.downloads.chmod(0o700)
        self.runtime = runtime or ToolRuntime()
        self.driver = driver
        self._injected_driver = driver is not None
        self._live_driver_started = False
        self.personal_profile = personal_profile

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
        normalized, host = normalize_network_target(url, resolve_dns=True)
        destinations = tuple(allowed_hosts) or (host,)
        request = ScopeRequest(
            "browser.navigate", "network", normalized, {"profile": "dedicated-tars"},
            task_id=task_id, session_id=session_id, allowed_hosts=destinations,
        )
        actions = self.runtime.authorize((("network", request),), {"network": approval_id})
        try:
            driver = self._driver()
            if hasattr(driver, "set_allowed_hosts"):
                driver.set_allowed_hosts(destinations)
            data = driver.call("navigate", url=normalized)
            _, final_host = normalize_network_target(data.get("url", normalized), resolve_dns=True)
            if final_host not in {
                normalize_network_target(value)[1] for value in destinations
            }:
                raise PermissionError("browser navigation left the authorized destination set")
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("browser", normalized, repr(data), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("browser.navigate", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def action(self, operation, *, approval_id=None, task_id=None, session_id=None, **kwargs):
        read_operations = {"snapshot", "tabs", "wait"}
        if operation == "evaluate":
            effect, elevated = "elevated", True
        else:
            effect, elevated = ("read", False) if operation in read_operations else ("write", False)
        if operation == "download":
            kwargs = dict(kwargs) | {"directory": str(self.downloads)}
        if operation == "screenshot" and "path" in kwargs:
            requested = (self.downloads / Path(kwargs["path"]).name).resolve()
            kwargs = dict(kwargs) | {"path": str(requested)}
        request = ScopeRequest(
            f"browser.{operation}", effect, "browser-session", kwargs,
            task_id=task_id, session_id=session_id, elevated=elevated,
        )
        actions = self.runtime.authorize((("action", request),), {"action": approval_id})
        try:
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
