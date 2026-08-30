from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    capability: str
    effect: str
    native: bool = True
    available: bool = True
    support: str = "tested"


NATIVE_TOOLS = (
    *(ToolDescriptor(f"fs.{name}", f"filesystem.{name}", effect)
      for name, effect in (
          ("list", "read"), ("stat", "read"), ("read", "read"), ("search", "read"),
          ("mkdir", "write"), ("copy", "write"), ("move", "write"),
          ("patch", "write"), ("write", "write"), ("delete", "destructive"),
      )),
    ToolDescriptor("terminal.run", "process.execute", "execute"),
    *(ToolDescriptor(f"process.{name}", f"process.{name}", effect)
      for name, effect in (("list", "read"), ("poll", "read"), ("wait", "read"),
                           ("logs", "read"), ("write", "execute"),
                           ("signal", "execute"), ("kill", "destructive"))),
    *(ToolDescriptor(f"git.{name}", f"git.{name}", effect)
      for name, effect in (("status", "read"), ("diff", "read"), ("log", "read"),
                           ("show", "read"), ("branch", "write"), ("switch", "write"),
                           ("commit", "write"), ("checkpoint", "read"),
                           ("rollback", "destructive"), ("push", "network"))),
    *(ToolDescriptor(f"service.{name}", f"service.{name}", effect)
      for name, effect in (("status", "read"), ("start", "service"),
                           ("stop", "service"), ("restart", "service"), ("logs", "read"))),
    *(ToolDescriptor(f"package.{name}", f"package.{name}", effect)
      for name, effect in (("search", "read"), ("info", "read"),
                           ("installed", "read"), ("install", "elevated"),
                           ("remove", "elevated"), ("upgrade", "elevated"),
                           ("orphans", "read"))),
    *(ToolDescriptor(f"system.{name}", f"system.{name}", "read")
      for name in ("info", "processes", "storage", "network", "hardware", "logs")),
    ToolDescriptor("http.request", "http.request", "network"),
    ToolDescriptor("http.download", "http.download", "write"),
    ToolDescriptor("web.search", "web.search", "network", support="optional"),
    ToolDescriptor("web.extract", "web.extract", "network", support="optional"),
    ToolDescriptor("web.crawl", "web.crawl", "network", support="optional"),
    *(ToolDescriptor(f"browser.{name}", f"browser.{name}", effect, support="optional")
      for name, effect in (("navigate", "network"), ("tabs", "read"),
                           ("snapshot", "read"), ("click", "write"), ("type", "write"),
                           ("select", "write"), ("key", "write"), ("scroll", "write"),
                           ("wait", "read"), ("screenshot", "write"),
                           ("download", "write"), ("back", "write"),
                           ("reload", "network"), ("close", "write"))),
    *(ToolDescriptor(f"archive.{name}", f"archive.{name}", effect)
      for name, effect in (("list", "read"), ("extract", "write"), ("create", "write"))),
    ToolDescriptor("artifact.hash", "artifact.hash", "read"),
    ToolDescriptor("artifact.verify", "artifact.verify", "read"),
    *(ToolDescriptor(f"pdf.{name}", f"pdf.{name}", effect, support="capability-reported")
      for name, effect in (("info", "read"), ("text", "read"), ("search", "read"),
                           ("render", "write"), ("merge", "write"), ("split", "write"),
                           ("reorder", "write"), ("rotate", "write"),
                           ("delete_pages", "write"), ("annotation", "write"),
                           ("redaction", "destructive"), ("form_fill", "write"),
                           ("export", "write"))),
    *(ToolDescriptor(f"document.{name}", f"document.{name}", effect,
                     support="capability-reported")
      for name, effect in (("inspect", "read"), ("extract", "read"),
                           ("convert", "write"), ("edit", "write"))),
    *(ToolDescriptor(f"spreadsheet.{name}", f"spreadsheet.{name}", effect,
                     support="capability-reported")
      for name, effect in (("inspect", "read"), ("read_range", "read"),
                           ("write_range", "write"), ("add_sheet", "write"),
                           ("formulas", "write"), ("export", "write"))),
    *(ToolDescriptor(f"image.{name}", f"image.{name}", effect,
                     support="capability-reported")
      for name, effect in (("info", "read"), ("resize", "write"), ("crop", "write"),
                           ("rotate", "write"), ("convert", "write"),
                           ("compress", "write"))),
    ToolDescriptor("notify.send", "desktop.notify", "write", support="optional"),
    ToolDescriptor("screen.capture", "desktop.capture", "write", support="optional"),
)


class ToolRegistry:
    def __init__(self, descriptors=NATIVE_TOOLS):
        self._tools = {tool.name: tool for tool in descriptors}

    def list(self):
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    def get(self, name):
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown native tool: {name}") from exc

    def select(self, capability, *, mcp_tools=(), terminal_fallback=True):
        native = next((tool for tool in self.list()
                       if tool.capability == capability and tool.available), None)
        if native:
            return {"kind": "native", "tool": native.name}
        mcp = next((tool for tool in mcp_tools if tool.get("capability") == capability
                    and tool.get("trusted")), None)
        if mcp:
            return {"kind": "mcp", "tool": mcp["name"]}
        if terminal_fallback:
            return {"kind": "terminal", "tool": "terminal.run"}
        return None
