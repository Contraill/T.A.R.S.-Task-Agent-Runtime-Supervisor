from __future__ import annotations

import json
import os

from .http_tools import HTTPTools
from .tool_core import ToolResult


class TavilyResearch:
    identity = "tavily"
    support = "mock-tested"

    ENDPOINTS = {
        "search": "https://api.tavily.com/search",
        "extract": "https://api.tavily.com/extract",
        "crawl": "https://api.tavily.com/crawl",
    }

    def __init__(self, *, secret_ref="env:TAVILY_API_KEY", http=None):
        self.secret_ref = secret_ref
        self.http = http or HTTPTools()

    def status(self):
        available = self.secret_ref.startswith("env:") and self.secret_ref[4:] in os.environ
        return {"backend": self.identity, "available": available, "support": self.support,
                "secret_ref": self.secret_ref,
                "message": "available" if available else "Tavily credential unavailable"}

    def _call(self, capability, payload, *, approval_ids=None, task_id=None, session_id=None):
        if capability not in self.ENDPOINTS:
            raise ValueError(f"unknown Tavily capability: {capability}")
        status = self.status()
        if not status["available"]:
            return ToolResult(f"web.{capability}", "unavailable", status,
                              error=status["message"])
        api_key = os.environ[self.secret_ref[4:]]
        body = json.dumps({"api_key": api_key, **payload}).encode()
        response = self.http.request(
            "POST", self.ENDPOINTS[capability],
            headers={"Content-Type": "application/json"}, body=body,
            approval_ids=approval_ids, task_id=task_id, session_id=session_id,
            sensitive_values=(api_key,),
        )
        if not response.succeeded:
            return ToolResult(f"web.{capability}", "failed", response.data,
                              error=response.error, action_ids=response.action_ids,
                              evidence_ids=response.evidence_ids)
        try:
            data = json.loads(response.data["content"])
        except json.JSONDecodeError as exc:
            return ToolResult(f"web.{capability}", "failed", response.data,
                              error=f"invalid Tavily response: {exc}",
                              action_ids=response.action_ids,
                              evidence_ids=response.evidence_ids)
        return ToolResult(f"web.{capability}", "succeeded", data,
                          action_ids=response.action_ids, evidence_ids=response.evidence_ids)

    def search(self, query, *, max_results=10, **kwargs):
        return self._call("search", {"query": query, "max_results": int(max_results)}, **kwargs)

    def extract(self, urls, **kwargs):
        return self._call("extract", {"urls": list(urls)}, **kwargs)

    def crawl(self, url, *, max_depth=2, limit=20, **kwargs):
        return self._call("crawl", {"url": url, "max_depth": int(max_depth),
                                    "limit": int(limit)}, **kwargs)
