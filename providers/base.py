import json
import pathlib
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Pattern, Tuple

import aiohttp
from aiohttp import ClientResponse, ClientSession, web

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from ..fc_enhance import _DEFAULT_TOOL_ERROR_PATTERNS, _compile_error_patterns

# 匹配长 base64/base32 字符串（≥64 个字符）
_LONG_BASE64_RE = re.compile(r'^[A-Za-z0-9+/\-_]{64,}={0,3}$')

_PLUGIN_NAME = "astrbot_plugin_streamify_router"

_FC_FAILURE_MSG = "工具调用失败，请如实告诉用户情况"


def _resolve_plugin_data_dir() -> pathlib.Path:
    return pathlib.Path(get_astrbot_data_path()) / "plugin_data" / _PLUGIN_NAME


def _sanitize_for_log(obj: Any, _depth: int = 0) -> Any:
    """递归将请求体中的长 base64/data-URI 字符串替换为占位符，避免刷屏。"""
    if _depth > 20:
        return "..."
    if isinstance(obj, dict):
        return {k: _sanitize_for_log(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_log(item, _depth + 1) for item in obj]
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return f"<data_uri {len(obj)} chars>"
        if len(obj) > 100 and _LONG_BASE64_RE.match(obj):
            return f"<base64 {len(obj)} chars>"
    return obj


class ProviderHandler:
    """Base handler for forwarding and stream utilities."""

    def __init__(
        self,
        target_url: str,
        proxy_url: str = "",
        session: Optional[ClientSession] = None,
        debug: bool = False,
        request_timeout: float = 120.0,
        pseudo_non_stream: bool = True,
        extract_args: bool = False,
        fix_retries: int = 1,
        tool_error_patterns: Optional[List[str]] = None,
    ):
        self.target = target_url.rstrip("/")
        self.proxy = proxy_url.strip() or None
        self.session = session
        self.debug = bool(debug)
        self.request_timeout = self._normalize_timeout(request_timeout)
        self.pseudo_non_stream = bool(pseudo_non_stream)
        self.extract_args = bool(extract_args)
        self.fix_retries = max(0, int(fix_retries))
        patterns = tool_error_patterns if tool_error_patterns is not None else _DEFAULT_TOOL_ERROR_PATTERNS
        self._error_patterns: List[Pattern[str]] = _compile_error_patterns(patterns)
        plugin_data_dir = _resolve_plugin_data_dir()
        plugin_data_dir.mkdir(parents=True, exist_ok=True)
        self._hint_tools_path = plugin_data_dir / f"hint_tools_{self.__class__.__name__.lower()}.json"
        try:
            self._hint_tools: set = set(
                json.loads(self._hint_tools_path.read_text("utf-8"))
            )
        except Exception:
            self._hint_tools: set = set()

    def _remember_hint_tool(self, tool_name: str) -> None:
        self._hint_tools.add(tool_name)
        try:
            self._hint_tools_path.write_text(
                json.dumps(sorted(self._hint_tools), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save hint tools: %s", exc)

    @staticmethod
    def _normalize_timeout(value: Any, default: float = 120.0) -> float:
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return default
        if timeout <= 0:
            return default
        return timeout

    def _get_session(self) -> ClientSession:
        if self.session is None or self.session.closed:
            raise RuntimeError("Shared aiohttp ClientSession is not available.")
        return self.session

    def _request(self, method: str, url: str, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = aiohttp.ClientTimeout(
                total=None,
                connect=self.request_timeout,
                sock_connect=self.request_timeout,
                sock_read=self.request_timeout,
            )
        return self._get_session().request(method, url, proxy=self.proxy, **kwargs)

    def matches(self, sub_path: str) -> bool:
        return False

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        return await self._passthrough(req, sub_path)

    def _build_url(self, sub_path: str) -> str:
        path = sub_path.lstrip("/")
        if not path:
            return self.target
        return f"{self.target}/{path}"

    def _forward_headers(self, req: web.Request) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key, value in req.headers.items():
            lower = key.lower()
            if lower in {"host", "content-length", "transfer-encoding", "connection"}:
                continue
            headers[key] = value
        return headers

    def _response_headers(self, resp: ClientResponse) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key, value in resp.headers.items():
            lower = key.lower()
            if lower in {"content-length", "transfer-encoding", "connection"}:
                continue
            headers[key] = value
        return headers

    async def _read_json(self, req: web.Request) -> Optional[Dict[str, Any]]:
        try:
            data = await req.json()
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None

    async def _passthrough(self, req: web.Request, sub_path: str, _body: Optional[bytes] = None) -> web.Response:
        url = self._build_url(sub_path)
        headers = self._forward_headers(req)
        data = _body if _body is not None else await req.read()
        async with self._request(
            req.method,
            url,
            headers=headers,
            params=req.query,
            data=data,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                status=resp.status,
                headers=self._response_headers(resp),
                body=resp_body,
            )

    async def _proxy_stream(
        self,
        req: web.Request,
        sub_path: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        params: Optional[Dict[str, str]] = None,
    ) -> web.Response:
        url = self._build_url(sub_path)
        async with self._request(
            req.method,
            url,
            json=body,
            headers=headers,
            params=params if params is not None else req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )

            stream_headers = {
                "Content-Type": resp.headers.get("Content-Type", "text/event-stream"),
                "Cache-Control": "no-cache",
            }
            response = web.StreamResponse(status=resp.status, headers=stream_headers)
            await response.prepare(req)

            async for chunk in resp.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            return response

    async def _iter_sse_events(
        self, resp: ClientResponse
    ) -> AsyncIterator[Tuple[Optional[str], str]]:
        event_name: Optional[str] = None
        data_lines: List[str] = []

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

            if line == "":
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = None
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            yield event_name, "\n".join(data_lines)
