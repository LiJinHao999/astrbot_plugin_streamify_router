import datetime
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

# --------------- handler 自注册表 ---------------
_handler_registry: list = []


def register_handler(cls):
    """装饰器：将 handler 类注册到全局列表，供 ProviderRoute 自动发现。"""
    _handler_registry.append(cls)
    return cls


def get_handler_classes() -> list:
    """返回所有已注册的 handler 类（按注册顺序）。"""
    return list(_handler_registry)


# 匹配长 base64/base32 字符串（≥64 个字符）
_LONG_BASE64_RE = re.compile(r'^[A-Za-z0-9+/\-_]{64,}={0,3}$')

_PLUGIN_NAME = "astrbot_plugin_streamify_router"

_FC_FAILURE_MSG = "工具调用失败，请如实告诉用户情况"


def _resolve_plugin_data_dir() -> pathlib.Path:
    return pathlib.Path(get_astrbot_data_path()) / "plugin_data" / _PLUGIN_NAME


def _get_debug_log_dir() -> pathlib.Path:
    """返回 debug_log 目录路径（惰性创建）。"""
    d = _resolve_plugin_data_dir() / "debug_log"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_debug_entry(entry: dict) -> None:
    """追加 JSON 行到当天 debug 日志文件。自动对 body 字段调用 _sanitize_for_log 脱敏。"""
    entry = dict(entry)
    if "body" in entry and isinstance(entry["body"], (dict, list)):
        entry["body"] = _sanitize_for_log(entry["body"])
    if "ts" not in entry:
        entry["ts"] = datetime.datetime.now().isoformat()
    try:
        log_dir = _get_debug_log_dir()
        today = datetime.date.today().isoformat()
        log_file = log_dir / f"debug_{today}.log"
        line = json.dumps(entry, ensure_ascii=False)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("写入 debug 日志失败: %s", exc)


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
        fc_enhance: int = 1,
        fix_retries: int = 1,
        tool_error_patterns: Optional[List[str]] = None,
        fc_context_turns: int = 1,
    ):
        self.target = target_url.rstrip("/")
        self.proxy = proxy_url.strip() or None
        self.session = session
        self.debug = bool(debug)
        self.request_timeout = self._normalize_timeout(request_timeout)
        self.pseudo_non_stream = bool(pseudo_non_stream)
        self.fc_enhance = max(0, min(2, int(fc_enhance)))
        self.fix_retries = max(0, int(fix_retries))
        self.fc_context_turns = max(0, int(fc_context_turns))
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

    def _log_fc_modify(self, route: str, layer: int, tool_name: str, before_args: Any, after_args: Any, *, hint: str = "", context: Any = None) -> None:
        """记录 FC 修复前后的参数对比。始终输出简要日志，debug 模式下额外写入文件。"""

        def _to_str(v: Any) -> str:
            if isinstance(v, str):
                return v
            return json.dumps(v, ensure_ascii=False)

        before_str = _to_str(before_args)
        after_str = _to_str(after_args)

        # 始终输出到控制台
        logger.info(
            "Streamify [Layer%d] %s: 工具 %s 参数重写 %s -> %s",
            layer, route, tool_name, before_str, after_str,
        )

        if not self.debug:
            return

        entry: Dict[str, Any] = {
            "type": "fc_modify",
            "route": route,
            "layer": layer,
            "tool": tool_name,
            "before": before_str,
            "after": after_str,
        }
        if hint:
            entry["hint"] = hint
        if context is not None:
            entry["context"] = _sanitize_for_log(context)
        _write_debug_entry(entry)

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

            stream_chunks: Optional[List[bytes]] = [] if self.debug else None
            stream_start = time.perf_counter()

            async for chunk in resp.content.iter_any():
                await response.write(chunk)
                if stream_chunks is not None:
                    stream_chunks.append(chunk)

            await response.write_eof()

            if stream_chunks is not None:
                self._log_stream_to_file(req, sub_path, resp.status, stream_start, stream_chunks)

            return response

    def _log_stream_to_file(
        self,
        req: web.Request,
        sub_path: str,
        status: int,
        start_time: float,
        chunks: List[bytes],
    ) -> None:
        """将缓冲的流式响应内容脱敏后写入 debug 日志文件。"""
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        route = req.match_info.get("route_name", "")
        path_text = f"/{route}/{sub_path.strip('/')}" if route else sub_path

        raw_text = b"".join(chunks).decode("utf-8", errors="replace")
        sanitized_lines: List[str] = []
        for line in raw_text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    obj = json.loads(payload)
                    sanitized = _sanitize_for_log(obj)
                    sanitized_lines.append(f"data: {json.dumps(sanitized, ensure_ascii=False)}")
                    continue
                except Exception:
                    pass
            sanitized_lines.append(line)

        _write_debug_entry({
            "type": "stream_response",
            "route": route,
            "method": req.method.upper(),
            "path": path_text,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "body": "\n".join(sanitized_lines),
        })

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
