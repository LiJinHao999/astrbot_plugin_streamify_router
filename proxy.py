import json
import pathlib
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Pattern, Tuple

import aiohttp
from aiohttp import ClientResponse, ClientSession, web

from astrbot.api import logger

from .fake_non_stream import OpenAIFakeNonStream, ClaudeFakeNonStream, GeminiFakeNonStream
from .fc_enhance import (
    OpenAIFCEnhance, ClaudeFCEnhance, GeminiFCEnhance,
    _DEFAULT_TOOL_ERROR_PATTERNS, _compile_error_patterns,
)

# 匹配长 base64/base32 字符串（≥64 个字符）
_LONG_BASE64_RE = re.compile(r'^[A-Za-z0-9+/\-_]{64,}={0,3}$')


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


class OpenAIChatHandler(ProviderHandler, OpenAIFakeNonStream, OpenAIFCEnhance):
    ENDPOINT = "v1/chat/completions"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)

        # Stream requests always pass through directly
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

        clean_body = body

        # Layer 2 (Reactive): 检测传入请求中是否已有工具执行错误
        if self.extract_args:
            error_info = self._find_tool_error_in_request(body)
            if error_info is not None:
                tool_name, tool_call_id, assistant_msg_idx = error_info
                ctx_messages = body["messages"][:assistant_msg_idx]
                extracted = await self._extract_args_as_json(
                    body, tool_name, sub_path, headers,
                    messages_override=ctx_messages,
                )
                if extracted is not None:
                    if self.debug:
                        logger.info(
                            "Streamify [Layer2]: 修正 OpenAI 工具 %s 的参数: %s",
                            tool_name, extracted,
                        )
                    return web.json_response(
                        self._build_corrected_tool_response(
                            tool_call_id, tool_name, extracted, body.get("model", "")
                        )
                    )
                elif self.debug:
                    logger.info(
                        "Streamify [Layer2]: OpenAI 工具 %s 参数提取失败，继续正常转发",
                        tool_name,
                    )
                # 无论提取是否成功，只要检测到错误就清理 body
                clean_body = {**body, "messages": ctx_messages}

        # 假非流禁用时直通
        if not self.pseudo_non_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        clean_body["stream"] = True
        async with self._request(
            "POST",
            self._build_url(sub_path),
            json=clean_body,
            headers=headers,
            params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )
            result = await self._build_non_stream_response(resp)

        # 提前检查：无失败 FC 则直接返回
        failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
        if failed_tc is None:
            return web.json_response(result)

        # Layer 1 (Proactive): 对照 schema 检测响应中是否缺少必填参数
        if self.extract_args:
            function_name = (failed_tc.get("function") or {}).get("name", "")
            extracted = await self._extract_args_as_json(
                clean_body, function_name, sub_path, headers
            )
            if extracted is not None:
                self._patch_function_call_args(result, function_name, extracted)
                if self.debug:
                    logger.info(
                        "Streamify [Layer1]: 成功提取 OpenAI 工具 %s 的参数: %s",
                        function_name, extracted,
                    )
                return web.json_response(result)
            elif self.debug:
                logger.info("Streamify [Layer1]: OpenAI JSON 参数提取失败，尝试提示注入重试")

        # 提示注入重试
        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info(
                    "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                    attempt + 1, self.fix_retries,
                )
            current_body = self._inject_hint(clean_body)
            current_body["stream"] = True
            async with self._request(
                "POST",
                self._build_url(sub_path),
                json=current_body,
                headers=headers,
                params=req.query,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )
                result = await self._build_non_stream_response(resp)
            failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
            if failed_tc is None:
                return web.json_response(result)
            if self.extract_args:
                function_name = (failed_tc.get("function") or {}).get("name", "")
                extracted = await self._extract_args_as_json(
                    clean_body, function_name, sub_path, headers
                )
                if extracted is not None:
                    self._patch_function_call_args(result, function_name, extracted)
                    if self.debug:
                        logger.info(
                            "Streamify [Layer1]: 成功提取 OpenAI 工具 %s 的参数: %s",
                            function_name, extracted,
                        )
                    return web.json_response(result)

        return web.json_response(result)  # 重试仍失败，返回原始结果（不报 500）


class ClaudeHandler(ProviderHandler, ClaudeFakeNonStream, ClaudeFCEnhance):
    ENDPOINT = "v1/messages"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)

        # Stream requests always pass through directly
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

        clean_body = body

        # Layer 2 (Reactive): 检测传入请求中是否已有工具执行错误
        if self.extract_args:
            error_info = self._find_tool_error_in_request(body)
            if error_info is not None:
                tool_name, tool_use_id, assistant_msg_idx = error_info
                ctx_messages = body["messages"][:assistant_msg_idx]
                extracted = await self._extract_args_as_json(
                    body, tool_name, sub_path, headers,
                    messages_override=ctx_messages,
                )
                if extracted is not None:
                    if self.debug:
                        logger.info(
                            "Streamify [Layer2]: 修正 Claude 工具 %s 的参数: %s",
                            tool_name, extracted,
                        )
                    return web.json_response(
                        self._build_corrected_tool_response(
                            tool_use_id, tool_name, extracted, body.get("model", "")
                        )
                    )
                elif self.debug:
                    logger.info(
                        "Streamify [Layer2]: Claude 工具 %s 参数提取失败，继续正常转发",
                        tool_name,
                    )
                # 无论提取是否成功，只要检测到错误就清理 body
                clean_body = {**body, "messages": ctx_messages}

        # 假非流禁用时直通
        if not self.pseudo_non_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        clean_body["stream"] = True
        async with self._request(
            "POST",
            self._build_url(sub_path),
            json=clean_body,
            headers=headers,
            params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )
            result = await self._build_non_stream_response(resp)

        # 提前检查：无失败 FC 则直接返回
        failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
        if failed_tc is None:
            return web.json_response(result)

        # Layer 1 (Proactive): 对照 schema 检测响应中是否缺少必填参数
        if self.extract_args:
            function_name = failed_tc.get("name", "")
            extracted = await self._extract_args_as_json(
                clean_body, function_name, sub_path, headers
            )
            if extracted is not None:
                self._patch_function_call_args(result, function_name, extracted)
                if self.debug:
                    logger.info(
                        "Streamify [Layer1]: 成功提取 Claude 工具 %s 的参数: %s",
                        function_name, extracted,
                    )
                return web.json_response(result)
            elif self.debug:
                logger.info("Streamify [Layer1]: Claude JSON 参数提取失败，尝试提示注入重试")

        # 提示注入重试
        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info(
                    "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                    attempt + 1, self.fix_retries,
                )
            current_body = self._inject_hint(clean_body)
            current_body["stream"] = True
            async with self._request(
                "POST",
                self._build_url(sub_path),
                json=current_body,
                headers=headers,
                params=req.query,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )
                result = await self._build_non_stream_response(resp)
            failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
            if failed_tc is None:
                return web.json_response(result)
            if self.extract_args:
                function_name = failed_tc.get("name", "")
                extracted = await self._extract_args_as_json(
                    clean_body, function_name, sub_path, headers
                )
                if extracted is not None:
                    self._patch_function_call_args(result, function_name, extracted)
                    if self.debug:
                        logger.info(
                            "Streamify [Layer1]: 成功提取 Claude 工具 %s 的参数: %s",
                            function_name, extracted,
                        )
                    return web.json_response(result)

        return web.json_response(result)  # 重试仍失败，返回原始结果（不报 500）


class GeminiHandler(ProviderHandler, GeminiFakeNonStream, GeminiFCEnhance):
    def matches(self, sub_path: str) -> bool:
        path = sub_path.strip("/")
        return ":generateContent" in path or ":streamGenerateContent" in path

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        path = sub_path.strip("/")
        if ":streamGenerateContent" in path:
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        stream_path = path.replace(":generateContent", ":streamGenerateContent", 1)
        headers = self._forward_headers(req)
        params = {k: v for k, v in req.query.items()}
        params["alt"] = "sse"

        clean_body = body

        # Layer 2 (Reactive): 检测传入请求中是否已有工具执行错误
        if self.extract_args:
            error_info = self._find_tool_error_in_request(body)
            if error_info is not None:
                tool_name, error_idx = error_info
                ctx_contents = body["contents"][:max(0, error_idx - 1)]
                extracted = await self._extract_args_as_json(
                    body, tool_name, stream_path, headers, params,
                    contents_override=ctx_contents,
                )
                if extracted is not None:
                    if self.debug:
                        logger.info(
                            "Streamify [Layer2]: 修正 Gemini 工具 %s 的参数: %s",
                            tool_name, extracted,
                        )
                    return web.json_response(
                        self._build_corrected_tool_response(tool_name, extracted)
                    )
                elif self.debug:
                    logger.info(
                        "Streamify [Layer2]: Gemini 工具 %s 参数提取失败，继续正常转发",
                        tool_name,
                    )
                # 无论提取是否成功，只要检测到错误就清理 body
                clean_body = {**body, "contents": ctx_contents}

        tools = clean_body.get("tools", [])

        # 假非流禁用时直通
        if not self.pseudo_non_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        async with self._request(
            "POST",
            self._build_url(stream_path),
            json=clean_body,
            headers=headers,
            params=params,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )
            response_data = await self._build_non_stream_response(resp)

        if not self._has_failed_function_call(response_data, tools):
            return web.json_response(response_data)

        # Layer 1 (Proactive): 对照 schema 提取参数
        if self.extract_args:
            failed_fc = self._find_failed_function_call(response_data, tools)
            if failed_fc is not None:
                extracted = await self._extract_args_as_json(
                    clean_body, failed_fc.get("name", ""), stream_path, headers, params
                )
                if extracted is not None:
                    self._patch_function_call_args(
                        response_data, failed_fc.get("name", ""), extracted
                    )
                    if self.debug:
                        logger.info(
                            "Streamify [Layer1]: 成功提取 Gemini 工具 %s 的参数: %s",
                            failed_fc.get("name"), extracted,
                        )
                    return web.json_response(response_data)
                elif self.debug:
                    logger.info("Streamify [Layer1]: Gemini JSON 参数提取失败，尝试提示注入重试")

        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info(
                    "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                    attempt + 1, self.fix_retries,
                )
            current_body = self._inject_hint(clean_body)
            async with self._request(
                "POST",
                self._build_url(stream_path),
                json=current_body,
                headers=headers,
                params=params,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )
                response_data = await self._build_non_stream_response(resp)

            if not self._has_failed_function_call(response_data, tools):
                return web.json_response(response_data)

        failed_fc = self._find_failed_function_call(response_data, tools)
        function_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
        logger.warning(
            "Streamify: 工具 %s 参数在 %d 次重试后仍为空，放弃调用",
            function_name, self.fix_retries,
        )
        return web.json_response(
            {
                "error": {
                    "code": 500,
                    "message": (
                        f"Gemini 未能为工具 `{function_name}` 生成有效参数，"
                        f"已重试 {self.fix_retries} 次仍失败。"
                    ),
                    "status": "FUNCTION_CALL_ARGS_EMPTY",
                }
            },
            status=500,
        )


class OpenAIResponsesHandler(ProviderHandler):
    ENDPOINT = "v1/responses"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

        # 假非流禁用时直通
        if not self.pseudo_non_stream:
            return await self._passthrough(req, sub_path)

        body["stream"] = True
        async with self._request(
            "POST",
            self._build_url(sub_path),
            json=body,
            headers=headers,
            params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )

            completed: Optional[Dict[str, Any]] = None
            fallback: Optional[Dict[str, Any]] = None

            async for event_name, data in self._iter_sse_events(resp):
                if not data:
                    continue
                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event_name or payload.get("type")
                if event_type == "response.completed":
                    response_obj = payload.get("response")
                    completed = response_obj if isinstance(response_obj, dict) else payload
                elif event_type == "response.created":
                    response_obj = payload.get("response")
                    if isinstance(response_obj, dict):
                        fallback = response_obj
                    elif isinstance(payload, dict):
                        fallback = payload

            if completed is None:
                completed = fallback
            if completed is None:
                return web.json_response(
                    {
                        "error": {
                            "message": "No completed response event received from upstream.",
                            "type": "proxy_error",
                        }
                    },
                    status=502,
                )

            return web.json_response(completed)


class ProviderRoute:
    def __init__(
        self,
        route_name: str,
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
        self.route_name = route_name
        self.target_url = target_url.rstrip("/")
        self.proxy_url = proxy_url.strip()
        self.debug = bool(debug)
        self.base = ProviderHandler(
            self.target_url,
            self.proxy_url,
            session=session,
            debug=self.debug,
            request_timeout=request_timeout,
        )
        self.handlers: List[ProviderHandler] = [
            OpenAIChatHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
                pseudo_non_stream=pseudo_non_stream,
                extract_args=extract_args,
                fix_retries=fix_retries,
                tool_error_patterns=tool_error_patterns,
            ),
            ClaudeHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
                pseudo_non_stream=pseudo_non_stream,
                extract_args=extract_args,
                fix_retries=fix_retries,
                tool_error_patterns=tool_error_patterns,
            ),
            GeminiHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
                pseudo_non_stream=pseudo_non_stream,
                extract_args=extract_args,
                fix_retries=fix_retries,
                tool_error_patterns=tool_error_patterns,
            ),
            OpenAIResponsesHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
                pseudo_non_stream=pseudo_non_stream,
            ),
        ]

    async def dispatch(self, req: web.Request, sub_path: str) -> web.Response:
        path = sub_path.strip("/")
        for handler in self.handlers:
            if handler.matches(path):
                return await handler.handle(req, path)
        return await self.base.handle(req, path)


class StreamifyProxy:
    def __init__(
        self,
        port: int = 23456,
        providers: Optional[List[Dict[str, Any]]] = None,
        debug: bool = False,
        request_timeout: float = 120.0,
        pseudo_non_stream: bool = True,
        extract_args: bool = False,
        fix_retries: int = 1,
        tool_error_patterns: Optional[List[str]] = None,
    ):
        self.port = port
        self.providers_config = providers or []
        self.debug = bool(debug)
        self.request_timeout = ProviderHandler._normalize_timeout(request_timeout)
        self.pseudo_non_stream = bool(pseudo_non_stream)
        self.extract_args = bool(extract_args)
        self.fix_retries = max(0, int(fix_retries))
        self.tool_error_patterns: Optional[List[str]] = tool_error_patterns
        self.providers: Dict[str, ProviderRoute] = {}
        self.session: Optional[ClientSession] = None
        self._debug_log_path = pathlib.Path(__file__).parent / "debug_requests.log"
        self.app = web.Application(client_max_size=20 * 1024 * 1024)
        self.app.add_routes(
            [
                web.route("*", "/{route_name}", self._dispatch),
                web.route("*", "/{route_name}/{sub_path:.*}", self._dispatch),
            ]
        )
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

    def _load_providers(self) -> None:
        if self.session is None or self.session.closed:
            raise RuntimeError("Proxy session not initialized. Call start() first.")

        self.providers = {}
        for item in self.providers_config:
            if not isinstance(item, dict):
                continue

            route_name = str(item.get("route_name", "")).strip().strip("/")
            target_url = str(item.get("target_url", "")).strip()
            proxy_url = str(item.get("proxy_url", "")).strip()

            if not route_name or not target_url:
                logger.warning(
                    "Skip invalid provider config: route_name=%r target_url=%r",
                    route_name,
                    target_url,
                )
                continue

            if route_name in self.providers:
                logger.warning("Duplicate route_name %s, overwrite previous target.", route_name)

            # 路由级 pseudo_non_stream 优先；未配置则继承全局值
            route_pseudo = item.get("pseudo_non_stream")
            if isinstance(route_pseudo, bool):
                pseudo_non_stream = route_pseudo
            else:
                pseudo_non_stream = self.pseudo_non_stream

            self.providers[route_name] = ProviderRoute(
                route_name,
                target_url,
                proxy_url,
                session=self.session,
                debug=self.debug,
                request_timeout=self.request_timeout,
                pseudo_non_stream=pseudo_non_stream,
                extract_args=self.extract_args,
                fix_retries=self.fix_retries,
                tool_error_patterns=self.tool_error_patterns,
            )

    async def _dispatch(self, req: web.Request) -> web.Response:
        route_name = req.match_info.get("route_name", "").strip("/")
        sub_path = req.match_info.get("sub_path", "")
        start_time = time.perf_counter()

        provider = self.providers.get(route_name)
        if provider is None:
            if self.debug:
                path_text = f"/{route_name}"
                if sub_path:
                    path_text = f"{path_text}/{sub_path.strip('/')}"
                logger.info("Streamify debug route miss: %s %s", req.method.upper(), path_text)
            return web.json_response(
                {
                    "error": {
                        "message": f"Unknown route_name '{route_name}'.",
                        "type": "route_not_found",
                    },
                    "available_routes": sorted(self.providers.keys()),
                },
                status=404,
            )

        path_text = f"/{route_name}"
        if sub_path:
            path_text = f"{path_text}/{sub_path.strip('/')}"

        if self.debug:
            try:
                if req.method.upper() in {"POST", "PUT", "PATCH"}:
                    raw = await req.read()
                    if raw:
                        try:
                            body_obj = json.loads(raw)
                            sanitized = _sanitize_for_log(body_obj)
                            detail = json.dumps(sanitized, ensure_ascii=False)
                        except Exception:
                            detail = f"<{len(raw)} bytes, non-JSON>"
                    else:
                        detail = "<empty body>"
                else:
                    qs = dict(req.query)
                    detail = json.dumps(qs, ensure_ascii=False) if qs else "<no query>"
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self._debug_log_path, "a", encoding="utf-8") as _f:
                    _f.write(f"[{ts}] >>> {req.method.upper()} {path_text} -> {provider.target_url}\n")
                    _f.write(detail + "\n\n")
            except Exception as _exc:
                logger.warning("Streamify debug: 写入请求日志失败: %s", _exc)

        try:
            response = await provider.dispatch(req, sub_path)
            if self.debug:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "Streamify debug handled: %s %s -> %s status=%s elapsed=%dms",
                    req.method.upper(),
                    path_text,
                    provider.target_url,
                    response.status,
                    elapsed_ms,
                )
                try:
                    if isinstance(response, web.Response) and response.body:
                        content_type = response.content_type or ""
                        if "json" in content_type or "text" in content_type:
                            body_text = response.body.decode("utf-8", errors="replace")
                            try:
                                resp_obj = json.loads(body_text)
                                sanitized_resp = _sanitize_for_log(resp_obj)
                                resp_detail = json.dumps(sanitized_resp, ensure_ascii=False)
                            except Exception:
                                resp_detail = body_text[:2000]
                        else:
                            resp_detail = f"<{len(response.body)} bytes, {content_type}>"
                    else:
                        resp_detail = "<streaming response>"
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(self._debug_log_path, "a", encoding="utf-8") as _f:
                        _f.write(
                            f"[{ts}] <<< {req.method.upper()} {path_text} "
                            f"status={response.status} elapsed={elapsed_ms}ms\n"
                        )
                        _f.write(resp_detail + "\n\n")
                except Exception as _exc:
                    logger.warning("Streamify debug: 写入响应日志失败: %s", _exc)
            return response
        except Exception as exc:
            logger.exception("Proxy dispatch failed for route %s: %s", route_name, exc)
            return web.json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "proxy_internal_error",
                    }
                },
                status=500,
            )

    def get_route_infos(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for route_name in sorted(self.providers.keys()):
            provider = self.providers[route_name]
            infos.append(
                {
                    "route_name": route_name,
                    "target_url": provider.target_url,
                    "proxy_url": provider.proxy_url,
                }
            )
        return infos

    async def start(self) -> None:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        self._load_providers()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await self.site.start()

        logger.info("Streamify proxy started at http://127.0.0.1:%s", self.port)
        if self.debug:
            logger.info("Streamify debug logging enabled.")
        if not self.providers:
            logger.warning("No provider routes configured.")
        else:
            for info in self.get_route_infos():
                proxy_text = info["proxy_url"] or "(none)"
                logger.info(
                    "Route /%s -> %s (proxy: %s)",
                    info["route_name"],
                    info["target_url"],
                    proxy_text,
                )

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            self.site = None

        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

        logger.info("Streamify proxy stopped.")
