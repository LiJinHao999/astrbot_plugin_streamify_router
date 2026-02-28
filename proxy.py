import json
import pathlib
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientResponse, ClientSession, web

from astrbot.api import logger

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


# 注入到请求的提示，引导模型正确填写工具参数
_EMPTY_ARGS_HINT = (
    "IMPORTANT: When calling a tool/function, you MUST provide valid JSON for ALL "
    "required arguments. Never invoke a tool with an empty argument object {} if the "
    "tool has required parameters. Fill every required field before calling the tool."
)


class ProviderHandler:
    """Base handler for forwarding and stream utilities."""

    def __init__(
        self,
        target_url: str,
        proxy_url: str = "",
        session: Optional[ClientSession] = None,
        debug: bool = False,
        request_timeout: float = 120.0,
    ):
        self.target = target_url.rstrip("/")
        self.proxy = proxy_url.strip() or None
        self.session = session
        self.debug = bool(debug)
        self.request_timeout = self._normalize_timeout(request_timeout)

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

    async def _passthrough(self, req: web.Request, sub_path: str) -> web.Response:
        url = self._build_url(sub_path)
        headers = self._forward_headers(req)
        body = await req.read()
        async with self._request(
            req.method,
            url,
            headers=headers,
            params=req.query,
            data=body,
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


class OpenAIChatHandler(ProviderHandler):
    ENDPOINT = "v1/chat/completions"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _new_choice_slot() -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content_parts": [],
            "finish_reason": None,
            "tool_calls": {},
            "tool_calls_seen": False,
            "legacy_function_call_seen": False,
        }

    @staticmethod
    def _new_tool_slot() -> Dict[str, Any]:
        return {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": []},
        }

    @staticmethod
    def _normalize_arguments_piece(arguments: Any) -> Optional[str]:
        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            return None
        try:
            return json.dumps(arguments, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(arguments)

    def _apply_function_payload(
        self,
        tc_slot: Dict[str, Any],
        function_payload: Any,
        append_arguments: bool,
    ) -> None:
        if not isinstance(function_payload, dict):
            return
        if isinstance(function_payload.get("name"), str):
            tc_slot["function"]["name"] = function_payload["name"]
        arguments_piece = self._normalize_arguments_piece(function_payload.get("arguments"))
        if arguments_piece is None:
            return
        if append_arguments:
            tc_slot["function"]["arguments"].append(arguments_piece)
        else:
            tc_slot["function"]["arguments"] = [arguments_piece]

    def _update_tool_calls(
        self,
        slot: Dict[str, Any],
        payload: Dict[str, Any],
        append_arguments: bool,
    ) -> None:
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            slot["tool_calls_seen"] = True
        for fallback_idx, tool_call in enumerate(tool_calls or []):
            if not isinstance(tool_call, dict):
                continue

            tc_idx = self._safe_int(tool_call.get("index", fallback_idx), fallback_idx)
            tc_slot = slot["tool_calls"].setdefault(tc_idx, self._new_tool_slot())
            if isinstance(tool_call.get("id"), str):
                tc_slot["id"] = tool_call["id"]
            if isinstance(tool_call.get("type"), str):
                tc_slot["type"] = tool_call["type"]
            self._apply_function_payload(
                tc_slot,
                tool_call.get("function") or {},
                append_arguments=append_arguments,
            )

        function_call = payload.get("function_call")
        if isinstance(function_call, dict):
            slot["legacy_function_call_seen"] = True
            tc_slot = slot["tool_calls"].setdefault(0, self._new_tool_slot())
            self._apply_function_payload(
                tc_slot,
                function_call,
                append_arguments=append_arguments,
            )

    def _update_choice_slot(self, slot: Dict[str, Any], choice: Dict[str, Any]) -> None:
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}

        if isinstance(delta.get("role"), str):
            slot["role"] = delta["role"]
        elif isinstance(message.get("role"), str):
            slot["role"] = message["role"]

        content_delta = delta.get("content")
        if isinstance(content_delta, str):
            slot["content_parts"].append(content_delta)
        elif isinstance(content_delta, list):
            for part in content_delta:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    slot["content_parts"].append(text)

        content_message = message.get("content")
        if isinstance(content_message, str):
            slot["content_parts"] = [content_message]
        elif isinstance(content_message, list):
            full_parts: List[str] = []
            for part in content_message:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    full_parts.append(text)
            if full_parts:
                slot["content_parts"] = full_parts

        self._update_tool_calls(slot, delta, append_arguments=True)
        self._update_tool_calls(slot, message, append_arguments=False)

        if choice.get("finish_reason") is not None:
            slot["finish_reason"] = choice.get("finish_reason")

    @staticmethod
    def _merge_chunk_meta(result: Dict[str, Any], chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not result["id"] and chunk.get("id"):
            result["id"] = chunk["id"]
        if not result["model"] and chunk.get("model"):
            result["model"] = chunk["model"]
        if chunk.get("created"):
            result["created"] = chunk["created"]
        if chunk.get("system_fingerprint"):
            result["system_fingerprint"] = chunk["system_fingerprint"]
        if isinstance(chunk.get("usage"), dict):
            return chunk["usage"]
        return None

    @staticmethod
    def _build_choice_output(idx: int, slot: Dict[str, Any]) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": slot["role"],
            "content": "".join(slot["content_parts"]),
        }
        inferred_finish_reason = "stop"
        if slot["tool_calls"]:
            assembled_tools = []
            for tc_idx in sorted(slot["tool_calls"].keys()):
                tool = slot["tool_calls"][tc_idx]
                assembled_tools.append(
                    {
                        "id": tool["id"] or f"call_{tc_idx}",
                        "type": tool["type"] or "function",
                        "function": {
                            "name": tool["function"]["name"],
                            "arguments": "".join(tool["function"]["arguments"]),
                        },
                    }
                )

            if slot.get("tool_calls_seen"):
                message["tool_calls"] = assembled_tools
                inferred_finish_reason = "tool_calls"
            elif slot.get("legacy_function_call_seen") and len(assembled_tools) == 1:
                message["function_call"] = assembled_tools[0]["function"]
                inferred_finish_reason = "function_call"
            else:
                message["tool_calls"] = assembled_tools
                inferred_finish_reason = "tool_calls"

            if message["content"] == "":
                message["content"] = None

        return {
            "index": idx,
            "message": message,
            "finish_reason": slot["finish_reason"] or inferred_finish_reason,
        }

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": "",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [],
        }
        choices: Dict[int, Dict[str, Any]] = {}
        usage: Optional[Dict[str, Any]] = None

        async for _event, data in self._iter_sse_events(resp):
            if not data:
                continue
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            merged_usage = self._merge_chunk_meta(result, chunk)
            if merged_usage is not None:
                usage = merged_usage

            for choice in chunk.get("choices", []):
                idx = self._safe_int(choice.get("index", 0))
                slot = choices.setdefault(idx, self._new_choice_slot())
                self._update_choice_slot(slot, choice)

        for idx in sorted(choices.keys()):
            result["choices"].append(self._build_choice_output(idx, choices[idx]))

        if usage is not None:
            result["usage"] = usage
        if not result["id"]:
            result["id"] = f"chatcmpl-proxy-{int(time.time() * 1000)}"
        return result

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

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

            result = await self._build_non_stream_response(resp)
            return web.json_response(result)


class ClaudeHandler(ProviderHandler):
    ENDPOINT = "v1/messages"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _init_state() -> Dict[str, Any]:
        return {
            "message_id": "",
            "model": "",
            "role": "assistant",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content_blocks": {},
        }

    @staticmethod
    def _handle_message_start(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        message = payload.get("message") or {}
        if isinstance(message.get("id"), str):
            state["message_id"] = message["id"]
        if isinstance(message.get("model"), str):
            state["model"] = message["model"]
        if isinstance(message.get("role"), str):
            state["role"] = message["role"]

        start_usage = message.get("usage") or {}
        if isinstance(start_usage.get("input_tokens"), int):
            state["usage"]["input_tokens"] = start_usage["input_tokens"]

        initial_content = message.get("content")
        if isinstance(initial_content, list):
            for idx, block in enumerate(initial_content):
                if isinstance(block, dict):
                    state["content_blocks"][idx] = dict(block)

    def _handle_content_block_start(self, state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        idx = self._safe_int(payload.get("index", len(state["content_blocks"])))
        block = payload.get("content_block") or {}
        if isinstance(block, dict):
            state["content_blocks"][idx] = dict(block)

    def _handle_content_block_delta(self, state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        idx = self._safe_int(payload.get("index", 0))
        block = state["content_blocks"].setdefault(idx, {"type": "text", "text": ""})
        delta = payload.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == "text_delta":
            block["type"] = "text"
            block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
        elif delta_type == "input_json_delta":
            block["_input_json"] = (
                f"{block.get('_input_json', '')}{delta.get('partial_json', '')}"
            )

    @staticmethod
    def _handle_message_delta(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        delta = payload.get("delta") or {}
        if "stop_reason" in delta:
            state["stop_reason"] = delta.get("stop_reason")
        if "stop_sequence" in delta:
            state["stop_sequence"] = delta.get("stop_sequence")

        delta_usage = payload.get("usage") or {}
        if isinstance(delta_usage.get("output_tokens"), int):
            state["usage"]["output_tokens"] = delta_usage["output_tokens"]

    @staticmethod
    def _build_content(content_blocks: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for idx in sorted(content_blocks.keys()):
            block = dict(content_blocks[idx])
            partial_input = block.pop("_input_json", None)
            if isinstance(partial_input, str) and partial_input:
                try:
                    block["input"] = json.loads(partial_input)
                except json.JSONDecodeError:
                    block["input"] = partial_input
            content.append(block)
        return content

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        state = self._init_state()
        async for event_name, data in self._iter_sse_events(resp):
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event_name or payload.get("type")
            if event_type == "message_start":
                self._handle_message_start(state, payload)
            elif event_type == "content_block_start":
                self._handle_content_block_start(state, payload)
            elif event_type == "content_block_delta":
                self._handle_content_block_delta(state, payload)
            elif event_type == "message_delta":
                self._handle_message_delta(state, payload)

        return {
            "id": state["message_id"] or f"msg_proxy_{int(time.time() * 1000)}",
            "type": "message",
            "role": state["role"],
            "model": state["model"],
            "content": self._build_content(state["content_blocks"]),
            "stop_reason": state["stop_reason"],
            "stop_sequence": state["stop_sequence"],
            "usage": state["usage"],
        }

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

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

            response_data = await self._build_non_stream_response(resp)
            return web.json_response(response_data)


class GeminiHandler(ProviderHandler):
    def __init__(
        self,
        target_url: str,
        proxy_url: str = "",
        session=None,
        debug: bool = False,
        request_timeout: float = 120.0,
        gemini_fix_retries: int = 2,
        gemini_extract_args: bool = False,
    ):
        super().__init__(target_url, proxy_url, session=session, debug=debug, request_timeout=request_timeout)
        self.gemini_fix_retries = max(0, int(gemini_fix_retries))
        self.gemini_extract_args = bool(gemini_extract_args)

    def matches(self, sub_path: str) -> bool:
        path = sub_path.strip("/")
        return ":generateContent" in path or ":streamGenerateContent" in path

    @staticmethod
    def _init_state() -> Dict[str, Any]:
        return {
            "content_parts": {},
            "candidate_meta": {},
            "role": "model",
            "usage_metadata": None,
            "prompt_feedback": None,
            "model_version": None,
        }

    @staticmethod
    def _first_candidate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        candidate = candidates[0]
        return candidate if isinstance(candidate, dict) else None

    @staticmethod
    def _merge_value(existing: Any, incoming: Any) -> Any:
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged = dict(existing)
            for key, value in incoming.items():
                if key in merged:
                    merged[key] = GeminiHandler._merge_value(merged[key], value)
                else:
                    merged[key] = value
            return merged
        return incoming

    @staticmethod
    def _append_content_parts(state: Dict[str, Any], content: Dict[str, Any]) -> None:
        if isinstance(content.get("role"), str):
            state["role"] = content["role"]

        parts = content.get("parts")
        if not isinstance(parts, list):
            return
        for idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            slot = state["content_parts"].setdefault(idx, {})
            text = part.get("text")
            if isinstance(text, str):
                if isinstance(slot.get("text"), str):
                    slot["text"] = f"{slot['text']}{text}"
                else:
                    slot["text"] = text
            for key, value in part.items():
                if key == "text":
                    continue
                slot[key] = GeminiHandler._merge_value(slot.get(key), value)

    @staticmethod
    def _merge_payload(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        candidate = GeminiHandler._first_candidate(payload)
        if candidate is not None:
            content = candidate.get("content")
            if isinstance(content, dict):
                GeminiHandler._append_content_parts(state, content)

            for key, value in candidate.items():
                if key != "content":
                    state["candidate_meta"][key] = value

        if isinstance(payload.get("usageMetadata"), dict):
            state["usage_metadata"] = payload["usageMetadata"]
        if isinstance(payload.get("promptFeedback"), dict):
            state["prompt_feedback"] = payload["promptFeedback"]
        if isinstance(payload.get("modelVersion"), str):
            state["model_version"] = payload["modelVersion"]

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        state = self._init_state()
        async for _event, data in self._iter_sse_events(resp):
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            self._merge_payload(state, payload)

        candidate_out: Dict[str, Any] = dict(state["candidate_meta"])
        merged_parts: List[Dict[str, Any]] = []
        for idx in sorted(state["content_parts"].keys()):
            part = state["content_parts"][idx]
            if isinstance(part, dict):
                merged_parts.append(part)
        candidate_out["content"] = {
            "role": state["role"],
            "parts": merged_parts if merged_parts else [{"text": ""}],
        }

        response_data: Dict[str, Any] = {"candidates": [candidate_out]}
        if state["usage_metadata"] is not None:
            response_data["usageMetadata"] = state["usage_metadata"]
        if state["prompt_feedback"] is not None:
            response_data["promptFeedback"] = state["prompt_feedback"]
        if state["model_version"] is not None:
            response_data["modelVersion"] = state["model_version"]
        return response_data

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
        current_body = body
        for attempt in range(self.gemini_fix_retries + 1):
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

            if not self._has_empty_function_call(response_data):
                break
            if attempt < self.gemini_fix_retries:
                if self.debug:
                    logger.info(
                        "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                        attempt + 1, self.gemini_fix_retries,
                    )
                current_body = self._inject_hint(body)

        if self.gemini_extract_args and self._has_empty_function_call(response_data):
            empty_fc = self._find_empty_function_call(response_data)
            if empty_fc is not None:
                extracted = await self._extract_args_as_json(
                    body, empty_fc.get("name", ""), stream_path, headers, params
                )
                if extracted is not None:
                    self._patch_function_call_args(response_data, empty_fc.get("name", ""), extracted)
                    if self.debug:
                        logger.info("Streamify: 成功提取 %s 的参数: %s", empty_fc.get("name"), extracted)
                elif self.debug:
                    logger.info("Streamify: JSON 参数提取失败，返回原始响应")

        return web.json_response(response_data)

    @staticmethod
    def _has_empty_function_call(response_data: Dict[str, Any]) -> bool:
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict) and not fc.get("args"):
                    return True
        return False

    @staticmethod
    def _inject_hint(body: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(body)
        hint_part = {"text": _EMPTY_ARGS_HINT}
        sys_inst = body.get("systemInstruction")
        if isinstance(sys_inst, dict):
            parts = list(sys_inst.get("parts", []))
            parts.append(hint_part)
            body["systemInstruction"] = {**sys_inst, "parts": parts}
        else:
            body["systemInstruction"] = {"parts": [hint_part]}
        return body

    @staticmethod
    def _find_empty_function_call(response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict) and not fc.get("args"):
                    return fc
        return None

    @staticmethod
    def _patch_function_call_args(
        response_data: Dict[str, Any], function_name: str, args: Dict[str, Any]
    ) -> None:
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict) and fc.get("name") == function_name and not fc.get("args"):
                    fc["args"] = args
                    return

    async def _extract_args_as_json(
        self,
        original_body: Dict[str, Any],
        function_name: str,
        stream_path: str,
        headers: Dict[str, str],
        params: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """构造聚焦请求：只含工具说明 + 参数 schema + 对话上下文，让模型输出 JSON 参数。"""
        # 找到目标工具的完整声明
        func_desc = ""
        func_params_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == function_name:
                    func_desc = fd.get("description", "")
                    func_params_schema = fd.get("parameters", {})
                    break

        # 构造聚焦的 system instruction，只包含工具说明，过滤掉原有 system
        tool_system = (
            f"你是一个参数提取助手。用户正在使用工具 `{function_name}`。\n"
            f"工具说明：{func_desc}\n"
            f"参数 schema：{json.dumps(func_params_schema, ensure_ascii=False)}\n"
            f"根据对话内容，只输出调用该工具所需的 JSON 参数对象，不要包含任何其他文字。"
        )

        # 只保留对话上下文（user/model 轮次），不传原 system
        contents = list(original_body.get("contents", []))

        gen_cfg = dict(original_body.get("generationConfig") or {})
        gen_cfg["responseMimeType"] = "application/json"

        extract_body: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": tool_system}]},
            "generationConfig": gen_cfg,
            # 不传 tools，避免模型再次发出空的 functionCall
        }

        try:
            async with self._request(
                "POST",
                self._build_url(stream_path),
                json=extract_body,
                headers=headers,
                params=params,
            ) as resp:
                if resp.status != 200:
                    return None
                text_parts: List[str] = []
                async for _event, data in self._iter_sse_events(resp):
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                        for cand in chunk.get("candidates", []):
                            for part in cand.get("content", {}).get("parts", []):
                                if isinstance(part.get("text"), str):
                                    text_parts.append(part["text"])
                    except Exception:
                        continue
                text = "".join(text_parts).strip()
                if not text:
                    return None
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                result = json.loads(text)
                return result if isinstance(result, dict) else None
        except Exception:
            return None


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
        gemini_fix_retries: int = 2,
        gemini_extract_args: bool = False,
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
            ),
            ClaudeHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
            ),
            GeminiHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
                gemini_fix_retries=gemini_fix_retries,
                gemini_extract_args=gemini_extract_args,
            ),
            OpenAIResponsesHandler(
                self.target_url,
                self.proxy_url,
                session=session,
                debug=self.debug,
                request_timeout=request_timeout,
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
        gemini_fix_retries: int = 2,
        gemini_extract_args: bool = False,
    ):
        self.port = port
        self.providers_config = providers or []
        self.debug = bool(debug)
        self.request_timeout = ProviderHandler._normalize_timeout(request_timeout)
        self.gemini_fix_retries = max(0, int(gemini_fix_retries))
        self.gemini_extract_args = bool(gemini_extract_args)
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

            self.providers[route_name] = ProviderRoute(
                route_name,
                target_url,
                proxy_url,
                session=self.session,
                debug=self.debug,
                request_timeout=self.request_timeout,
                gemini_fix_retries=self.gemini_fix_retries,
                gemini_extract_args=self.gemini_extract_args,
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
                        _f.write(f"[{ts}] <<< {req.method.upper()} {path_text} status={response.status} elapsed={elapsed_ms}ms\n")
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
