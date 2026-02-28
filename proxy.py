import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientResponse, web

from astrbot.api import logger


class ProviderHandler:
    """Base handler for forwarding and stream utilities."""

    def __init__(self, target_url: str, proxy_url: str = "", debug: bool = False):
        self.target = target_url.rstrip("/")
        self.proxy = proxy_url.strip() or None
        self.debug = bool(debug)

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
        async with aiohttp.ClientSession() as session:
            async with session.request(
                req.method,
                url,
                headers=headers,
                params=req.query,
                data=body,
                proxy=self.proxy,
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
        async with aiohttp.ClientSession() as session:
            async with session.request(
                req.method,
                url,
                json=body,
                headers=headers,
                params=params if params is not None else req.query,
                proxy=self.proxy,
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
        buffer = ""
        event_name: Optional[str] = None
        data_lines: List[str] = []

        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                raw_line, buffer = buffer.split("\n", 1)
                line = raw_line.rstrip("\r")

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
        url = self._build_url(sub_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                params=req.query,
                proxy=self.proxy,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )

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

                    if not result["id"] and chunk.get("id"):
                        result["id"] = chunk["id"]
                    if not result["model"] and chunk.get("model"):
                        result["model"] = chunk["model"]
                    if chunk.get("created"):
                        result["created"] = chunk["created"]
                    if chunk.get("system_fingerprint"):
                        result["system_fingerprint"] = chunk["system_fingerprint"]
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]

                    for choice in chunk.get("choices", []):
                        idx = int(choice.get("index", 0))
                        slot = choices.setdefault(
                            idx,
                            {
                                "role": "assistant",
                                "content_parts": [],
                                "finish_reason": None,
                                "tool_calls": {},
                            },
                        )

                        delta = choice.get("delta") or {}
                        if isinstance(delta.get("role"), str):
                            slot["role"] = delta["role"]

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

                        tool_calls = delta.get("tool_calls")
                        if isinstance(tool_calls, list):
                            for tool_call in tool_calls:
                                if not isinstance(tool_call, dict):
                                    continue
                                tc_idx = int(tool_call.get("index", 0))
                                tc_slot = slot["tool_calls"].setdefault(
                                    tc_idx,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": []},
                                    },
                                )
                                if isinstance(tool_call.get("id"), str):
                                    tc_slot["id"] = tool_call["id"]
                                if isinstance(tool_call.get("type"), str):
                                    tc_slot["type"] = tool_call["type"]

                                function = tool_call.get("function") or {}
                                if isinstance(function.get("name"), str):
                                    tc_slot["function"]["name"] = function["name"]
                                arguments = function.get("arguments")
                                if isinstance(arguments, str):
                                    tc_slot["function"]["arguments"].append(arguments)

                        if choice.get("finish_reason") is not None:
                            slot["finish_reason"] = choice.get("finish_reason")

                for idx in sorted(choices.keys()):
                    slot = choices[idx]
                    message: Dict[str, Any] = {
                        "role": slot["role"],
                        "content": "".join(slot["content_parts"]),
                    }

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
                        message["tool_calls"] = assembled_tools
                        if message["content"] == "":
                            message["content"] = None

                    result["choices"].append(
                        {
                            "index": idx,
                            "message": message,
                            "finish_reason": slot["finish_reason"] or "stop",
                        }
                    )

                if usage is not None:
                    result["usage"] = usage
                if not result["id"]:
                    result["id"] = f"chatcmpl-proxy-{int(time.time() * 1000)}"

                return web.json_response(result)


class ClaudeHandler(ProviderHandler):
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
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

        body["stream"] = True
        url = self._build_url(sub_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                params=req.query,
                proxy=self.proxy,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )

                message_id = ""
                model = ""
                role = "assistant"
                stop_reason = None
                stop_sequence = None
                usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
                content_blocks: Dict[int, Dict[str, Any]] = {}

                async for event_name, data in self._iter_sse_events(resp):
                    if not data:
                        continue

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event_name or payload.get("type")

                    if event_type == "message_start":
                        message = payload.get("message") or {}
                        if isinstance(message.get("id"), str):
                            message_id = message["id"]
                        if isinstance(message.get("model"), str):
                            model = message["model"]
                        if isinstance(message.get("role"), str):
                            role = message["role"]

                        start_usage = message.get("usage") or {}
                        if isinstance(start_usage.get("input_tokens"), int):
                            usage["input_tokens"] = start_usage["input_tokens"]

                        initial_content = message.get("content")
                        if isinstance(initial_content, list):
                            for idx, block in enumerate(initial_content):
                                if isinstance(block, dict):
                                    content_blocks[idx] = dict(block)

                    elif event_type == "content_block_start":
                        idx = int(payload.get("index", len(content_blocks)))
                        block = payload.get("content_block") or {}
                        if isinstance(block, dict):
                            content_blocks[idx] = dict(block)

                    elif event_type == "content_block_delta":
                        idx = int(payload.get("index", 0))
                        block = content_blocks.setdefault(idx, {"type": "text", "text": ""})
                        delta = payload.get("delta") or {}
                        delta_type = delta.get("type")

                        if delta_type == "text_delta":
                            block["type"] = "text"
                            block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
                        elif delta_type == "input_json_delta":
                            block["_input_json"] = f"{block.get('_input_json', '')}{delta.get('partial_json', '')}"

                    elif event_type == "message_delta":
                        delta = payload.get("delta") or {}
                        if "stop_reason" in delta:
                            stop_reason = delta.get("stop_reason")
                        if "stop_sequence" in delta:
                            stop_sequence = delta.get("stop_sequence")

                        delta_usage = payload.get("usage") or {}
                        if isinstance(delta_usage.get("output_tokens"), int):
                            usage["output_tokens"] = delta_usage["output_tokens"]

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

                response_data = {
                    "id": message_id or f"msg_proxy_{int(time.time() * 1000)}",
                    "type": "message",
                    "role": role,
                    "model": model,
                    "content": content,
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                    "usage": usage,
                }
                return web.json_response(response_data)


class GeminiHandler(ProviderHandler):
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
        url = self._build_url(stream_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                params=params,
                proxy=self.proxy,
            ) as resp:
                if resp.status != 200:
                    return web.Response(
                        status=resp.status,
                        headers=self._response_headers(resp),
                        text=await resp.text(),
                    )

                text_parts: List[str] = []
                candidate_meta: Dict[str, Any] = {}
                role = "model"
                usage_metadata: Optional[Dict[str, Any]] = None
                prompt_feedback: Optional[Dict[str, Any]] = None
                model_version: Optional[str] = None

                async for _event, data in self._iter_sse_events(resp):
                    if not data:
                        continue

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    candidates = payload.get("candidates")
                    if isinstance(candidates, list) and candidates:
                        candidate = candidates[0]
                        if isinstance(candidate, dict):
                            content = candidate.get("content") or {}
                            if isinstance(content, dict):
                                if isinstance(content.get("role"), str):
                                    role = content["role"]
                                parts = content.get("parts")
                                if isinstance(parts, list):
                                    for part in parts:
                                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                                            text_parts.append(part["text"])

                            for key, value in candidate.items():
                                if key == "content":
                                    continue
                                candidate_meta[key] = value

                    if isinstance(payload.get("usageMetadata"), dict):
                        usage_metadata = payload["usageMetadata"]
                    if isinstance(payload.get("promptFeedback"), dict):
                        prompt_feedback = payload["promptFeedback"]
                    if isinstance(payload.get("modelVersion"), str):
                        model_version = payload["modelVersion"]

                candidate_out: Dict[str, Any] = dict(candidate_meta)
                candidate_out["content"] = {
                    "role": role,
                    "parts": [{"text": "".join(text_parts)}],
                }

                response_data: Dict[str, Any] = {
                    "candidates": [candidate_out],
                }
                if usage_metadata is not None:
                    response_data["usageMetadata"] = usage_metadata
                if prompt_feedback is not None:
                    response_data["promptFeedback"] = prompt_feedback
                if model_version is not None:
                    response_data["modelVersion"] = model_version

                return web.json_response(response_data)


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
        url = self._build_url(sub_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                params=req.query,
                proxy=self.proxy,
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
        debug: bool = False,
    ):
        self.route_name = route_name
        self.target_url = target_url.rstrip("/")
        self.proxy_url = proxy_url.strip()
        self.debug = bool(debug)
        self.base = ProviderHandler(self.target_url, self.proxy_url, self.debug)
        self.handlers: List[ProviderHandler] = [
            OpenAIChatHandler(self.target_url, self.proxy_url, self.debug),
            ClaudeHandler(self.target_url, self.proxy_url, self.debug),
            GeminiHandler(self.target_url, self.proxy_url, self.debug),
            OpenAIResponsesHandler(self.target_url, self.proxy_url, self.debug),
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
    ):
        self.port = port
        self.providers_config = providers or []
        self.debug = bool(debug)
        self.providers: Dict[str, ProviderRoute] = {}
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
                debug=self.debug,
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

        try:
            response = await provider.dispatch(req, sub_path)
            if self.debug:
                path_text = f"/{route_name}"
                if sub_path:
                    path_text = f"{path_text}/{sub_path.strip('/')}"
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "Streamify debug handled: %s %s -> %s status=%s elapsed=%dms",
                    req.method.upper(),
                    path_text,
                    provider.target_url,
                    response.status,
                    elapsed_ms,
                )
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
            logger.info("Streamify proxy stopped.")
