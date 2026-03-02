import json
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import OpenAIFakeNonStream
from ..fc_enhance import OpenAIFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG, register_handler


def _inject_fc_failure_text_openai(result: Dict[str, Any], tool_name: str) -> None:
    """向 OpenAI 格式响应注入工具调用失败提示。"""
    hint = f"[{_FC_FAILURE_MSG}]"
    for choice in result.get("choices", []):
        msg = choice.get("message")
        if isinstance(msg, dict):
            msg["content"] = hint
            msg.pop("tool_calls", None)
            msg.pop("function_call", None)
            choice["finish_reason"] = "stop"
            return


@register_handler
class OpenAIChatHandler(ProviderHandler, OpenAIFakeNonStream, OpenAIFCEnhance):
    ENDPOINT = "v1/chat/completions"

    def _find_hinted_empty_tc(self, result: dict) -> Optional[dict]:
        for choice in (result.get("choices") or []):
            for tc in ((choice.get("message") or {}).get("tool_calls") or []):
                fn = tc.get("function") or {}
                if fn.get("name", "") not in self._hint_tools:
                    continue
                args = fn.get("arguments", "")
                try:
                    parsed = json.loads(args) if isinstance(args, str) and args else {}
                except Exception:
                    parsed = {}
                if not parsed:
                    return tc
        return None

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    # ------------------------------------------------------------------
    # Layer 2 用：将完整的 non-stream 结果以 SSE 格式返回给流式客户端
    # ------------------------------------------------------------------
    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)
        base: Dict[str, Any] = {
            "id": result.get("id", ""),
            "object": "chat.completion.chunk",
            "created": result.get("created", int(time.time())),
            "model": result.get("model", ""),
        }
        if "system_fingerprint" in result:
            base["system_fingerprint"] = result["system_fingerprint"]
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            delta: Dict[str, Any] = {}
            if "role" in msg:
                delta["role"] = msg["role"]
            if msg.get("content") is not None:
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = [
                    {**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])
                ]
            chunk = {**base, "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}]}
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            finish_chunk = {**base, "choices": [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason")}]}
            if "usage" in result:
                finish_chunk["usage"] = result["usage"]
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    # ------------------------------------------------------------------
    # 流式 FC hook 专用辅助方法
    # ------------------------------------------------------------------
    def _assemble_tc_chunks(
        self, chunks: List[Dict[str, Any]], base_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从已解析的 TC chunk 列表组装 non-stream 格式，用于 FC 检测。"""
        result: Dict[str, Any] = {
            "id": base_meta.get("id") or f"chatcmpl-proxy-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": base_meta.get("created", int(time.time())),
            "model": base_meta.get("model", ""),
            "choices": [],
        }
        choices: Dict[int, Dict[str, Any]] = {}
        for chunk in chunks:
            for choice in (chunk.get("choices") or []):
                if isinstance(choice, dict):
                    idx = int(choice.get("index", 0))
                    slot = choices.setdefault(idx, self._new_choice_slot())
                    self._update_choice_slot(slot, choice)
        for idx in sorted(choices.keys()):
            result["choices"].append(self._build_choice_output(idx, choices[idx]))
        return result

    async def _write_tc_sse(
        self,
        client: web.StreamResponse,
        result: Dict[str, Any],
        base_meta: Dict[str, Any],
    ) -> None:
        """将修复后的 result（仅 TC / 失败提示部分）以 SSE chunk 格式写出。"""
        base: Dict[str, Any] = {
            "id": base_meta.get("id") or result.get("id", ""),
            "object": "chat.completion.chunk",
            "created": base_meta.get("created") or result.get("created", int(time.time())),
            "model": base_meta.get("model") or result.get("model", ""),
        }
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            delta: Dict[str, Any] = {}
            if msg.get("content") is not None:
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = [
                    {**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])
                ]
            if msg.get("function_call"):
                delta["function_call"] = msg["function_call"]
            if delta:
                chunk = {**base, "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}]}
                await client.write(f"data: {json.dumps(chunk)}\n\n".encode())
            finish_chunk = {**base, "choices": [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]}
            await client.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())

    # ------------------------------------------------------------------
    # 流式请求主入口：文本 chunk 直通，TC chunk 拦截检测修复后发出
    # ------------------------------------------------------------------
    async def _handle_stream_fc_hook(
        self,
        req: web.Request,
        sub_path: str,
        clean_body: Dict[str, Any],
        headers: Dict[str, str],
    ) -> web.Response:
        upstream_body = {**clean_body, "stream": True}

        async with self._request(
            "POST", self._build_url(sub_path),
            json=upstream_body, headers=headers, params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )

            client = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            })
            await client.prepare(req)

            base_meta: Dict[str, Any] = {"id": "", "model": "", "created": 0}
            tc_buffer: List[Dict[str, Any]] = []
            tc_detected = False

            async for _, data in self._iter_sse_events(resp):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    if not tc_detected:
                        await client.write(f"data: {data}\n\n".encode())
                    continue
                if not isinstance(chunk, dict):
                    continue

                if chunk.get("id"):
                    base_meta["id"] = chunk["id"]
                if chunk.get("model"):
                    base_meta["model"] = chunk["model"]
                if chunk.get("created"):
                    base_meta["created"] = chunk["created"]

                choices = chunk.get("choices") or []
                is_tc = any(
                    bool((c.get("delta") or {}).get("tool_calls"))
                    or bool((c.get("delta") or {}).get("function_call"))
                    or c.get("finish_reason") in ("tool_calls", "function_call")
                    for c in choices
                )
                if is_tc:
                    tc_detected = True
                    tc_buffer.append(chunk)
                else:
                    await client.write(f"data: {json.dumps(chunk)}\n\n".encode())

        # 纯文本响应：文本已实时发出，直接结束
        if not tc_detected:
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client

        # 组装 TC 结果用于 FC 检测
        assembled = self._assemble_tc_chunks(tc_buffer, base_meta)
        tools = clean_body.get("tools") or []

        failed = self._find_failed_function_call(assembled, tools)
        if failed is None and self._hint_tools:
            failed = self._find_hinted_empty_tc(assembled)

        if failed is None:
            # 无需修复：重放原始 TC chunks
            for c in tc_buffer:
                await client.write(f"data: {json.dumps(c)}\n\n".encode())
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client

        # 需要修复：Layer 1 + 重试
        result = assembled

        if self.extract_args:
            fn_name = (failed.get("function") or {}).get("name", "")
            extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers)
            if extracted is not None:
                _before = (failed.get("function") or {}).get("arguments", "{}")
                self._patch_function_call_args(result, fn_name, extracted)
                self._log_fc_modify("openai", 1, fn_name, _before, extracted)
                if self.debug:
                    logger.info("Streamify [Layer1]: 成功提取 OpenAI 工具 %s 参数(流式)", fn_name)
                await self._write_tc_sse(client, result, base_meta)
                await client.write(b"data: [DONE]\n\n")
                await client.write_eof()
                return client
            elif self.debug:
                logger.info("Streamify [Layer1]: OpenAI JSON 提取失败，重试(流式)")

        retry_name = (failed.get("function") or {}).get("name", "") if failed else ""
        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info("Streamify: 流式 TC 修复重试 (%d/%d)", attempt + 1, self.fix_retries)
            current_body = self._inject_hint(clean_body, retry_name)
            current_body["stream"] = True
            async with self._request(
                "POST", self._build_url(sub_path),
                json=current_body, headers=headers, params=req.query,
            ) as retry_resp:
                if retry_resp.status != 200:
                    break
                result = await self._build_non_stream_response(retry_resp)

            failed = self._find_failed_function_call(result, tools)
            if failed is None and self._hint_tools:
                failed = self._find_hinted_empty_tc(result)
            if failed is None:
                await self._write_tc_sse(client, result, base_meta)
                await client.write(b"data: [DONE]\n\n")
                await client.write_eof()
                return client

            if self.extract_args:
                fn_name = (failed.get("function") or {}).get("name", "")
                extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers)
                if extracted is not None:
                    _before = (failed.get("function") or {}).get("arguments", "{}")
                    self._patch_function_call_args(result, fn_name, extracted)
                    self._log_fc_modify("openai", 1, fn_name, _before, extracted)
                    await self._write_tc_sse(client, result, base_meta)
                    await client.write(b"data: [DONE]\n\n")
                    await client.write_eof()
                    return client

        # 全部重试失败，注入失败提示
        fail_name = (failed.get("function") or {}).get("name", "unknown") if failed else "unknown"
        _inject_fc_failure_text_openai(result, fail_name)
        await self._write_tc_sse(client, result, base_meta)
        await client.write(b"data: [DONE]\n\n")
        await client.write_eof()
        return client

    # ------------------------------------------------------------------
    # 主 handle
    # ------------------------------------------------------------------
    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)
        client_wants_stream = bool(body.get("stream"))
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
                    self._log_fc_modify("openai", 2, tool_name, "{}", extracted)
                    self._remember_hint_tool(tool_name)
                    result = self._build_corrected_tool_response(
                        tool_call_id, tool_name, extracted, body.get("model", "")
                    )
                    if client_wants_stream:
                        return await self._emit_as_sse(result, req)
                    return web.json_response(result)
                elif self.debug:
                    logger.info(
                        "Streamify [Layer2]: OpenAI 工具 %s 参数提取失败，继续正常转发",
                        tool_name,
                    )
                clean_body = {**body, "messages": ctx_messages}

        # 假非流禁用时直通（仅对非流式客户端；流式客户端始终进行 FC 处理）
        if not self.pseudo_non_stream and not client_wants_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        # 流式客户端：有条件透传 + TC 拦截检测修复
        if client_wants_stream:
            # 无 tools 定义时直接透传，不走 FC hook
            if not clean_body.get("tools") and not clean_body.get("functions"):
                return await self._proxy_stream(req, sub_path, clean_body, headers)
            return await self._handle_stream_fc_hook(req, sub_path, clean_body, headers)

        # 非流式客户端：内部转流收集，FC 检测修复后返回 JSON
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

        failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
        if failed_tc is None and self._hint_tools:
            failed_tc = self._find_hinted_empty_tc(result)
        if failed_tc is None:
            return web.json_response(result)

        if self.extract_args:
            function_name = (failed_tc.get("function") or {}).get("name", "")
            extracted = await self._extract_args_as_json(
                clean_body, function_name, sub_path, headers
            )
            if extracted is not None:
                _before = (failed_tc.get("function") or {}).get("arguments", "{}")
                self._patch_function_call_args(result, function_name, extracted)
                self._log_fc_modify("openai", 1, function_name, _before, extracted)
                if self.debug:
                    logger.info(
                        "Streamify [Layer1]: 成功提取 OpenAI 工具 %s 的参数: %s",
                        function_name, extracted,
                    )
                return web.json_response(result)
            elif self.debug:
                logger.info("Streamify [Layer1]: OpenAI JSON 参数提取失败，尝试提示注入重试")

        retry_tool_name = (failed_tc.get("function") or {}).get("name", "") if failed_tc else ""
        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info(
                    "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                    attempt + 1, self.fix_retries,
                )
            current_body = self._inject_hint(clean_body, retry_tool_name)
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
            if failed_tc is None and self._hint_tools:
                failed_tc = self._find_hinted_empty_tc(result)
            if failed_tc is None:
                return web.json_response(result)
            if self.extract_args:
                function_name = (failed_tc.get("function") or {}).get("name", "")
                extracted = await self._extract_args_as_json(
                    clean_body, function_name, sub_path, headers
                )
                if extracted is not None:
                    _before = (failed_tc.get("function") or {}).get("arguments", "{}")
                    self._patch_function_call_args(result, function_name, extracted)
                    self._log_fc_modify("openai", 1, function_name, _before, extracted)
                    if self.debug:
                        logger.info(
                            "Streamify [Layer1]: 成功提取 OpenAI 工具 %s 的参数: %s",
                            function_name, extracted,
                        )
                    return web.json_response(result)

        _fail_name = (failed_tc.get("function") or {}).get("name", "unknown") if failed_tc else "unknown"
        _inject_fc_failure_text_openai(result, _fail_name)
        return web.json_response(result)
