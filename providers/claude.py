import json
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import ClaudeFakeNonStream
from ..fc_enhance import ClaudeFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG, register_handler


def _inject_fc_failure_text_claude(result: Dict[str, Any], tool_name: str) -> None:
    """向 Claude 格式响应注入工具调用失败提示。"""
    hint = f"[{_FC_FAILURE_MSG}]"
    result["content"] = [{"type": "text", "text": hint}]
    result["stop_reason"] = "end_turn"


@register_handler
class ClaudeHandler(ProviderHandler, ClaudeFakeNonStream, ClaudeFCEnhance):
    ENDPOINT = "v1/messages"

    def _find_hinted_empty_block(self, result: dict) -> Optional[dict]:
        for block in (result.get("content") or []):
            if block.get("type") != "tool_use":
                continue
            if block.get("name", "") not in self._hint_tools:
                continue
            inp = block.get("input")
            if not inp or (isinstance(inp, dict) and not inp):
                return block
        return None

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    # ------------------------------------------------------------------
    # Layer 2 用：将完整结果以 Claude SSE 格式返回给流式客户端
    # ------------------------------------------------------------------
    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)

        async def send(name: str, data: Dict[str, Any]) -> None:
            await response.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

        await send("message_start", {
            "type": "message_start",
            "message": {
                "id": result.get("id", ""),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": result.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": result.get("usage", {}),
            },
        })
        for i, block in enumerate(result.get("content", [])):
            if block.get("type") == "tool_use":
                cb_start = {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", "")}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}
            else:
                cb_start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block.get("text", "")}

            await send("content_block_start", {"type": "content_block_start", "index": i, "content_block": cb_start})
            await send("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta})
            await send("content_block_stop", {"type": "content_block_stop", "index": i})

        await send("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": result.get("stop_reason"), "stop_sequence": result.get("stop_sequence")},
            "usage": {"output_tokens": (result.get("usage") or {}).get("output_tokens", 0)},
        })
        await send("message_stop", {"type": "message_stop"})
        await response.write_eof()
        return response

    # ------------------------------------------------------------------
    # 流式 FC hook 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    async def _write_event(client: web.StreamResponse, event_name: Optional[str], payload: Dict[str, Any]) -> None:
        if event_name:
            await client.write(f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode())
        else:
            await client.write(f"data: {json.dumps(payload)}\n\n".encode())

    async def _write_tc_events(
        self,
        client: web.StreamResponse,
        result: Dict[str, Any],
        start_idx: int,
        msg_meta: Dict[str, Any],
    ) -> None:
        """将修复后的 tool_use / 失败提示块以 Claude SSE 事件格式写出。"""
        for i, block in enumerate(result.get("content", [])):
            idx = start_idx + i
            btype = block.get("type", "text")

            if btype == "tool_use":
                cb_start = {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", "")}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}
            else:
                cb_start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block.get("text", "")}

            await self._write_event(client, "content_block_start",
                                    {"type": "content_block_start", "index": idx, "content_block": cb_start})
            await self._write_event(client, "content_block_delta",
                                    {"type": "content_block_delta", "index": idx, "delta": delta})
            await self._write_event(client, "content_block_stop",
                                    {"type": "content_block_stop", "index": idx})

        stop_reason = result.get("stop_reason", "tool_use")
        await self._write_event(client, "message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": result.get("stop_sequence")},
            "usage": {"output_tokens": (result.get("usage") or {}).get("output_tokens", 0)},
        })
        await self._write_event(client, "message_stop", {"type": "message_stop"})

    # ------------------------------------------------------------------
    # 流式请求主入口：文本 block 直通，tool_use block 拦截检测修复后发出
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

            # 记录 message 元信息
            msg_meta: Dict[str, Any] = {"id": "", "model": "", "usage": {}}
            # 一旦检测到 tool_use block，后续所有事件都进缓冲
            tc_detected = False
            tc_buffer: List[Tuple[Optional[str], Dict[str, Any]]] = []
            # 手动累积 tool_use block 数据（用于 FC 检测）
            tc_blocks: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, _input}
            # 非 tool_use content block 的最大 index + 1（用于 start_idx 计算）
            max_forwarded_idx: int = -1

            async for event_name, data in self._iter_sse_events(resp):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    if not tc_detected:
                        sse = (f"event: {event_name}\ndata: {data}\n\n"
                               if event_name else f"data: {data}\n\n")
                        await client.write(sse.encode())
                    continue
                if not isinstance(payload, dict):
                    continue

                event_type = event_name or payload.get("type")

                if event_type == "message_start":
                    msg = payload.get("message") or {}
                    msg_meta["id"] = msg.get("id", "")
                    msg_meta["model"] = msg.get("model", "")
                    msg_meta["usage"] = msg.get("usage") or {}
                    await self._write_event(client, event_name, payload)

                elif event_type == "content_block_start":
                    cb = payload.get("content_block") or {}
                    idx = int(payload.get("index", 0))
                    if cb.get("type") == "tool_use":
                        tc_detected = True
                        tc_blocks[idx] = {
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "_input": "",
                        }
                        tc_buffer.append((event_name, payload))
                    else:
                        max_forwarded_idx = max(max_forwarded_idx, idx)
                        await self._write_event(client, event_name, payload)

                elif event_type == "content_block_delta":
                    idx = int(payload.get("index", 0))
                    if idx in tc_blocks:
                        delta = payload.get("delta") or {}
                        if delta.get("type") == "input_json_delta":
                            tc_blocks[idx]["_input"] += delta.get("partial_json", "")
                        tc_buffer.append((event_name, payload))
                    else:
                        await self._write_event(client, event_name, payload)

                elif event_type == "content_block_stop":
                    idx = int(payload.get("index", 0))
                    if idx in tc_blocks:
                        tc_buffer.append((event_name, payload))
                    else:
                        await self._write_event(client, event_name, payload)

                elif event_type in ("message_delta", "message_stop"):
                    if tc_detected:
                        tc_buffer.append((event_name, payload))
                    else:
                        await self._write_event(client, event_name, payload)

                else:
                    # 未知事件：不缓冲直接透传
                    if not tc_detected:
                        await self._write_event(client, event_name, payload)

        if not tc_detected:
            await client.write_eof()
            return client

        # 构建用于 FC 检测的 assembled 结果
        tc_content: List[Dict[str, Any]] = []
        for idx in sorted(tc_blocks.keys()):
            b = tc_blocks[idx]
            try:
                inp = json.loads(b["_input"]) if b["_input"] else {}
            except Exception:
                inp = b["_input"]
            tc_content.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": inp})

        assembled: Dict[str, Any] = {
            "id": msg_meta["id"],
            "type": "message",
            "role": "assistant",
            "model": msg_meta["model"],
            "content": tc_content,
            "stop_reason": "tool_use",
            "usage": msg_meta.get("usage") or {},
        }

        tools = clean_body.get("tools") or []
        failed = self._find_failed_function_call(assembled, tools)
        if failed is None and self._hint_tools:
            failed = self._find_hinted_empty_block(assembled)

        start_idx = max_forwarded_idx + 1

        if failed is None:
            # 无需修复：重放缓冲事件
            for evt_name, evt_payload in tc_buffer:
                await self._write_event(client, evt_name, evt_payload)
            await client.write_eof()
            return client

        # 需要修复：Layer 1 + 重试
        result = assembled

        if self.extract_args:
            fn_name = failed.get("name", "")
            extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers)
            if extracted is not None:
                self._patch_function_call_args(result, fn_name, extracted)
                if self.debug:
                    logger.info("Streamify [Layer1]: 成功提取 Claude 工具 %s 参数(流式)", fn_name)
                await self._write_tc_events(client, result, start_idx, msg_meta)
                await client.write_eof()
                return client
            elif self.debug:
                logger.info("Streamify [Layer1]: Claude JSON 提取失败，重试(流式)")

        retry_name = failed.get("name", "") if failed else ""
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
                failed = self._find_hinted_empty_block(result)
            if failed is None:
                await self._write_tc_events(client, result, start_idx, msg_meta)
                await client.write_eof()
                return client

            if self.extract_args:
                fn_name = failed.get("name", "")
                extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers)
                if extracted is not None:
                    self._patch_function_call_args(result, fn_name, extracted)
                    await self._write_tc_events(client, result, start_idx, msg_meta)
                    await client.write_eof()
                    return client

        fail_name = failed.get("name", "unknown") if failed else "unknown"
        _inject_fc_failure_text_claude(result, fail_name)
        await self._write_tc_events(client, result, start_idx, msg_meta)
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
                    self._remember_hint_tool(tool_name)
                    result = self._build_corrected_tool_response(
                        tool_use_id, tool_name, extracted, body.get("model", "")
                    )
                    if client_wants_stream:
                        return await self._emit_as_sse(result, req)
                    return web.json_response(result)
                elif self.debug:
                    logger.info(
                        "Streamify [Layer2]: Claude 工具 %s 参数提取失败，继续正常转发",
                        tool_name,
                    )
                clean_body = {**body, "messages": ctx_messages}

        if not self.pseudo_non_stream and not client_wants_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        if client_wants_stream:
            # 无 tools 定义时直接透传
            if not clean_body.get("tools"):
                return await self._proxy_stream(req, sub_path, clean_body, headers)
            return await self._handle_stream_fc_hook(req, sub_path, clean_body, headers)

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
            failed_tc = self._find_hinted_empty_block(result)
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
            elif self.debug:
                logger.info("Streamify [Layer1]: Claude JSON 参数提取失败，尝试提示注入重试")

        retry_tool_name = failed_tc.get("name", "") if failed_tc else ""
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
                failed_tc = self._find_hinted_empty_block(result)
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

        _fail_name = failed_tc.get("name", "unknown") if failed_tc else "unknown"
        _inject_fc_failure_text_claude(result, _fail_name)
        return web.json_response(result)
