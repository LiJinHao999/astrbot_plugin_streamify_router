import json
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import ClaudeFakeNonStream
from ..fc_enhance import ClaudeFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG


def _inject_fc_failure_text_claude(result: Dict[str, Any], tool_name: str) -> None:
    """向 Claude 格式响应注入工具调用失败提示。"""
    hint = f"[{_FC_FAILURE_MSG}]"
    result["content"] = [{"type": "text", "text": hint}]
    result["stop_reason"] = "end_turn"


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

    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        """将聚合后的 Claude Messages 结果重新以 SSE 流格式返回给客户端。"""
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)

        async def send_event(name: str, data: Dict[str, Any]) -> None:
            line = f"event: {name}\ndata: {json.dumps(data)}\n\n"
            await response.write(line.encode())

        await send_event("message_start", {
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
            await send_event("content_block_start", {
                "type": "content_block_start",
                "index": i,
                "content_block": block,
            })

            if block.get("type") == "text":
                delta: Dict[str, Any] = {"type": "text_delta", "text": block.get("text", "")}
            elif block.get("type") == "tool_use":
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}
            else:
                delta = {"type": "text_delta", "text": ""}

            await send_event("content_block_delta", {
                "type": "content_block_delta",
                "index": i,
                "delta": delta,
            })
            await send_event("content_block_stop", {
                "type": "content_block_stop",
                "index": i,
            })

        await send_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": result.get("stop_reason"),
                "stop_sequence": result.get("stop_sequence"),
            },
            "usage": {"output_tokens": (result.get("usage") or {}).get("output_tokens", 0)},
        })
        await send_event("message_stop", {"type": "message_stop"})

        await response.write_eof()
        return response

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

        # 提前检查：无失败 FC 则直接返回（含记忆工具兜底）
        failed_tc = self._find_failed_function_call(result, clean_body.get("tools", []))
        if failed_tc is None and self._hint_tools:
            failed_tc = self._find_hinted_empty_block(result)
        if failed_tc is None:
            if client_wants_stream:
                return await self._emit_as_sse(result, req)
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
                if client_wants_stream:
                    return await self._emit_as_sse(result, req)
                return web.json_response(result)
            elif self.debug:
                logger.info("Streamify [Layer1]: Claude JSON 参数提取失败，尝试提示注入重试")

        # 提示注入重试
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
                if client_wants_stream:
                    return await self._emit_as_sse(result, req)
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
                    if client_wants_stream:
                        return await self._emit_as_sse(result, req)
                    return web.json_response(result)

        # 重试仍失败，注入失败提示后返回
        _fail_name = failed_tc.get("name", "unknown") if failed_tc else "unknown"
        _inject_fc_failure_text_claude(result, _fail_name)
        if client_wants_stream:
            return await self._emit_as_sse(result, req)
        return web.json_response(result)
