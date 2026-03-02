import json
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import OpenAIFakeNonStream
from ..fc_enhance import OpenAIFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG


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

    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        """将聚合后的 chat.completion 结果重新以 SSE 流格式返回给客户端。"""
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
            if "tool_calls" in msg:
                delta["tool_calls"] = [
                    {**tc, "index": i}
                    for i, tc in enumerate(msg["tool_calls"])
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
            failed_tc = self._find_hinted_empty_tc(result)
        if failed_tc is None:
            if client_wants_stream:
                return await self._emit_as_sse(result, req)
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
                if client_wants_stream:
                    return await self._emit_as_sse(result, req)
                return web.json_response(result)
            elif self.debug:
                logger.info("Streamify [Layer1]: OpenAI JSON 参数提取失败，尝试提示注入重试")

        # 提示注入重试
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
                if client_wants_stream:
                    return await self._emit_as_sse(result, req)
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
                    if client_wants_stream:
                        return await self._emit_as_sse(result, req)
                    return web.json_response(result)

        # 重试仍失败，注入失败提示后返回
        _fail_name = (failed_tc.get("function") or {}).get("name", "unknown") if failed_tc else "unknown"
        _inject_fc_failure_text_openai(result, _fail_name)
        if client_wants_stream:
            return await self._emit_as_sse(result, req)
        return web.json_response(result)
