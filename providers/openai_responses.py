import json
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import OpenAIResponsesFakeNonStream
from ..fc_enhance import OpenAIResponsesFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG, register_handler


def _inject_fc_failure_text_responses(result: Dict[str, Any], tool_name: str) -> None:
    """向 Responses API 格式响应注入工具调用失败提示。"""
    hint = f"[{_FC_FAILURE_MSG}]"
    result["output"] = [{
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": hint}],
    }]
    result["status"] = "completed"


@register_handler
class OpenAIResponsesHandler(ProviderHandler, OpenAIResponsesFakeNonStream, OpenAIResponsesFCEnhance):
    ENDPOINT = "v1/responses"

    def _find_hinted_empty_fc(self, response_data: dict) -> Optional[dict]:
        for item in (response_data.get("output") or []):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            if item.get("name", "") not in self._hint_tools:
                continue
            args = item.get("arguments", "")
            try:
                parsed = json.loads(args) if isinstance(args, str) and args else {}
            except Exception:
                parsed = {}
            if not parsed:
                return item
        return None

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    # ------------------------------------------------------------------
    # 将完整结果以 Responses API SSE 格式返回给流式客户端
    # ------------------------------------------------------------------
    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)

        async def send(event_type: str, data: Dict[str, Any]) -> None:
            await response.write(f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode())

        await send("response.completed", {"type": "response.completed", "response": result})
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    # ------------------------------------------------------------------
    # 流式 FC hook
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

            fc_detected = False
            fc_buffer: List[str] = []  # 原始 SSE 行
            completed_data: Optional[Dict[str, Any]] = None
            # 跟踪哪些 output_item 是 function_call（按 item_id）
            fc_item_ids: set = set()

            async for event_name, data in self._iter_sse_events(resp):
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    sse = f"event: {event_name}\ndata: {data}\n\n" if event_name else f"data: {data}\n\n"
                    await client.write(sse.encode())
                    continue
                if not isinstance(payload, dict):
                    continue

                event_type = event_name or payload.get("type")
                sse = f"event: {event_name}\ndata: {json.dumps(payload)}\n\n" if event_name else f"data: {json.dumps(payload)}\n\n"

                # 判断事件是否与 function_call 相关
                is_fc_event = False

                if event_type == "response.output_item.added":
                    item = payload.get("item") or {}
                    if item.get("type") == "function_call":
                        fc_detected = True
                        is_fc_event = True
                        item_id = item.get("id", "")
                        if item_id:
                            fc_item_ids.add(item_id)

                elif event_type == "response.output_item.done":
                    item = payload.get("item") or {}
                    if item.get("type") == "function_call" or item.get("id", "") in fc_item_ids:
                        is_fc_event = True

                elif event_type in ("response.function_call_arguments.delta",
                                    "response.function_call_arguments.done"):
                    is_fc_event = True

                elif event_type == "response.completed":
                    response_obj = payload.get("response")
                    completed_data = response_obj if isinstance(response_obj, dict) else payload
                    if fc_detected:
                        is_fc_event = True

                if is_fc_event:
                    fc_buffer.append(sse)
                else:
                    await client.write(sse.encode())

        if not fc_detected or completed_data is None:
            if not fc_detected:
                await client.write(b"data: [DONE]\n\n")
                await client.write_eof()
                return client
            # fc_detected 但没有 completed_data：重放缓冲
            for line in fc_buffer:
                await client.write(line.encode())
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client

        tools = clean_body.get("tools") or []
        failed = self._find_failed_function_call(completed_data, tools)
        if failed is None and self._hint_tools:
            failed = self._find_hinted_empty_fc(completed_data)

        target_fc = failed  # reuse the already-found failed FC
        is_failed = target_fc is not None

        if self.fc_enhance >= 2 and target_fc is None:
            target_fc = self._find_first_function_call(completed_data)
            if target_fc is not None:
                logger.info("Streamify [全部拦截]: 拦截 Responses 工具 %s，尝试重写参数(流式)", target_fc.get("name", ""))

        if target_fc is None:
            for line in fc_buffer:
                await client.write(line.encode())
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client

        # 需要修复
        result = completed_data
        model_reply = self._extract_text_from_response(completed_data)

        fn_name = target_fc.get("name", "")
        extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers,
                                                     model_reply=model_reply)
        if extracted is not None:
            _before = target_fc.get("arguments", "{}")
            self._patch_function_call_args(result, fn_name, extracted)
            self._log_fc_modify("openai_responses", 1, fn_name, _before, extracted,
                hint=self._build_extract_hint(clean_body.get("tools", []), fn_name, model_reply=model_reply),
                context=clean_body.get("input", []))
            if self.debug:
                logger.info("Streamify [Layer1]: 成功提取 Responses 工具 %s 参数(流式)", fn_name)
            completed_evt = json.dumps({"type": "response.completed", "response": result})
            await client.write(f"event: response.completed\ndata: {completed_evt}\n\n".encode())
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client
        elif self.debug:
            logger.info("Streamify [Layer1]: Responses JSON 提取失败，重试(流式)")

        if is_failed:
            retry_name = fn_name
            for attempt in range(self.fix_retries):
                if self.debug:
                    logger.info("Streamify: 流式 Responses FC 修复重试 (%d/%d)", attempt + 1, self.fix_retries)
                current_body = self._inject_hint(clean_body, retry_name)
                current_body["stream"] = True
                async with self._request(
                    "POST", self._build_url(sub_path),
                    json=current_body, headers=headers, params=req.query,
                ) as retry_resp:
                    if retry_resp.status != 200:
                        break
                    result = await self._build_non_stream_response(retry_resp)

                failed_fc = self._find_failed_function_call(result, tools)
                if failed_fc is None and self._hint_tools:
                    failed_fc = self._find_hinted_empty_fc(result)
                if failed_fc is None:
                    completed_evt = json.dumps({"type": "response.completed", "response": result})
                    await client.write(f"event: response.completed\ndata: {completed_evt}\n\n".encode())
                    await client.write(b"data: [DONE]\n\n")
                    await client.write_eof()
                    return client

                fn2 = failed_fc.get("name", "")
                extracted = await self._extract_args_as_json(clean_body, fn2, sub_path, headers,
                                                             model_reply=model_reply)
                if extracted is not None:
                    _before = failed_fc.get("arguments", "{}")
                    self._patch_function_call_args(result, fn2, extracted)
                    self._log_fc_modify("openai_responses", 1, fn2, _before, extracted,
                        hint=self._build_extract_hint(clean_body.get("tools", []), fn2, model_reply=model_reply),
                        context=clean_body.get("input", []))
                    completed_evt = json.dumps({"type": "response.completed", "response": result})
                    await client.write(f"event: response.completed\ndata: {completed_evt}\n\n".encode())
                    await client.write(b"data: [DONE]\n\n")
                    await client.write_eof()
                    return client

            fail_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
            logger.warning("Streamify: 工具 %s 参数在 %d 次重试后仍为空(流式Responses)", fail_name, self.fix_retries)
            _inject_fc_failure_text_responses(result, fail_name)
            completed_evt = json.dumps({"type": "response.completed", "response": result})
            await client.write(f"event: response.completed\ndata: {completed_evt}\n\n".encode())
            await client.write(b"data: [DONE]\n\n")
            await client.write_eof()
            return client
        else:
            # Level 2 提取失败但原参数非空，重放原始 FC 事件
            logger.info("Streamify [全部拦截]: 工具 %s 参数提取失败，保留原始参数(流式)", fn_name)
            for line in fc_buffer:
                await client.write(line.encode())
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

        # Layer 2 (Reactive)
        if self.fc_enhance >= 1:
            error_info = self._find_tool_error_in_request(body)
            if error_info is not None:
                tool_name, call_id, fc_idx = error_info
                input_items = body.get("input", [])
                ctx_input = input_items[:fc_idx] if isinstance(input_items, list) else []
                extracted = await self._extract_args_as_json(
                    body, tool_name, sub_path, headers,
                    input_override=ctx_input,
                )
                if extracted is not None:
                    if self.debug:
                        logger.info("Streamify [Layer2]: 修正 Responses 工具 %s 的参数: %s", tool_name, extracted)
                    self._log_fc_modify("openai_responses", 2, tool_name, "{}", extracted,
                        hint=self._build_extract_hint(body.get("tools", []), tool_name),
                        context=ctx_input)
                    self._remember_hint_tool(tool_name)
                    result = self._build_corrected_tool_response(call_id, tool_name, extracted, body.get("model", ""))
                    if client_wants_stream:
                        return await self._emit_as_sse(result, req)
                    return web.json_response(result)
                elif self.debug:
                    logger.info("Streamify [Layer2]: Responses 工具 %s 参数提取失败，继续正常转发", tool_name)
                clean_body = {**body, "input": ctx_input}

        if not self.pseudo_non_stream and not client_wants_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path, _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        # 流式客户端
        if client_wants_stream:
            if not clean_body.get("tools") or self.fc_enhance == 0:
                return await self._proxy_stream(req, sub_path, clean_body, headers)
            return await self._handle_stream_fc_hook(req, sub_path, clean_body, headers)

        # 非流式：内部转流收集
        clean_body["stream"] = True
        async with self._request(
            "POST", self._build_url(sub_path),
            json=clean_body, headers=headers, params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )
            result = await self._build_non_stream_response(resp)

        if "error" in result and "output" not in result:
            return web.json_response(result, status=502)

        if self.fc_enhance == 0:
            return web.json_response(result)

        tools = clean_body.get("tools") or []
        target_fc = self._find_failed_function_call(result, tools)
        if target_fc is None and self._hint_tools:
            target_fc = self._find_hinted_empty_fc(result)
        is_failed = target_fc is not None

        if self.fc_enhance >= 2 and target_fc is None:
            target_fc = self._find_first_function_call(result)
            if target_fc is not None:
                logger.info("Streamify [全部拦截]: 拦截 Responses 工具 %s，尝试重写参数", target_fc.get("name", ""))

        if target_fc is None:
            return web.json_response(result)

        fn_name = target_fc.get("name", "")
        model_reply = self._extract_text_from_response(result)
        extracted = await self._extract_args_as_json(clean_body, fn_name, sub_path, headers,
                                                     model_reply=model_reply)
        if extracted is not None:
            _before = target_fc.get("arguments", "{}")
            self._patch_function_call_args(result, fn_name, extracted)
            self._log_fc_modify("openai_responses", 1, fn_name, _before, extracted,
                hint=self._build_extract_hint(clean_body.get("tools", []), fn_name, model_reply=model_reply),
                context=clean_body.get("input", []))
            if self.debug:
                logger.info("Streamify [Layer1]: 成功提取 Responses 工具 %s 的参数: %s", fn_name, extracted)
            return web.json_response(result)
        elif self.debug:
            logger.info("Streamify [Layer1]: Responses JSON 参数提取失败，尝试提示注入重试")

        if is_failed:
            for attempt in range(self.fix_retries):
                if self.debug:
                    logger.info("Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)", attempt + 1, self.fix_retries)
                current_body = self._inject_hint(clean_body, fn_name)
                current_body["stream"] = True
                async with self._request(
                    "POST", self._build_url(sub_path),
                    json=current_body, headers=headers, params=req.query,
                ) as resp:
                    if resp.status != 200:
                        return web.Response(
                            status=resp.status,
                            headers=self._response_headers(resp),
                            text=await resp.text(),
                        )
                    result = await self._build_non_stream_response(resp)

                failed_fc = self._find_failed_function_call(result, tools)
                if failed_fc is None and self._hint_tools:
                    failed_fc = self._find_hinted_empty_fc(result)
                if failed_fc is None:
                    return web.json_response(result)
                fn2 = failed_fc.get("name", "")
                _retry_reply = self._extract_text_from_response(result)
                extracted = await self._extract_args_as_json(clean_body, fn2, sub_path, headers,
                                                             model_reply=_retry_reply)
                if extracted is not None:
                    _before = failed_fc.get("arguments", "{}")
                    self._patch_function_call_args(result, fn2, extracted)
                    self._log_fc_modify("openai_responses", 1, fn2, _before, extracted,
                        hint=self._build_extract_hint(clean_body.get("tools", []), fn2, model_reply=_retry_reply),
                        context=clean_body.get("input", []))
                    if self.debug:
                        logger.info("Streamify [Layer1]: 成功提取 Responses 工具 %s 的参数: %s", fn2, extracted)
                    return web.json_response(result)

            fail_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
            logger.warning("Streamify: 工具 %s 参数在 %d 次重试后仍为空", fail_name, self.fix_retries)
            _inject_fc_failure_text_responses(result, fail_name)
        elif not is_failed:
            logger.info("Streamify [全部拦截]: 工具 %s 参数提取失败，保留原始参数", fn_name)

        return web.json_response(result)
