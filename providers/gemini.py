import json
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import GeminiFakeNonStream
from ..fc_enhance import GeminiFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG, register_handler


def _inject_fc_failure_text_gemini(result: Dict[str, Any], tool_name: str) -> None:
    """向 Gemini 格式响应注入工具调用失败提示。"""
    hint = f"[{_FC_FAILURE_MSG}]"
    for candidate in result.get("candidates", []):
        candidate["content"] = {
            "role": "model",
            "parts": [{"text": hint}],
        }
        candidate["finishReason"] = "STOP"
        return


@register_handler
class GeminiHandler(ProviderHandler, GeminiFakeNonStream, GeminiFCEnhance):

    def _find_hinted_empty_fc(self, response_data: dict) -> Optional[dict]:
        for candidate in (response_data.get("candidates") or []):
            for part in ((candidate.get("content") or {}).get("parts") or []):
                fc = part.get("functionCall")
                if not isinstance(fc, dict):
                    continue
                if fc.get("name", "") not in self._hint_tools:
                    continue
                args = fc.get("args")
                if not args or (isinstance(args, dict) and not args):
                    return fc
        return None

    def matches(self, sub_path: str) -> bool:
        path = sub_path.strip("/")
        return ":generateContent" in path or ":streamGenerateContent" in path

    @staticmethod
    def _payload_has_fc(payload: Dict[str, Any]) -> bool:
        """判断 Gemini SSE 事件中是否包含 functionCall。"""
        for candidate in (payload.get("candidates") or []):
            for part in ((candidate.get("content") or {}).get("parts") or []):
                if isinstance(part.get("functionCall"), dict):
                    return True
        return False

    def _assemble_gemini_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从已解析的 payload 列表组装 generateContent 格式响应。"""
        state = self._init_state()
        for payload in payloads:
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

    # ------------------------------------------------------------------
    # Layer 2 用：将完整结果以单条 Gemini SSE 事件返回
    # ------------------------------------------------------------------
    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)
        await response.write(f"data: {json.dumps(result)}\n\n".encode())
        await response.write_eof()
        return response

    # ------------------------------------------------------------------
    # 流式请求主入口：无 functionCall 的事件直通，有 functionCall 的拦截修复后发出
    # ------------------------------------------------------------------
    async def _handle_stream_fc_hook(
        self,
        req: web.Request,
        stream_path: str,
        clean_body: Dict[str, Any],
        headers: Dict[str, str],
        params: Dict[str, str],
    ) -> web.Response:
        async with self._request(
            "POST", self._build_url(stream_path),
            json=clean_body, headers=headers, params=params,
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

            fc_buffer: List[Dict[str, Any]] = []  # 含 functionCall 的 payloads
            fc_detected = False
            forwarded_text = ""  # 累加已转发的文本，用于 extract_args

            async for _, data in self._iter_sse_events(resp):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    await client.write(f"data: {data}\n\n".encode())
                    continue
                if not isinstance(payload, dict):
                    continue

                if self._payload_has_fc(payload):
                    fc_detected = True
                    # 先转发文本部分
                    for candidate in (payload.get("candidates") or []):
                        text_parts = [p for p in ((candidate.get("content") or {}).get("parts") or []) if "text" in p]
                        if text_parts:
                            logger.info("Streamify [文本转发]: 检测到 FC payload 中的文本，转发 %d 个 text parts", len(text_parts))
                            text_payload = {"candidates": [{"content": {"parts": text_parts, "role": "model"}}]}
                            await client.write(f"data: {json.dumps(text_payload)}\n\n".encode())
                            for p in text_parts:
                                if isinstance(p.get("text"), str):
                                    forwarded_text += p["text"]
                    fc_buffer.append(payload)
                else:
                    # 累加已转发 payload 中的文本
                    for candidate in (payload.get("candidates") or []):
                        for part in ((candidate.get("content") or {}).get("parts") or []):
                            if isinstance(part.get("text"), str):
                                forwarded_text += part["text"]
                    await client.write(f"data: {json.dumps(payload)}\n\n".encode())

        if not fc_detected:
            await client.write_eof()
            return client

        # 组装 FC 结果用于检测
        response_data = self._assemble_gemini_payloads(fc_buffer)
        tools = clean_body.get("tools", [])

        target_fc = self._find_failed_function_call(response_data, tools)
        if target_fc is None and self._hint_tools:
            target_fc = self._find_hinted_empty_fc(response_data)
        is_failed = target_fc is not None

        if self.fc_enhance >= 2 and target_fc is None:
            target_fc = self._find_first_function_call(response_data)
            if target_fc is not None:
                logger.info("Streamify [全部拦截]: 拦截 Gemini 工具 %s，尝试重写参数(流式)", target_fc.get("name", ""))

        if target_fc is None:
            # 无需修复：重放原始 FC 事件
            for payload in fc_buffer:
                await client.write(f"data: {json.dumps(payload)}\n\n".encode())
            await client.write_eof()
            return client

        # 需要修复
        # 提取模型回复文本（已转发文本 + FC 响应中的文本 parts）
        _reply_parts = []
        for candidate in response_data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    _reply_parts.append(part["text"])
        model_reply = forwarded_text + "".join(_reply_parts)
        logger.info("Streamify [文本]: forwarded_text 长度=%d, _reply_parts 长度=%d, model_reply 长度=%d",
                    len(forwarded_text), len("".join(_reply_parts)), len(model_reply))

        # 先发送未转发的文本（在提取参数之前）
        if model_reply and model_reply != forwarded_text:
            remaining_text = model_reply[len(forwarded_text):] if model_reply.startswith(forwarded_text) else model_reply
            if remaining_text:
                logger.info("Streamify [文本]: 发送剩余文本，长度=%d, 内容(前50): %s", len(remaining_text), remaining_text[:50])
                text_payload = {"candidates": [{"content": {"parts": [{"text": remaining_text}], "role": "model"}}]}
                await client.write(f"data: {json.dumps(text_payload)}\n\n".encode())

        fn_name = target_fc.get("name", "")
        extracted = await self._extract_args_as_json(
            clean_body, fn_name, stream_path, headers, params,
            model_reply=model_reply,
        )
        if extracted is not None:
            _before = target_fc.get("args", {})
            self._patch_function_call_args(response_data, fn_name, extracted)
            self._log_fc_modify("gemini", 1, fn_name, _before, extracted,
                hint=self._build_extract_hint(clean_body.get("tools", []), fn_name, model_reply=model_reply),
                context=clean_body.get("contents", []))
            if self.debug:
                logger.info("Streamify [Layer1]: 成功提取 Gemini 工具 %s 参数(流式)", fn_name)
            logger.info("Streamify [debug]: 准备写入 response_data，工具 %s", fn_name)
            await client.write(f"data: {json.dumps(response_data)}\n\n".encode())
            logger.info("Streamify [debug]: 已写入 response_data，准备 write_eof")
            await client.write_eof()
            logger.info("Streamify [debug]: 已完成 write_eof，返回 client")
            return client
        elif self.debug:
            logger.info("Streamify [Layer1]: Gemini JSON 提取失败，重试(流式)")

        if is_failed:
            retry_name = fn_name
            for attempt in range(self.fix_retries):
                if self.debug:
                    logger.info("Streamify: 流式 Gemini FC 修复重试 (%d/%d)", attempt + 1, self.fix_retries)
                current_body = self._inject_hint(clean_body, retry_name)
                async with self._request(
                    "POST", self._build_url(stream_path),
                    json=current_body, headers=headers, params=params,
                ) as retry_resp:
                    if retry_resp.status != 200:
                        break
                    response_data = await self._build_non_stream_response(retry_resp)

                has_failed_retry = self._has_failed_function_call(response_data, tools)
                if not has_failed_retry and self._hint_tools:
                    has_failed_retry = self._find_hinted_empty_fc(response_data) is not None
                if not has_failed_retry:
                    await client.write(f"data: {json.dumps(response_data)}\n\n".encode())
                    await client.write_eof()
                    return client

                retry_failed_fc = self._find_failed_function_call(response_data, tools)
                if retry_failed_fc is None and self._hint_tools:
                    retry_failed_fc = self._find_hinted_empty_fc(response_data)
                if retry_failed_fc is not None:
                    extracted = await self._extract_args_as_json(
                        clean_body, retry_failed_fc.get("name", ""), stream_path, headers, params,
                        model_reply=model_reply,
                    )
                    if extracted is not None:
                        _before = retry_failed_fc.get("args", {})
                        self._patch_function_call_args(response_data, retry_failed_fc.get("name", ""), extracted)
                        self._log_fc_modify("gemini", 1, retry_failed_fc.get("name", ""), _before, extracted,
                            hint=self._build_extract_hint(clean_body.get("tools", []), retry_failed_fc.get("name", ""), model_reply=model_reply),
                            context=clean_body.get("contents", []))
                        await client.write(f"data: {json.dumps(response_data)}\n\n".encode())
                        await client.write_eof()
                        return client

            failed_fc = self._find_failed_function_call(response_data, tools)
            if failed_fc is None and self._hint_tools:
                failed_fc = self._find_hinted_empty_fc(response_data)
            fail_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
            logger.warning("Streamify: 工具 %s 参数在 %d 次重试后仍为空(流式)", fail_name, self.fix_retries)
            _inject_fc_failure_text_gemini(response_data, fail_name)
            await client.write(f"data: {json.dumps(response_data)}\n\n".encode())
            await client.write_eof()
            return client
        else:
            # Level 2 提取失败但原参数非空，重放原始 FC 事件
            logger.info("Streamify [全部拦截]: 工具 %s 参数提取失败，保留原始参数(流式)", fn_name)
            for payload in fc_buffer:
                await client.write(f"data: {json.dumps(payload)}\n\n".encode())
            await client.write_eof()
            return client

    # ------------------------------------------------------------------
    # 主 handle
    # ------------------------------------------------------------------
    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        path = sub_path.strip("/")
        client_wants_stream = ":streamGenerateContent" in path

        if client_wants_stream:
            stream_path = path
        else:
            stream_path = path.replace(":generateContent", ":streamGenerateContent", 1)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        if body and self.debug:
            for i, c in enumerate(body.get("contents", [])):
                for j, p in enumerate(c.get("parts", [])):
                    if "inlineData" in p:
                        d = p["inlineData"]
                        logger.info(
                            "Streamify image debug: contents[%d].parts[%d] mimeType=%s dataLen=%d",
                            i, j, d.get("mimeType", "?"), len(d.get("data", "")),
                        )
                    elif "fileData" in p:
                        logger.info(
                            "Streamify image debug: contents[%d].parts[%d] fileData=%s",
                            i, j, p["fileData"],
                        )

        headers = self._forward_headers(req)
        params = {k: v for k, v in req.query.items()}
        params["alt"] = "sse"

        clean_body = body

        # Layer 2 (Reactive)
        if self.fc_enhance >= 1:
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
                    self._log_fc_modify("gemini", 2, tool_name, "{}", extracted,
                        hint=self._build_extract_hint(body.get("tools", []), tool_name),
                        context=ctx_contents)
                    self._remember_hint_tool(tool_name)
                    args_hint = (
                        f"重要：你必须使用以下参数调用工具 `{tool_name}`：\n"
                        f"{json.dumps(extracted, ensure_ascii=False)}"
                    )
                    clean_body = {**body, "contents": ctx_contents, "_full_contents": body.get("contents", [])}
                    sys_inst = clean_body.get("systemInstruction")
                    if isinstance(sys_inst, dict):
                        parts = list(sys_inst.get("parts", []))
                        parts.append({"text": args_hint})
                        clean_body["systemInstruction"] = {**sys_inst, "parts": parts}
                    else:
                        clean_body["systemInstruction"] = {"parts": [{"text": args_hint}]}
                else:
                    if self.debug:
                        logger.info(
                            "Streamify [Layer2]: Gemini 工具 %s 参数提取失败，继续正常转发",
                            tool_name,
                        )
                    clean_body = {**body, "contents": ctx_contents, "_full_contents": body.get("contents", [])}

        tools = clean_body.get("tools", [])

        if not self.pseudo_non_stream and not client_wants_stream:
            if clean_body is not body:
                return await self._passthrough(req, sub_path,
                                               _body=json.dumps(clean_body).encode())
            return await self._passthrough(req, sub_path)

        # 流式请求：有条件透传 + FC 拦截检测修复
        if client_wants_stream:
            if not clean_body.get("tools") or self.fc_enhance == 0:
                return await self._proxy_stream(req, stream_path, clean_body, headers, params)
            return await self._handle_stream_fc_hook(req, stream_path, clean_body, headers, params)

        # 非流式请求：内部转流收集，FC 检测修复后返回 JSON
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

        if self.fc_enhance == 0:
            return web.json_response(response_data)

        target_fc = self._find_failed_function_call(response_data, tools)
        if target_fc is None and self._hint_tools:
            target_fc = self._find_hinted_empty_fc(response_data)
        is_failed = target_fc is not None

        if self.fc_enhance >= 2 and target_fc is None:
            target_fc = self._find_first_function_call(response_data)
            if target_fc is not None:
                logger.info("Streamify [全部拦截]: 拦截 Gemini 工具 %s，尝试重写参数", target_fc.get("name", ""))

        if target_fc is None:
            return web.json_response(response_data)

        # 提取模型回复文本
        _reply_parts = []
        for candidate in response_data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    _reply_parts.append(part["text"])
        model_reply = "".join(_reply_parts)

        function_name = target_fc.get("name", "")
        extracted = await self._extract_args_as_json(
            clean_body, function_name, stream_path, headers, params,
            model_reply=model_reply,
        )
        if extracted is not None:
            _before = target_fc.get("args", {})
            self._patch_function_call_args(response_data, function_name, extracted)
            self._log_fc_modify("gemini", 1, function_name, _before, extracted,
                hint=self._build_extract_hint(clean_body.get("tools", []), function_name, model_reply=model_reply),
                context=clean_body.get("contents", []))
            if self.debug:
                logger.info(
                    "Streamify [Layer1]: 成功提取 Gemini 工具 %s 的参数: %s",
                    function_name, extracted,
                )
            return web.json_response(response_data)
        elif self.debug:
            logger.info("Streamify [Layer1]: Gemini JSON 参数提取失败，尝试提示注入重试")

        if is_failed:
            for attempt in range(self.fix_retries):
                if self.debug:
                    logger.info(
                        "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                        attempt + 1, self.fix_retries,
                    )
                current_body = self._inject_hint(clean_body, function_name)
                if self.debug:
                    schema = self._extract_tool_schema(clean_body.get("tools", []), function_name)
                    logger.info(
                        "Streamify debug: retry_tool_name=%r, schema_found=%s",
                        function_name, schema is not None,
                    )
                    injected_sys = current_body.get("systemInstruction")
                    if injected_sys:
                        parts = injected_sys.get("parts", [])
                        if parts:
                            hint_text = parts[-1].get("text", "")
                            logger.info(
                                "Streamify debug: 注入的 hint (%d chars): %s",
                                len(hint_text), hint_text[:1000],
                            )
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

                has_failed = self._has_failed_function_call(response_data, tools)
                if not has_failed and self._hint_tools:
                    has_failed = self._find_hinted_empty_fc(response_data) is not None
                if not has_failed:
                    return web.json_response(response_data)
                retry_failed_fc = self._find_failed_function_call(response_data, tools)
                if retry_failed_fc is None and self._hint_tools:
                    retry_failed_fc = self._find_hinted_empty_fc(response_data)
                if retry_failed_fc is not None:
                    _rp = []
                    for cand in response_data.get("candidates", []):
                        for pt in (cand.get("content") or {}).get("parts", []):
                            if isinstance(pt.get("text"), str):
                                _rp.append(pt["text"])
                    _retry_reply = "".join(_rp)
                    extracted = await self._extract_args_as_json(
                        clean_body, retry_failed_fc.get("name", ""), stream_path, headers, params,
                        model_reply=_retry_reply,
                    )
                    if extracted is not None:
                        _before = retry_failed_fc.get("args", {})
                        self._patch_function_call_args(
                            response_data, retry_failed_fc.get("name", ""), extracted
                        )
                        self._log_fc_modify("gemini", 1, retry_failed_fc.get("name", ""), _before, extracted,
                            hint=self._build_extract_hint(clean_body.get("tools", []), retry_failed_fc.get("name", ""), model_reply=_retry_reply),
                            context=clean_body.get("contents", []))
                        if self.debug:
                            logger.info(
                                "Streamify [Layer1]: 成功提取 Gemini 工具 %s 的参数: %s",
                                retry_failed_fc.get("name"), extracted,
                            )
                        return web.json_response(response_data)

            failed_fc = self._find_failed_function_call(response_data, tools)
            if failed_fc is None and self._hint_tools:
                failed_fc = self._find_hinted_empty_fc(response_data)
            fail_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
            logger.warning(
                "Streamify: 工具 %s 参数在 %d 次重试后仍为空，返回原始结果",
                fail_name, self.fix_retries,
            )
            _inject_fc_failure_text_gemini(response_data, fail_name)
        elif not is_failed:
            logger.info("Streamify [全部拦截]: 工具 %s 参数提取失败，保留原始参数", function_name)

        return web.json_response(response_data)
