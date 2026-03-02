import json
from typing import Any, Dict, List, Optional

from aiohttp import web

from astrbot.api import logger

from ..fake_non_stream import GeminiFakeNonStream
from ..fc_enhance import GeminiFCEnhance
from .base import ProviderHandler, _FC_FAILURE_MSG


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

    async def _emit_as_sse(self, result: Dict[str, Any], req: web.Request) -> web.StreamResponse:
        """将聚合后的 generateContent 结果重新以 SSE 流格式返回给客户端。"""
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(req)
        await response.write(f"data: {json.dumps(result)}\n\n".encode())
        await response.write_eof()
        return response

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        path = sub_path.strip("/")
        client_wants_stream = ":streamGenerateContent" in path

        # 推导上游 streamGenerateContent 路径
        if client_wants_stream:
            stream_path = path
        else:
            stream_path = path.replace(":generateContent", ":streamGenerateContent", 1)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        # 临时调试：打印 contents 中的图片 part 结构（不打印实际数据）
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
                    self._remember_hint_tool(tool_name)
                    # 不返回合成响应（缺 thoughtSignature 会导致后续请求 400），
                    # 而是把提取到的参数注入上下文，让模型自己生成带签名的 function call
                    args_hint = (
                        f"重要：你必须使用以下参数调用工具 `{tool_name}`：\n"
                        f"{json.dumps(extracted, ensure_ascii=False)}"
                    )
                    clean_body = {**body, "contents": ctx_contents}
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
                    # 提取失败时仅清理 contents（不覆盖已注入 args_hint 的 clean_body）
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

        has_failed = self._has_failed_function_call(response_data, tools)
        if not has_failed and self._hint_tools:
            has_failed = self._find_hinted_empty_fc(response_data) is not None
        if not has_failed:
            if client_wants_stream:
                return await self._emit_as_sse(response_data, req)
            return web.json_response(response_data)

        # Layer 1 (Proactive): 对照 schema 提取参数
        if self.extract_args:
            failed_fc = self._find_failed_function_call(response_data, tools)
            if failed_fc is None and self._hint_tools:
                failed_fc = self._find_hinted_empty_fc(response_data)
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
                    if client_wants_stream:
                        return await self._emit_as_sse(response_data, req)
                    return web.json_response(response_data)
                elif self.debug:
                    logger.info("Streamify [Layer1]: Gemini JSON 参数提取失败，尝试提示注入重试")

        retry_fc = self._find_failed_function_call(response_data, tools)
        if retry_fc is None and self._hint_tools:
            retry_fc = self._find_hinted_empty_fc(response_data)
        retry_tool_name = retry_fc.get("name", "") if retry_fc else ""
        for attempt in range(self.fix_retries):
            if self.debug:
                logger.info(
                    "Streamify: 检测到空工具参数，注入提示后重试 (%d/%d)",
                    attempt + 1, self.fix_retries,
                )
            current_body = self._inject_hint(clean_body, retry_tool_name)
            if self.debug:
                schema = self._extract_tool_schema(clean_body.get("tools", []), retry_tool_name)
                logger.info(
                    "Streamify debug: retry_tool_name=%r, schema_found=%s",
                    retry_tool_name, schema is not None,
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
                if client_wants_stream:
                    return await self._emit_as_sse(response_data, req)
                return web.json_response(response_data)
            if self.extract_args:
                retry_failed_fc = self._find_failed_function_call(response_data, tools)
                if retry_failed_fc is None and self._hint_tools:
                    retry_failed_fc = self._find_hinted_empty_fc(response_data)
                if retry_failed_fc is not None:
                    extracted = await self._extract_args_as_json(
                        clean_body, retry_failed_fc.get("name", ""), stream_path, headers, params
                    )
                    if extracted is not None:
                        self._patch_function_call_args(
                            response_data, retry_failed_fc.get("name", ""), extracted
                        )
                        if self.debug:
                            logger.info(
                                "Streamify [Layer1]: 成功提取 Gemini 工具 %s 的参数: %s",
                                retry_failed_fc.get("name"), extracted,
                            )
                        if client_wants_stream:
                            return await self._emit_as_sse(response_data, req)
                        return web.json_response(response_data)

        failed_fc = self._find_failed_function_call(response_data, tools)
        if failed_fc is None and self._hint_tools:
            failed_fc = self._find_hinted_empty_fc(response_data)
        function_name = failed_fc.get("name", "unknown") if failed_fc else "unknown"
        logger.warning(
            "Streamify: 工具 %s 参数在 %d 次重试后仍为空，返回原始结果",
            function_name, self.fix_retries,
        )
        _inject_fc_failure_text_gemini(response_data, function_name)
        if client_wants_stream:
            return await self._emit_as_sse(response_data, req)
        return web.json_response(response_data)
