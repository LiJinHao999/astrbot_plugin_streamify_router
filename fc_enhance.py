import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .fake_non_stream import _EMPTY_ARGS_HINT


def _is_tool_execution_error(content: str) -> bool:
    """判断工具执行结果是否为错误消息（三个 Mixin 共用）。"""
    if not content:
        return False
    lower = content.lower()
    return (
        content.startswith("error:") or
        "parameter mismatch" in lower or
        "missing 1 required" in lower or
        "missing required positional argument" in lower
    )


class OpenAIFCEnhance:
    """OpenAI FC enhancement mixin."""

    @staticmethod
    def _tool_has_required_params(tools: List[Dict[str, Any]], tool_name: str) -> bool:
        """检查工具 schema 中是否存在必填参数（OpenAI 格式）。"""
        for tool in tools:
            fn = (tool.get("function") or {})
            if fn.get("name") == tool_name:
                required = (fn.get("parameters") or {}).get("required", [])
                return bool(required)
        return False

    @staticmethod
    def _find_failed_function_call(
        response_data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        tools = tools or []
        for choice in response_data.get("choices", []):
            message = (choice.get("message") or {})
            for tc in (message.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tool_name = fn.get("name", "")
                args = fn.get("arguments", "")
                if not args or args == "{}":
                    if OpenAIFCEnhance._tool_has_required_params(tools, tool_name):
                        return tc
                    continue
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    return tc
        return None

    @staticmethod
    def _find_tool_error_in_request(
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, str, int]]:
        """在请求消息历史中查找工具执行错误。

        返回 (tool_name, tool_call_id, assistant_msg_index) 或 None。
        """
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not _is_tool_execution_error(content):
                continue
            tool_call_id = msg.get("tool_call_id", "")
            for j in range(i - 1, -1, -1):
                asst = messages[j]
                if not isinstance(asst, dict):
                    continue
                if asst.get("role") == "assistant":
                    for tc in (asst.get("tool_calls") or []):
                        if not isinstance(tc, dict):
                            continue
                        if tc.get("id") == tool_call_id or not tool_call_id:
                            tool_name = (tc.get("function") or {}).get("name", "")
                            return (tool_name, tool_call_id, j)
                    break
        return None

    @staticmethod
    def _build_corrected_tool_response(
        tool_call_id: str, tool_name: str, args: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        """构造带正确参数的合成 chat.completion 响应。"""
        return {
            "id": f"chatcmpl-proxy-fix-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(args)},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }

    @staticmethod
    def _patch_function_call_args(
        response_data: Dict[str, Any], function_name: str, args: Dict[str, Any]
    ) -> None:
        for choice in response_data.get("choices", []):
            message = (choice.get("message") or {})
            for tc in (message.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if fn.get("name") == function_name:
                    fn["arguments"] = json.dumps(args)
                    return

    async def _extract_args_as_json(
        self,
        original_body: Dict[str, Any],
        function_name: str,
        sub_path: str,
        headers: Dict[str, str],
        messages_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            fn = (tool.get("function") or {})
            if fn.get("name") == function_name:
                func_desc = fn.get("description", "")
                func_schema = fn.get("parameters", {})
                break

        tool_system = (
            f"你是一个参数提取助手。用户正在使用工具 `{function_name}`。\n"
            f"工具说明：{func_desc}\n"
            f"参数 schema：{json.dumps(func_schema, ensure_ascii=False)}\n"
            f"根据对话内容，只输出调用该工具所需的 JSON 参数对象，不要包含任何其他文字。"
        )

        source_messages = (
            messages_override if messages_override is not None
            else original_body.get("messages", [])
        )
        messages = [{"role": "system", "content": tool_system}]
        for msg in source_messages:
            if isinstance(msg, dict) and msg.get("role") != "system":
                messages.append(msg)

        extract_body: Dict[str, Any] = {
            "model": original_body.get("model", ""),
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": True,
        }

        try:
            async with self._request(  # type: ignore[attr-defined]
                "POST",
                self._build_url(sub_path),  # type: ignore[attr-defined]
                json=extract_body,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return None
                result = await self._build_non_stream_response(resp)  # type: ignore[attr-defined]
                choices = result.get("choices", [])
                if not choices:
                    return None
                content = (choices[0].get("message") or {}).get("content", "")
                if not content:
                    return None
                content = content.strip()
                if content.startswith("```"):
                    lines = content.splitlines()
                    content = "\n".join(
                        lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                    )
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _inject_hint(body: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(body)
        messages = list(body.get("messages", []))
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                existing = msg.get("content", "")
                new_msg = dict(msg)
                new_msg["content"] = f"{existing}\n\n{_EMPTY_ARGS_HINT}" if existing else _EMPTY_ARGS_HINT
                messages[i] = new_msg
                break
        else:
            messages.insert(0, {"role": "system", "content": _EMPTY_ARGS_HINT})
        body["messages"] = messages
        return body


class ClaudeFCEnhance:
    """Claude FC enhancement mixin."""

    @staticmethod
    def _tool_has_required_params(tools: List[Dict[str, Any]], tool_name: str) -> bool:
        """检查工具 schema 中是否存在必填参数（Claude 格式）。"""
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("name") == tool_name:
                required = (tool.get("input_schema") or {}).get("required", [])
                return bool(required)
        return False

    @staticmethod
    def _find_failed_function_call(
        response_data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        tools = tools or []
        for block in response_data.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and not block.get("input"):
                tool_name = block.get("name", "")
                if ClaudeFCEnhance._tool_has_required_params(tools, tool_name):
                    return block
        return None

    @staticmethod
    def _find_tool_error_in_request(
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, str, int]]:
        """在请求消息历史中查找工具执行错误（Claude 格式）。

        返回 (tool_name, tool_use_id, assistant_msg_index) 或 None。
        """
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    text = " ".join(
                        c.get("text", "") for c in result_content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                elif isinstance(result_content, str):
                    text = result_content
                else:
                    text = ""
                if not _is_tool_execution_error(text):
                    continue
                tool_use_id = block.get("tool_use_id", "")
                for j in range(i - 1, -1, -1):
                    asst = messages[j]
                    if not isinstance(asst, dict):
                        continue
                    if asst.get("role") == "assistant":
                        asst_content = asst.get("content", [])
                        if not isinstance(asst_content, list):
                            break
                        for b2 in asst_content:
                            if not isinstance(b2, dict) or b2.get("type") != "tool_use":
                                continue
                            if b2.get("id") == tool_use_id or not tool_use_id:
                                return (b2.get("name", ""), tool_use_id, j)
                        break
        return None

    @staticmethod
    def _build_corrected_tool_response(
        tool_use_id: str, tool_name: str, args: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        """构造带正确参数的合成 Claude Messages API 响应。"""
        return {
            "id": f"msg_proxy_fix_{int(time.time() * 1000)}",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": args,
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    @staticmethod
    def _patch_function_call_args(
        response_data: Dict[str, Any], function_name: str, args: Dict[str, Any]
    ) -> None:
        for block in response_data.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == function_name:
                block["input"] = args
                return

    async def _extract_args_as_json(
        self,
        original_body: Dict[str, Any],
        function_name: str,
        sub_path: str,
        headers: Dict[str, str],
        messages_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        func_desc = ""
        input_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            if not isinstance(tool, dict):
                continue
            if tool.get("name") == function_name:
                func_desc = tool.get("description", "")
                input_schema = tool.get("input_schema", {})
                break

        tool_system = (
            f"你是一个参数提取助手。用户正在使用工具 `{function_name}`。\n"
            f"工具说明：{func_desc}\n"
            f"参数 schema：{json.dumps(input_schema, ensure_ascii=False)}\n"
            f"根据对话内容，只输出调用该工具所需的 JSON 参数对象，不要包含任何其他文字。"
        )

        source_messages = (
            messages_override if messages_override is not None
            else original_body.get("messages", [])
        )
        extract_body: Dict[str, Any] = {
            "model": original_body.get("model", ""),
            "max_tokens": original_body.get("max_tokens", 1024),
            "system": tool_system,
            "messages": source_messages,
            "stream": True,
        }

        try:
            async with self._request(  # type: ignore[attr-defined]
                "POST",
                self._build_url(sub_path),  # type: ignore[attr-defined]
                json=extract_body,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return None
                result = await self._build_non_stream_response(resp)  # type: ignore[attr-defined]
                for block in result.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            if text.startswith("```"):
                                lines = text.splitlines()
                                text = "\n".join(
                                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                                )
                            parsed = json.loads(text)
                            return parsed if isinstance(parsed, dict) else None
                return None
        except Exception:
            return None

    @staticmethod
    def _inject_hint(body: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(body)
        existing = body.get("system", "")
        if isinstance(existing, str):
            body["system"] = f"{existing}\n\n{_EMPTY_ARGS_HINT}" if existing else _EMPTY_ARGS_HINT
        elif isinstance(existing, list):
            new_system = list(existing)
            new_system.append({"type": "text", "text": _EMPTY_ARGS_HINT})
            body["system"] = new_system
        else:
            body["system"] = _EMPTY_ARGS_HINT
        return body


class GeminiFCEnhance:
    """Gemini FC enhancement mixin."""

    @staticmethod
    def _tool_has_required_params(tools: List[Dict[str, Any]], tool_name: str) -> bool:
        """检查工具 schema 中是否存在必填参数（Gemini 格式）。"""
        for tool in tools:
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == tool_name:
                    required = (fd.get("parameters") or {}).get("required", [])
                    return bool(required)
        return False

    @staticmethod
    def _has_failed_function_call(
        response_data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        tools = tools or []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict) and not fc.get("args"):
                    if GeminiFCEnhance._tool_has_required_params(tools, fc.get("name", "")):
                        return True
        return False

    @staticmethod
    def _find_failed_function_call(
        response_data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        tools = tools or []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict) and not fc.get("args"):
                    if GeminiFCEnhance._tool_has_required_params(tools, fc.get("name", "")):
                        return fc
        return None

    @staticmethod
    def _find_tool_error_in_request(
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, int]]:
        """在请求 contents 中查找工具执行错误（Gemini 格式）。

        返回 (tool_name, error_content_index) 或 None。
        """
        contents = body.get("contents", [])
        for i in range(len(contents) - 1, -1, -1):
            item = contents[i]
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            parts = item.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                func_resp = part.get("functionResponse")
                if not isinstance(func_resp, dict):
                    continue
                response = func_resp.get("response", {})
                content = response.get("content", "")
                if isinstance(content, str) and _is_tool_execution_error(content):
                    return (func_resp.get("name", ""), i)
        return None

    @staticmethod
    def _build_corrected_tool_response(
        tool_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构造带正确参数的合成 Gemini 响应。"""
        return {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": tool_name, "args": args}}],
                },
                "finishReason": "STOP",
            }],
        }

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
        contents_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """构造聚焦请求：只含工具说明 + 参数 schema + 对话上下文，让模型输出 JSON 参数。"""
        func_desc = ""
        func_params_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == function_name:
                    func_desc = fd.get("description", "")
                    func_params_schema = fd.get("parameters", {})
                    break

        tool_system = (
            f"你是一个参数提取助手。用户正在使用工具 `{function_name}`。\n"
            f"工具说明：{func_desc}\n"
            f"参数 schema：{json.dumps(func_params_schema, ensure_ascii=False)}\n"
            f"根据对话内容，只输出调用该工具所需的 JSON 参数对象，不要包含任何其他文字。"
        )

        contents = list(
            contents_override if contents_override is not None
            else original_body.get("contents", [])
        )

        gen_cfg = dict(original_body.get("generationConfig") or {})
        gen_cfg["responseMimeType"] = "application/json"

        extract_body: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": tool_system}]},
            "generationConfig": gen_cfg,
            # 不传 tools，避免模型再次发出空的 functionCall
        }

        try:
            async with self._request(  # type: ignore[attr-defined]
                "POST",
                self._build_url(stream_path),  # type: ignore[attr-defined]
                json=extract_body,
                headers=headers,
                params=params,
            ) as resp:
                if resp.status != 200:
                    return None
                text_parts: List[str] = []
                async for _event, data in self._iter_sse_events(resp):  # type: ignore[attr-defined]
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
                    text = "\n".join(
                        lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                    )
                result = json.loads(text)
                return result if isinstance(result, dict) else None
        except Exception:
            return None

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
