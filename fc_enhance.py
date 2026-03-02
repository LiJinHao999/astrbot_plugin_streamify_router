import json
import re
import time
from typing import Any, Dict, List, Optional, Pattern, Tuple
from astrbot.api import logger
from .fake_non_stream import _EMPTY_ARGS_HINT

# 默认工具错误识别正则列表（用户可在配置中覆盖）
_DEFAULT_TOOL_ERROR_PATTERNS: List[str] = [
    r"(?i)^error:",
    r"(?i)parameter mismatch",
    r"(?i)missing \d+ required",
    r"(?i)missing required positional argument",
    r"不能.*为空",
    r"缺少.*参数",
    r"(?i)cannot be empty",
    r"(?i)must not be empty",
    r"(?i)\bempty\b",
]


def _trim_messages_by_turns(
    messages: List[Dict[str, Any]], turns: int, user_role: str = "user"
) -> List[Dict[str, Any]]:
    """保留最近 N 轮对话（每轮以 user 消息为起点）。turns=0 不传入对话。"""
    if not messages:
        return messages
    if turns == 0:
        return []
    if turns < 0:
        return messages
    count = 0
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == user_role:
            count += 1
            if count >= turns:
                cut = i
                break
    return messages[cut:]


def _trim_contents_by_turns(
    contents: List[Dict[str, Any]], turns: int, user_role: str = "user"
) -> List[Dict[str, Any]]:
    """保留最近 N 轮对话（Gemini contents 格式，无 system）。turns=0 不传入对话。"""
    if not contents:
        return contents
    if turns == 0:
        return []
    if turns < 0:
        return contents
    count = 0
    cut = 0
    for i in range(len(contents) - 1, -1, -1):
        if isinstance(contents[i], dict) and contents[i].get("role") == user_role:
            count += 1
            if count >= turns:
                cut = i
                break
    return contents[cut:]


_EXTRACT_PROMPT_TEMPLATE = (
    "你是一个参数提取助手。用户正在使用工具 `{function_name}`。\n"
    "工具说明：{func_desc}\n"
    "参数 schema：{func_schema}\n"
    "{model_reply_section}"
    "{tool_history_section}"
    "请仔细分析以下完整对话上下文（包括用户消息、助手回复），"
    "从中提取调用该工具所需的参数。\n"
    "你只输出并且必须输出 JSON 参数对象。"
)


def _strip_system_tags(text: str) -> str:
    """去除 <system_reminder>；从 <conversation_scene> 中仅提取 <content> 的值，无 <content> 则保留原文。"""
    text = re.sub(r'<system_reminder>.*?</system_reminder>', '', text, flags=re.DOTALL)

    def _extract_content(m: re.Match) -> str:
        inner = m.group(1)
        cm = re.search(r'<content>(.*?)</content>', inner, re.DOTALL)
        if cm:
            return cm.group(1).strip()
        return m.group(0)  # 无 <content> 标签，保留原文

    text = re.sub(r'<conversation_scene>(.*?)</conversation_scene>', _extract_content, text, flags=re.DOTALL)
    return text


def _extract_text_from_openai_msg(msg: Dict[str, Any]) -> str:
    """从 OpenAI 格式消息中提取纯文本，去除 system tags 和非文本内容。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _strip_system_tags(content).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                cleaned = _strip_system_tags(part["text"]).strip()
                if cleaned:
                    parts.append(cleaned)
        return "\n".join(parts)
    return ""


def _extract_text_from_responses_item(item: Dict[str, Any]) -> str:
    """从 Responses API 格式的 input item 中提取纯文本。"""
    content = item.get("content", "")
    if isinstance(content, str):
        return _strip_system_tags(content).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                # 支持 input_text / output_text 等带 text 字段的 part
                cleaned = _strip_system_tags(part["text"]).strip()
                if cleaned:
                    parts.append(cleaned)
        return "\n".join(parts)
    return ""


def _compile_error_patterns(patterns: List[str]) -> List[Pattern[str]]:
    """编译正则列表，跳过无效规则并记录警告。"""
    compiled: List[Pattern[str]] = []
    for p in patterns:
        if not isinstance(p, str):
            logger.warning("[Streamify] 工具错误正则不是字符串，已跳过: %r", p)
            continue
        try:
            compiled.append(re.compile(p))
        except re.error as exc:
            logger.warning("[Streamify] 无效的工具错误正则表达式 %r，已跳过: %s", p, exc)
    return compiled


def _is_tool_execution_error(content: str, patterns: List[Pattern[str]]) -> bool:
    """判断工具执行结果是否为错误消息（三个 Mixin 共用）。"""
    if not content:
        return False
    return any(pat.search(content) for pat in patterns)


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
    def _extract_tool_schema(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in tools:
            fn = tool.get("function") or {}
            if fn.get("name") == tool_name:
                return fn.get("parameters")
        return None

    @staticmethod
    def _build_extract_hint(tools: List[Dict[str, Any]], function_name: str, model_reply: str = "") -> str:
        """构建 extract_args 使用的提示词，用于 debug 日志。"""
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in tools:
            fn = (tool.get("function") or {})
            if fn.get("name") == function_name:
                func_desc = fn.get("description", "")
                func_schema = fn.get("parameters", {})
                break
        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值或意图）：\n{model_reply}\n"
            if model_reply else ""
        )
        return _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section="",
        )

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
    def _find_first_function_call(
        response_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """返回响应中第一个 tool_call（无论参数是否为空）。"""
        for choice in response_data.get("choices", []):
            for tc in ((choice.get("message") or {}).get("tool_calls") or []):
                if isinstance(tc, dict):
                    return tc
        return None

    def _find_tool_error_in_request(
        self,
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, str, int]]:
        """在请求消息历史中查找工具执行错误。

        返回 (tool_name, tool_call_id, assistant_msg_index) 或 None。
        """
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict):
                break
            role = msg.get("role")
            # 遇到 assistant 消息即停止，只检查最近一轮的 tool 结果
            if role == "assistant":
                break
            if role != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not _is_tool_execution_error(content, self._error_patterns):  # type: ignore[attr-defined]
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
        model_reply: str = "",
    ) -> Optional[Dict[str, Any]]:
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            fn = (tool.get("function") or {})
            if fn.get("name") == function_name:
                func_desc = fn.get("description", "")
                func_schema = fn.get("parameters", {})
                break

        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )

        source_messages = (
            messages_override if messages_override is not None
            else original_body.get("messages", [])
        )
        source_messages = _trim_messages_by_turns(source_messages, self.fc_context_turns)  # type: ignore[attr-defined]

        # 提取同名工具的历史调用参数和结果（OpenAI 格式）
        tool_result_map: Dict[str, str] = {}
        for m in source_messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                tid = m.get("tool_call_id", "")
                if tid:
                    tool_result_map[tid] = m.get("content", "") or ""
        history_entries: List[str] = []
        for m in source_messages:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if fn.get("name") != function_name:
                    continue
                args_str = fn.get("arguments", "")
                res_str = tool_result_map.get(tc.get("id", ""), "")
                if len(res_str) > 500:
                    res_str = res_str[:500] + "…(截断)"
                entry = f"- 参数: {args_str}"
                if res_str:
                    entry += f"\n  结果: {res_str}"
                history_entries.append(entry)
        tool_history_section = ""
        if history_entries:
            tool_history_section = (
                "以下是该工具之前的调用历史（参数和结果），请严格避免重复使用过去的参数，探索更多可能性(如更换关键词，将中文换为英文等)：\n"
                + "\n".join(history_entries) + "\n"
            )

        tool_system = _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section=tool_history_section,
        )

        # 提取所有消息的纯文本，扁平化为单条 user 消息发送
        context_lines: List[str] = []
        for msg in source_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role in ("system", "tool"):
                continue
            text = _extract_text_from_openai_msg(msg)
            if not text:
                continue
            prefix = "用户" if role == "user" else "助手"
            context_lines.append(f"{prefix}: {text}")
        messages: List[Dict[str, Any]] = [{"role": "system", "content": tool_system}]
        if context_lines:
            messages.append({"role": "user", "content": "\n".join(context_lines)})
        self._last_extract_context = messages  # type: ignore[attr-defined]

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
                return parsed if isinstance(parsed, dict) and parsed else None
        except Exception:
            return None

    @staticmethod
    def _inject_hint(body: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
        body = dict(body)
        hint = _EMPTY_ARGS_HINT
        if tool_name:
            schema = OpenAIFCEnhance._extract_tool_schema(body.get("tools", []), tool_name)
            if schema:
                hint = (
                    f"{hint}\n\n工具 `{tool_name}` 必须使用以下参数调用：\n"
                    f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
                )
            else:
                hint = f"{hint}\n\n特别注意：工具 `{tool_name}` 的参数不能为空。"
        # 1) system 消息注入
        messages = list(body.get("messages", []))
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                existing = msg.get("content", "")
                new_msg = dict(msg)
                new_msg["content"] = f"{existing}\n\n{hint}" if existing else hint
                messages[i] = new_msg
                break
        else:
            messages.insert(0, {"role": "system", "content": hint})
        # 2) messages 末尾追加 user 消息，强化引导
        if tool_name:
            messages.append({
                "role": "user",
                "content": f"请立即调用工具 `{tool_name}`，并填写所有必填参数，不要留空。",
            })
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
    def _extract_tool_schema(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                return tool.get("input_schema")
        return None

    @staticmethod
    def _build_extract_hint(tools: List[Dict[str, Any]], function_name: str, model_reply: str = "") -> str:
        """构建 extract_args 使用的提示词，用于 debug 日志。"""
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name") == function_name:
                func_desc = tool.get("description", "")
                func_schema = tool.get("input_schema", {})
                break
        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )
        return _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section="",
        )

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
    def _find_first_function_call(
        response_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """返回响应中第一个 tool_use block（无论参数是否为空）。"""
        for block in response_data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block
        return None

    def _find_tool_error_in_request(
        self,
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, str, int]]:
        """在请求消息历史中查找工具执行错误（Claude 格式）。

        返回 (tool_name, tool_use_id, assistant_msg_index) 或 None。
        """
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict):
                break
            role = msg.get("role")
            # 遇到 assistant 消息即停止，只检查最近一轮的 tool 结果
            if role == "assistant":
                break
            if role != "user":
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
                if not _is_tool_execution_error(text, self._error_patterns):  # type: ignore[attr-defined]
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
        model_reply: str = "",
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

        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )

        source_messages = (
            messages_override if messages_override is not None
            else original_body.get("messages", [])
        )
        source_messages = _trim_messages_by_turns(source_messages, self.fc_context_turns)  # type: ignore[attr-defined]

        # 提取同名工具的历史调用参数和结果（Claude 格式）
        tool_result_map: Dict[str, str] = {}
        for m in source_messages:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            cnt = m.get("content", [])
            if not isinstance(cnt, list):
                continue
            for blk in cnt:
                if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                    continue
                tid = blk.get("tool_use_id", "")
                if not tid:
                    continue
                rc = blk.get("content", "")
                if isinstance(rc, list):
                    rc = " ".join(
                        c.get("text", "") for c in rc
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                elif not isinstance(rc, str):
                    rc = ""
                tool_result_map[tid] = rc
        history_entries: List[str] = []
        for m in source_messages:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            cnt = m.get("content", [])
            if not isinstance(cnt, list):
                continue
            for blk in cnt:
                if not isinstance(blk, dict) or blk.get("type") != "tool_use":
                    continue
                if blk.get("name") != function_name:
                    continue
                args_str = json.dumps(blk.get("input", {}), ensure_ascii=False)
                res_str = tool_result_map.get(blk.get("id", ""), "")
                if len(res_str) > 500:
                    res_str = res_str[:500] + "…(截断)"
                entry = f"- 参数: {args_str}"
                if res_str:
                    entry += f"\n  结果: {res_str}"
                history_entries.append(entry)
        tool_history_section = ""
        if history_entries:
            tool_history_section = (
                "以下是该工具之前的调用历史（参数和结果），请严格避免重复使用过去的参数，探索更多可能性(如更换关键词，将中文换为英文等)：\n"
                + "\n".join(history_entries) + "\n"
            )

        tool_system = _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(input_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section=tool_history_section,
        )

        cleaned_messages: List[Dict[str, Any]] = []
        for msg in source_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    cleaned = _strip_system_tags(content)
                    if not cleaned.strip():
                        continue
                    msg = {**msg, "content": cleaned}
                elif isinstance(content, list):
                    new_parts = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        # 跳过 tool_result（工具执行结果）
                        if part.get("type") == "tool_result":
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            cleaned = _strip_system_tags(part["text"])
                            if cleaned.strip():
                                new_parts.append({**part, "text": cleaned})
                        else:
                            new_parts.append(part)
                    if not new_parts:
                        continue
                    msg = {**msg, "content": new_parts}
            elif role == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    # 只保留 text 类型的 block，去除 tool_use
                    text_blocks = [b for b in content
                                   if isinstance(b, dict) and b.get("type") == "text"
                                   and b.get("text", "").strip()]
                    if not text_blocks:
                        continue
                    msg = {**msg, "content": text_blocks}
            cleaned_messages.append(msg)
        self._last_extract_context = cleaned_messages  # type: ignore[attr-defined]
        extract_body: Dict[str, Any] = {
            "model": original_body.get("model", ""),
            "max_tokens": original_body.get("max_tokens", 1024),
            "system": tool_system,
            "messages": cleaned_messages,
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
                            return parsed if isinstance(parsed, dict) and parsed else None
                return None
        except Exception:
            return None

    @staticmethod
    def _inject_hint(body: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
        body = dict(body)
        hint = _EMPTY_ARGS_HINT
        if tool_name:
            schema = ClaudeFCEnhance._extract_tool_schema(body.get("tools", []), tool_name)
            if schema:
                hint = (
                    f"{hint}\n\n工具 `{tool_name}` 必须使用以下参数调用：\n"
                    f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
                )
            else:
                hint = f"{hint}\n\n特别注意：工具 `{tool_name}` 的参数不能为空。"
        # 1) system 注入
        existing = body.get("system", "")
        if isinstance(existing, str):
            body["system"] = f"{existing}\n\n{hint}" if existing else hint
        elif isinstance(existing, list):
            new_system = list(existing)
            new_system.append({"type": "text", "text": hint})
            body["system"] = new_system
        else:
            body["system"] = hint
        # 2) messages 末尾追加 user 消息，强化引导
        if tool_name:
            messages = list(body.get("messages", []))
            messages.append({
                "role": "user",
                "content": f"请立即调用工具 `{tool_name}`，并填写所有必填参数，不要留空。",
            })
            body["messages"] = messages
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
    def _extract_tool_schema(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in tools:
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == tool_name:
                    return fd.get("parameters")
        return None

    @staticmethod
    def _build_extract_hint(tools: List[Dict[str, Any]], function_name: str, model_reply: str = "") -> str:
        """构建 extract_args 使用的提示词，用于 debug 日志。"""
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in tools:
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == function_name:
                    func_desc = fd.get("description", "")
                    func_schema = fd.get("parameters", {})
                    break
        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )
        return _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section="",
        )

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
    def _find_first_function_call(
        response_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """返回响应中第一个 functionCall（无论参数是否为空）。"""
        for candidate in response_data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                fc = part.get("functionCall")
                if isinstance(fc, dict):
                    return fc
        return None

    def _find_tool_error_in_request(
        self,
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, int]]:
        """在请求 contents 中查找工具执行错误（Gemini 格式）。

        返回 (tool_name, error_content_index) 或 None。
        """
        contents = body.get("contents", [])
        for i in range(len(contents) - 1, -1, -1):
            item = contents[i]
            if not isinstance(item, dict):
                break
            role = item.get("role")
            # 遇到 model 消息即停止，只检查最近一轮的 tool 结果
            if role == "model":
                break
            if role != "user":
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
                if isinstance(content, str) and _is_tool_execution_error(content, self._error_patterns):  # type: ignore[attr-defined]
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
                if isinstance(fc, dict) and fc.get("name") == function_name:
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
        model_reply: str = "",
    ) -> Optional[Dict[str, Any]]:
        func_desc = ""
        func_params_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            for fd in tool.get("functionDeclarations", []):
                if fd.get("name") == function_name:
                    func_desc = fd.get("description", "")
                    func_params_schema = fd.get("parameters", {})
                    break

        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )

        source_contents = list(
            contents_override if contents_override is not None
            else original_body.get("contents", [])
        )
        # 提取侧请求不做 trim：需要完整对话来收集 FC/FR 历史和原始用户文本

        # 收集同名工具的历史调用参数和结果
        func_resp_map: Dict[str, List[str]] = {}
        for item in source_contents:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            for part in item.get("parts", []):
                if not isinstance(part, dict):
                    continue
                fr = part.get("functionResponse")
                if not isinstance(fr, dict):
                    continue
                resp_content = (fr.get("response") or {}).get("content", "")
                if isinstance(resp_content, str):
                    func_resp_map.setdefault(fr.get("name", ""), []).append(resp_content)

        # 收集同名工具的历史调用：(args_dict, result_str) 对
        history_data: List[Tuple[Dict[str, Any], str]] = []
        resp_idx: Dict[str, int] = {}
        for item in source_contents:
            if not isinstance(item, dict) or item.get("role") != "model":
                continue
            for part in item.get("parts", []):
                if not isinstance(part, dict):
                    continue
                fc = part.get("functionCall")
                if not isinstance(fc, dict) or fc.get("name") != function_name:
                    continue
                args = fc.get("args") or {}
                idx = resp_idx.get(function_name, 0)
                resp_list = func_resp_map.get(function_name, [])
                res_str = resp_list[idx] if idx < len(resp_list) else ""
                resp_idx[function_name] = idx + 1
                if len(res_str) > 500:
                    res_str = res_str[:500] + "…(截断)"
                history_data.append((args, res_str))

        tool_system = _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_params_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section="",
        )

        # 提取对话中的纯文本，去除 functionCall / functionResponse
        context_lines: List[str] = []
        for item in source_contents:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            for part in item.get("parts", []):
                if not isinstance(part, dict):
                    continue
                if "functionCall" in part or "functionResponse" in part:
                    continue
                text = part.get("text", "")
                if not text:
                    continue
                text = _strip_system_tags(text)
                if not text.strip():
                    continue
                prefix = "用户" if role == "user" else "助手"
                context_lines.append(f"{prefix}: {text}")

        # 构建多轮提取对话
        extract_contents: List[Dict[str, Any]] = []
        if context_lines:
            extract_contents.append({
                "role": "user",
                "parts": [{"text": "\n".join(context_lines)}],
            })

        # 历史调用作为多轮：model 输出之前的 JSON → user 反馈失败
        for args, res in history_data:
            extract_contents.append({
                "role": "model",
                "parts": [{"text": json.dumps(args, ensure_ascii=False)}],
            })
            feedback = "这组参数导致调用失败。"
            if res:
                feedback += f" 返回: {res}"
            feedback += " 请换个思路重新推断参数。"
            extract_contents.append({
                "role": "user",
                "parts": [{"text": feedback}],
            })

        if not extract_contents:
            extract_contents.append({
                "role": "user",
                "parts": [{"text": f"请为工具 `{function_name}` 推断参数。"}],
            })

        self._last_extract_context = extract_contents  # type: ignore[attr-defined]

        extract_body: Dict[str, Any] = {
            "contents": extract_contents,
            "systemInstruction": {"parts": [{"text": tool_system}]},
            "generationConfig": {"responseMimeType": "application/json"},
        }
        if "safetySettings" in original_body:
            extract_body["safetySettings"] = original_body["safetySettings"]

        logger.info(
            "Streamify [extract]: 工具 %s, contents %d 条, system(前200): %s",
            function_name, len(extract_contents),
            tool_system[:200],
        )
        # 打印每条 content 的角色和文本前100字
        for ci, c in enumerate(extract_contents):
            _role = c.get("role", "?")
            _txt = ""
            for _p in c.get("parts", []):
                if isinstance(_p, dict) and "text" in _p:
                    _txt = _p["text"][:100]
                    break
            logger.info("Streamify [extract]:   [%d] %s: %s", ci, _role, _txt)

        try:
            url = self._build_url(stream_path)  # type: ignore[attr-defined]
            logger.info("Streamify [extract]: 请求 URL: %s", url)
            async with self._request(  # type: ignore[attr-defined]
                "POST",
                url,
                json=extract_body,
                headers=headers,
                params=params,
            ) as resp:
                if resp.status != 200:
                    err_text = ""
                    try:
                        err_text = await resp.text()
                        if len(err_text) > 500:
                            err_text = err_text[:500] + "…"
                    except Exception:
                        pass
                    logger.warning(
                        "Streamify [extract]: API 返回 %d, 工具 %s, 详情: %s",
                        resp.status, function_name, err_text,
                    )
                    return None
                text_parts: List[str] = []
                async for _event, data in self._iter_sse_events(resp):  # type: ignore[attr-defined]
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                        for cand in chunk.get("candidates", []):
                            for part in (cand.get("content") or {}).get("parts", []):
                                t = part.get("text")
                                if isinstance(t, str) and t.strip():
                                    text_parts.append(t)
                    except Exception:
                        continue
                if not text_parts:
                    logger.warning("Streamify [extract]: 工具 %s 响应无文本", function_name)
                    return None
                text = "".join(text_parts).strip()
                logger.info(
                    "Streamify [extract]: 工具 %s 原始响应(前300): %s",
                    function_name, text[:300],
                )
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(
                        lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                    )
                # 先尝试直接解析，失败则用 raw_decode 从第一个 { 开始解析
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                    decoder = json.JSONDecoder()
                    pos = 0
                    while pos < len(text):
                        idx = text.find("{", pos)
                        if idx == -1:
                            break
                        try:
                            obj, end_pos = decoder.raw_decode(text, idx)
                            if isinstance(obj, dict) and obj:
                                parsed = obj
                                break
                        except json.JSONDecodeError:
                            pos = idx + 1
                            continue
                    if parsed is None:
                        logger.warning("Streamify [extract]: 工具 %s 响应中无 JSON 对象", function_name)
                        return None
                if isinstance(parsed, dict) and parsed:
                    logger.info(
                        "Streamify [extract]: 工具 %s 提取成功: %s",
                        function_name, json.dumps(parsed, ensure_ascii=False),
                    )
                    return parsed
                logger.warning("Streamify [extract]: 工具 %s 解析结果为空", function_name)
                return None
        except Exception as exc:
            logger.warning("Streamify [extract]: 工具 %s 异常: %s", function_name, exc)
            return None

    @staticmethod
    def _inject_hint(body: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
        body = dict(body)
        hint = _EMPTY_ARGS_HINT
        if tool_name:
            schema = GeminiFCEnhance._extract_tool_schema(body.get("tools", []), tool_name)
            if schema:
                hint = (
                    f"{hint}\n\n工具 `{tool_name}` 必须使用以下参数调用：\n"
                    f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
                )
            else:
                hint = f"{hint}\n\n特别注意：工具 `{tool_name}` 的参数不能为空。"
        # 1) systemInstruction 注入
        hint_part = {"text": hint}
        sys_inst = body.get("systemInstruction")
        if isinstance(sys_inst, dict):
            parts = list(sys_inst.get("parts", []))
            parts.append(hint_part)
            body["systemInstruction"] = {**sys_inst, "parts": parts}
        else:
            body["systemInstruction"] = {"parts": [hint_part]}
        # 2) contents 中追加引导，合并到末尾 user 消息避免连续 user 角色
        if tool_name:
            schema_for_hint = GeminiFCEnhance._extract_tool_schema(body.get("tools", []), tool_name)
            required = (schema_for_hint or {}).get("required", [])
            if required:
                params_list = "、".join(f"`{p}`" for p in required)
                user_hint = (
                    f"请根据上述对话内容，立即调用工具 `{tool_name}`。"
                    f"必填参数为 {params_list}，请从对话中推断每个参数的具体值，不要留空。"
                )
            else:
                user_hint = (
                    f"请根据上述对话内容，立即调用工具 `{tool_name}`，"
                    f"并为所有参数提供合理的值，不要使用空对象 {{}} 调用。"
                )
            contents = list(body.get("contents", []))
            if contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "user":
                last = dict(contents[-1])
                last["parts"] = list(last.get("parts", [])) + [{"text": user_hint}]
                contents[-1] = last
            else:
                contents.append({"role": "user", "parts": [{"text": user_hint}]})
            body["contents"] = contents
        return body


class OpenAIResponsesFCEnhance:
    """OpenAI Responses API FC enhancement mixin."""

    @staticmethod
    def _tool_has_required_params(tools: List[Dict[str, Any]], tool_name: str) -> bool:
        for tool in tools:
            if tool.get("name") == tool_name:
                required = (tool.get("parameters") or {}).get("required", [])
                return bool(required)
        return False

    @staticmethod
    def _extract_tool_schema(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool.get("parameters")
        return None

    @staticmethod
    def _build_extract_hint(tools: List[Dict[str, Any]], function_name: str, model_reply: str = "") -> str:
        """构建 extract_args 使用的提示词，用于 debug 日志。"""
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in tools:
            if tool.get("name") == function_name:
                func_desc = tool.get("description", "")
                func_schema = tool.get("parameters", {})
                break
        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )
        return _EXTRACT_PROMPT_TEMPLATE.format(
            function_name=function_name,
            func_desc=func_desc,
            func_schema=json.dumps(func_schema, ensure_ascii=False),
            model_reply_section=model_reply_section,
            tool_history_section="",
        )

    @staticmethod
    def _find_failed_function_call(
        response_data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        tools = tools or []
        for item in response_data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            args = item.get("arguments", "")
            tool_name = item.get("name", "")
            if not args or args == "{}":
                if OpenAIResponsesFCEnhance._tool_has_required_params(tools, tool_name):
                    return item
                continue
            try:
                json.loads(args)
            except json.JSONDecodeError:
                return item
        return None

    @staticmethod
    def _find_first_function_call(
        response_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """返回响应中第一个 function_call 项（无论参数是否为空）。"""
        for item in response_data.get("output", []):
            if isinstance(item, dict) and item.get("type") == "function_call":
                return item
        return None

    def _find_tool_error_in_request(
        self,
        body: Dict[str, Any],
    ) -> Optional[Tuple[str, str, int]]:
        """在 input 中查找工具执行错误（Responses API 格式）。

        返回 (tool_name, call_id, function_call_item_index) 或 None。
        """
        input_items = body.get("input", [])
        if not isinstance(input_items, list):
            return None
        for i in range(len(input_items) - 1, -1, -1):
            item = input_items[i]
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                break
            if item.get("type") != "function_call_output":
                continue
            content = item.get("output", "")
            if not isinstance(content, str) or not _is_tool_execution_error(content, self._error_patterns):  # type: ignore[attr-defined]
                continue
            call_id = item.get("call_id", "")
            for j in range(i - 1, -1, -1):
                prev = input_items[j]
                if not isinstance(prev, dict):
                    continue
                if prev.get("type") == "function_call":
                    if prev.get("call_id") == call_id or not call_id:
                        return (prev.get("name", ""), call_id, j)
                    break
        return None

    @staticmethod
    def _build_corrected_tool_response(
        call_id: str, tool_name: str, args: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        return {
            "id": f"resp_proxy_fix_{int(time.time() * 1000)}",
            "object": "response",
            "status": "completed",
            "model": model_name,
            "output": [{
                "type": "function_call",
                "id": f"fc_proxy_{int(time.time() * 1000)}",
                "call_id": call_id,
                "name": tool_name,
                "arguments": json.dumps(args),
            }],
        }

    @staticmethod
    def _patch_function_call_args(
        response_data: Dict[str, Any], function_name: str, args: Dict[str, Any]
    ) -> None:
        for item in response_data.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call" and item.get("name") == function_name:
                item["arguments"] = json.dumps(args)
                return

    async def _extract_args_as_json(
        self,
        original_body: Dict[str, Any],
        function_name: str,
        sub_path: str,
        headers: Dict[str, str],
        input_override: Optional[List[Dict[str, Any]]] = None,
        model_reply: str = "",
    ) -> Optional[Dict[str, Any]]:
        func_desc = ""
        func_schema: Dict[str, Any] = {}
        for tool in original_body.get("tools", []):
            if tool.get("name") == function_name:
                func_desc = tool.get("description", "")
                func_schema = tool.get("parameters", {})
                break

        model_reply_section = (
            f"模型已生成的回复文本（请优先从中提取具体参数值）：\n{model_reply}\n"
            if model_reply else ""
        )

        source_input = (
            input_override if input_override is not None
            else original_body.get("input", [])
        )
        extract_input: Any = []
        if isinstance(source_input, list):
            source_input = _trim_messages_by_turns(source_input, self.fc_context_turns, user_role="user")  # type: ignore[attr-defined]

            # 提取同名工具的历史调用参数和结果（Responses 格式）
            call_output_map: Dict[str, str] = {}
            for item in source_input:
                if not isinstance(item, dict) or item.get("type") != "function_call_output":
                    continue
                cid = item.get("call_id", "")
                if cid:
                    call_output_map[cid] = item.get("output", "") or ""
            history_entries: List[str] = []
            for item in source_input:
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                if item.get("name") != function_name:
                    continue
                args_str = item.get("arguments", "")
                res_str = call_output_map.get(item.get("call_id", ""), "")
                if len(res_str) > 500:
                    res_str = res_str[:500] + "…(截断)"
                entry = f"- 参数: {args_str}"
                if res_str:
                    entry += f"\n  结果: {res_str}"
                history_entries.append(entry)
            tool_history_section = ""
            if history_entries:
                tool_history_section = (
                    "以下是该工具之前的调用历史（参数和结果），请严格避免重复使用过去的参数，探索更多可能性(如更换关键词，将中文换为英文等)：\n"
                    + "\n".join(history_entries) + "\n"
                )

            tool_system = _EXTRACT_PROMPT_TEMPLATE.format(
                function_name=function_name,
                func_desc=func_desc,
                func_schema=json.dumps(func_schema, ensure_ascii=False),
                model_reply_section=model_reply_section,
                tool_history_section=tool_history_section,
            )

            # 提取所有消息的纯文本，扁平化为单条 user 消息
            context_lines: List[str] = []
            for item in source_input:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type in ("function_call", "function_call_output"):
                    continue
                text = _extract_text_from_responses_item(item)
                if not text:
                    continue
                role = item.get("role", "")
                prefix = "用户" if role == "user" else "助手"
                context_lines.append(f"{prefix}: {text}")
            if context_lines:
                extract_input = "\n".join(context_lines)
        else:
            tool_system = _EXTRACT_PROMPT_TEMPLATE.format(
                function_name=function_name,
                func_desc=func_desc,
                func_schema=json.dumps(func_schema, ensure_ascii=False),
                model_reply_section=model_reply_section,
                tool_history_section="",
            )

        self._last_extract_context = extract_input  # type: ignore[attr-defined]
        extract_body: Dict[str, Any] = {
            "model": original_body.get("model", ""),
            "instructions": tool_system,
            "input": extract_input,
            "text": {"format": {"type": "json_object"}},
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
                text = self._extract_text_from_response(result)
                if not text:
                    return None
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(
                        lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                    )
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) and parsed else None
        except Exception:
            return None

    @staticmethod
    def _extract_text_from_response(response_data: Dict[str, Any]) -> str:
        """从 Responses API 响应中提取文本内容。"""
        for item in response_data.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text", "").strip()
                        if text:
                            return text
        return ""

    @staticmethod
    def _inject_hint(body: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
        body = dict(body)
        hint = _EMPTY_ARGS_HINT
        if tool_name:
            schema = OpenAIResponsesFCEnhance._extract_tool_schema(body.get("tools", []), tool_name)
            if schema:
                hint = (
                    f"{hint}\n\n工具 `{tool_name}` 必须使用以下参数调用：\n"
                    f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
                )
            else:
                hint = f"{hint}\n\n特别注意：工具 `{tool_name}` 的参数不能为空。"
        existing = body.get("instructions", "")
        body["instructions"] = f"{existing}\n\n{hint}" if existing else hint
        if tool_name:
            input_items = list(body.get("input", []))
            input_items.append({
                "role": "user",
                "content": f"请立即调用工具 `{tool_name}`，并填写所有必填参数，不要留空。",
            })
            body["input"] = input_items
        return body
