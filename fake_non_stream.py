import json
import time
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import ClientResponse

# 注入到请求的提示，引导模型正确填写工具参数
_EMPTY_ARGS_HINT = (
    "重要：调用工具/函数时，必须为所有必填参数提供有效的 JSON 值。"
    "如果工具有必填参数，绝对不要用空对象 {} 调用。调用前请填写每一个必填字段。"
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class OpenAIFakeNonStream:
    """OpenAI chat completions SSE→non-stream reassembly mixin."""

    @staticmethod
    def _new_choice_slot() -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content_parts": [],
            "finish_reason": None,
            "tool_calls": {},
            "tool_calls_seen": False,
            "legacy_function_call_seen": False,
        }

    @staticmethod
    def _new_tool_slot() -> Dict[str, Any]:
        return {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": []},
        }

    @staticmethod
    def _normalize_arguments_piece(arguments: Any) -> Optional[str]:
        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            return None
        try:
            return json.dumps(arguments, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(arguments)

    def _apply_function_payload(
        self,
        tc_slot: Dict[str, Any],
        function_payload: Any,
        append_arguments: bool,
    ) -> None:
        if not isinstance(function_payload, dict):
            return
        if isinstance(function_payload.get("name"), str):
            tc_slot["function"]["name"] = function_payload["name"]
        arguments_piece = self._normalize_arguments_piece(function_payload.get("arguments"))
        if arguments_piece is None:
            return
        if append_arguments:
            tc_slot["function"]["arguments"].append(arguments_piece)
        else:
            tc_slot["function"]["arguments"] = [arguments_piece]

    def _update_tool_calls(
        self,
        slot: Dict[str, Any],
        payload: Dict[str, Any],
        append_arguments: bool,
    ) -> None:
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            slot["tool_calls_seen"] = True
        for fallback_idx, tool_call in enumerate(tool_calls or []):
            if not isinstance(tool_call, dict):
                continue

            tc_idx = _safe_int(tool_call.get("index", fallback_idx), fallback_idx)
            tc_slot = slot["tool_calls"].setdefault(tc_idx, self._new_tool_slot())
            if isinstance(tool_call.get("id"), str):
                tc_slot["id"] = tool_call["id"]
            if isinstance(tool_call.get("type"), str):
                tc_slot["type"] = tool_call["type"]
            self._apply_function_payload(
                tc_slot,
                tool_call.get("function") or {},
                append_arguments=append_arguments,
            )

        function_call = payload.get("function_call")
        if isinstance(function_call, dict):
            slot["legacy_function_call_seen"] = True
            tc_slot = slot["tool_calls"].setdefault(0, self._new_tool_slot())
            self._apply_function_payload(
                tc_slot,
                function_call,
                append_arguments=append_arguments,
            )

    def _update_choice_slot(self, slot: Dict[str, Any], choice: Dict[str, Any]) -> None:
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}

        if isinstance(delta.get("role"), str):
            slot["role"] = delta["role"]
        elif isinstance(message.get("role"), str):
            slot["role"] = message["role"]

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

        content_message = message.get("content")
        if isinstance(content_message, str):
            slot["content_parts"] = [content_message]
        elif isinstance(content_message, list):
            full_parts: List[str] = []
            for part in content_message:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    full_parts.append(text)
            if full_parts:
                slot["content_parts"] = full_parts

        self._update_tool_calls(slot, delta, append_arguments=True)
        self._update_tool_calls(slot, message, append_arguments=False)

        if choice.get("finish_reason") is not None:
            slot["finish_reason"] = choice.get("finish_reason")

    @staticmethod
    def _merge_chunk_meta(
        result: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not result["id"] and chunk.get("id"):
            result["id"] = chunk["id"]
        if not result["model"] and chunk.get("model"):
            result["model"] = chunk["model"]
        if chunk.get("created"):
            result["created"] = chunk["created"]
        if chunk.get("system_fingerprint"):
            result["system_fingerprint"] = chunk["system_fingerprint"]
        if isinstance(chunk.get("usage"), dict):
            return chunk["usage"]
        return None

    @staticmethod
    def _build_choice_output(idx: int, slot: Dict[str, Any]) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": slot["role"],
            "content": "".join(slot["content_parts"]),
        }
        inferred_finish_reason = "stop"
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

            if slot.get("tool_calls_seen"):
                message["tool_calls"] = assembled_tools
                inferred_finish_reason = "tool_calls"
            elif slot.get("legacy_function_call_seen") and len(assembled_tools) == 1:
                message["function_call"] = assembled_tools[0]["function"]
                inferred_finish_reason = "function_call"
            else:
                message["tool_calls"] = assembled_tools
                inferred_finish_reason = "tool_calls"

            if message["content"] == "":
                message["content"] = None

        return {
            "index": idx,
            "message": message,
            "finish_reason": slot["finish_reason"] or inferred_finish_reason,
        }

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": "",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [],
        }
        choices: Dict[int, Dict[str, Any]] = {}
        usage: Optional[Dict[str, Any]] = None

        async for _event, data in self._iter_sse_events(resp):  # type: ignore[attr-defined]
            if not data:
                continue
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            merged_usage = self._merge_chunk_meta(result, chunk)
            if merged_usage is not None:
                usage = merged_usage

            for choice in chunk.get("choices", []):
                idx = _safe_int(choice.get("index", 0))
                slot = choices.setdefault(idx, self._new_choice_slot())
                self._update_choice_slot(slot, choice)

        for idx in sorted(choices.keys()):
            result["choices"].append(self._build_choice_output(idx, choices[idx]))

        if usage is not None:
            result["usage"] = usage
        if not result["id"]:
            result["id"] = f"chatcmpl-proxy-{int(time.time() * 1000)}"
        return result


class ClaudeFakeNonStream:
    """Claude Messages API SSE→non-stream reassembly mixin."""

    @staticmethod
    def _init_state() -> Dict[str, Any]:
        return {
            "message_id": "",
            "model": "",
            "role": "assistant",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content_blocks": {},
        }

    @staticmethod
    def _handle_message_start(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        message = payload.get("message") or {}
        if isinstance(message.get("id"), str):
            state["message_id"] = message["id"]
        if isinstance(message.get("model"), str):
            state["model"] = message["model"]
        if isinstance(message.get("role"), str):
            state["role"] = message["role"]

        start_usage = message.get("usage") or {}
        if isinstance(start_usage.get("input_tokens"), int):
            state["usage"]["input_tokens"] = start_usage["input_tokens"]

        initial_content = message.get("content")
        if isinstance(initial_content, list):
            for idx, block in enumerate(initial_content):
                if isinstance(block, dict):
                    state["content_blocks"][idx] = dict(block)

    def _handle_content_block_start(
        self, state: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        idx = _safe_int(payload.get("index", len(state["content_blocks"])))
        block = payload.get("content_block") or {}
        if isinstance(block, dict):
            state["content_blocks"][idx] = dict(block)

    def _handle_content_block_delta(
        self, state: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        idx = _safe_int(payload.get("index", 0))
        block = state["content_blocks"].setdefault(idx, {"type": "text", "text": ""})
        delta = payload.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == "text_delta":
            block["type"] = "text"
            block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
        elif delta_type == "input_json_delta":
            block["_input_json"] = (
                f"{block.get('_input_json', '')}{delta.get('partial_json', '')}"
            )

    @staticmethod
    def _handle_message_delta(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        delta = payload.get("delta") or {}
        if "stop_reason" in delta:
            state["stop_reason"] = delta.get("stop_reason")
        if "stop_sequence" in delta:
            state["stop_sequence"] = delta.get("stop_sequence")

        delta_usage = payload.get("usage") or {}
        if isinstance(delta_usage.get("output_tokens"), int):
            state["usage"]["output_tokens"] = delta_usage["output_tokens"]

    @staticmethod
    def _build_content(content_blocks: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        return content

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        state = self._init_state()
        async for event_name, data in self._iter_sse_events(resp):  # type: ignore[attr-defined]
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event_name or payload.get("type")
            if event_type == "message_start":
                self._handle_message_start(state, payload)
            elif event_type == "content_block_start":
                self._handle_content_block_start(state, payload)
            elif event_type == "content_block_delta":
                self._handle_content_block_delta(state, payload)
            elif event_type == "message_delta":
                self._handle_message_delta(state, payload)

        return {
            "id": state["message_id"] or f"msg_proxy_{int(time.time() * 1000)}",
            "type": "message",
            "role": state["role"],
            "model": state["model"],
            "content": self._build_content(state["content_blocks"]),
            "stop_reason": state["stop_reason"],
            "stop_sequence": state["stop_sequence"],
            "usage": state["usage"],
        }


class GeminiFakeNonStream:
    """Gemini generateContent SSE→non-stream reassembly mixin."""

    @staticmethod
    def _init_state() -> Dict[str, Any]:
        return {
            "content_parts": {},
            "candidate_meta": {},
            "role": "model",
            "usage_metadata": None,
            "prompt_feedback": None,
            "model_version": None,
        }

    @staticmethod
    def _first_candidate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        candidate = candidates[0]
        return candidate if isinstance(candidate, dict) else None

    @staticmethod
    def _merge_value(existing: Any, incoming: Any) -> Any:
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged = dict(existing)
            for key, value in incoming.items():
                if key in merged:
                    merged[key] = GeminiFakeNonStream._merge_value(merged[key], value)
                else:
                    merged[key] = value
            return merged
        return incoming

    @staticmethod
    def _append_content_parts(state: Dict[str, Any], content: Dict[str, Any]) -> None:
        if isinstance(content.get("role"), str):
            state["role"] = content["role"]

        parts = content.get("parts")
        if not isinstance(parts, list):
            return
        for idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            slot = state["content_parts"].setdefault(idx, {})
            text = part.get("text")
            if isinstance(text, str):
                if isinstance(slot.get("text"), str):
                    slot["text"] = f"{slot['text']}{text}"
                else:
                    slot["text"] = text
            for key, value in part.items():
                if key == "text":
                    continue
                slot[key] = GeminiFakeNonStream._merge_value(slot.get(key), value)

    @staticmethod
    def _merge_payload(state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        candidate = GeminiFakeNonStream._first_candidate(payload)
        if candidate is not None:
            content = candidate.get("content")
            if isinstance(content, dict):
                GeminiFakeNonStream._append_content_parts(state, content)

            for key, value in candidate.items():
                if key != "content":
                    state["candidate_meta"][key] = value

        if isinstance(payload.get("usageMetadata"), dict):
            state["usage_metadata"] = payload["usageMetadata"]
        if isinstance(payload.get("promptFeedback"), dict):
            state["prompt_feedback"] = payload["promptFeedback"]
        if isinstance(payload.get("modelVersion"), str):
            state["model_version"] = payload["modelVersion"]

    async def _build_non_stream_response(self, resp: ClientResponse) -> Dict[str, Any]:
        state = self._init_state()
        async for _event, data in self._iter_sse_events(resp):  # type: ignore[attr-defined]
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
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

