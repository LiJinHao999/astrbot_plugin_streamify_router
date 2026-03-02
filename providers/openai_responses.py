import json
from typing import Any, Dict, Optional

from aiohttp import web

from .base import ProviderHandler


class OpenAIResponsesHandler(ProviderHandler):
    ENDPOINT = "v1/responses"

    def matches(self, sub_path: str) -> bool:
        return sub_path.strip("/") == self.ENDPOINT

    async def handle(self, req: web.Request, sub_path: str) -> web.Response:
        if req.method.upper() != "POST" or not self.matches(sub_path):
            return await self._passthrough(req, sub_path)

        body = await self._read_json(req)
        if body is None:
            return await self._passthrough(req, sub_path)

        headers = self._forward_headers(req)
        if body.get("stream"):
            return await self._proxy_stream(req, sub_path, body, headers)

        # 假非流禁用时直通
        if not self.pseudo_non_stream:
            return await self._passthrough(req, sub_path)

        body["stream"] = True
        async with self._request(
            "POST",
            self._build_url(sub_path),
            json=body,
            headers=headers,
            params=req.query,
        ) as resp:
            if resp.status != 200:
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp),
                    text=await resp.text(),
                )

            completed: Optional[Dict[str, Any]] = None
            fallback: Optional[Dict[str, Any]] = None

            async for event_name, data in self._iter_sse_events(resp):
                if not data:
                    continue
                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event_name or payload.get("type")
                if event_type == "response.completed":
                    response_obj = payload.get("response")
                    completed = response_obj if isinstance(response_obj, dict) else payload
                elif event_type == "response.created":
                    response_obj = payload.get("response")
                    if isinstance(response_obj, dict):
                        fallback = response_obj
                    elif isinstance(payload, dict):
                        fallback = payload

            if completed is None:
                completed = fallback
            if completed is None:
                return web.json_response(
                    {
                        "error": {
                            "message": "No completed response event received from upstream.",
                            "type": "proxy_error",
                        }
                    },
                    status=502,
                )

            return web.json_response(completed)
