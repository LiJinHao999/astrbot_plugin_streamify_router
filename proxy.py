import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientSession, web

from astrbot.api import logger

from .providers import (
    ProviderHandler,
    _sanitize_for_log,
    get_handler_classes,
)


class ProviderRoute:
    def __init__(
        self,
        route_name: str,
        target_url: str,
        proxy_url: str = "",
        session: Optional[ClientSession] = None,
        debug: bool = False,
        request_timeout: float = 120.0,
        pseudo_non_stream: bool = True,
        extract_args: bool = False,
        fix_retries: int = 1,
        tool_error_patterns: Optional[List[str]] = None,
    ):
        self.route_name = route_name
        self.target_url = target_url.rstrip("/")
        self.proxy_url = proxy_url.strip()
        self.debug = bool(debug)
        self.base = ProviderHandler(
            self.target_url,
            self.proxy_url,
            session=session,
            debug=self.debug,
            request_timeout=request_timeout,
        )
        common_kwargs = dict(
            session=session,
            debug=self.debug,
            request_timeout=request_timeout,
            pseudo_non_stream=pseudo_non_stream,
            extract_args=extract_args,
            fix_retries=fix_retries,
            tool_error_patterns=tool_error_patterns,
        )
        self.handlers: List[ProviderHandler] = [
            cls(self.target_url, self.proxy_url, **common_kwargs)
            for cls in get_handler_classes()
        ]

    async def dispatch(self, req: web.Request, sub_path: str) -> web.Response:
        path = sub_path.strip("/")
        for handler in self.handlers:
            if handler.matches(path):
                return await handler.handle(req, path)
        return await self.base.handle(req, path)


class StreamifyProxy:
    def __init__(
        self,
        port: int = 23456,
        providers: Optional[List[Dict[str, Any]]] = None,
        debug: bool = False,
        request_timeout: float = 120.0,
        pseudo_non_stream: bool = True,
        extract_args: bool = False,
        fix_retries: int = 1,
        tool_error_patterns: Optional[List[str]] = None,
    ):
        self.port = port
        self.providers_config = providers or []
        self.debug = bool(debug)
        self.request_timeout = ProviderHandler._normalize_timeout(request_timeout)
        self.pseudo_non_stream = bool(pseudo_non_stream)
        self.extract_args = bool(extract_args)
        self.fix_retries = max(0, int(fix_retries))
        self.tool_error_patterns: Optional[List[str]] = tool_error_patterns
        self.providers: Dict[str, ProviderRoute] = {}
        self.session: Optional[ClientSession] = None
        self.app = web.Application(client_max_size=20 * 1024 * 1024)
        self.app.add_routes(
            [
                web.route("*", "/{route_name}", self._dispatch),
                web.route("*", "/{route_name}/{sub_path:.*}", self._dispatch),
            ]
        )
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

    def _load_providers(self) -> None:
        if self.session is None or self.session.closed:
            raise RuntimeError("Proxy session not initialized. Call start() first.")

        self.providers = {}
        for item in self.providers_config:
            if not isinstance(item, dict):
                continue

            route_name = str(item.get("route_name", "")).strip().strip("/")
            target_url = str(item.get("target_url", "")).strip()
            proxy_url = str(item.get("proxy_url", "")).strip()

            if not route_name or not target_url:
                logger.warning(
                    "Skip invalid provider config: route_name=%r target_url=%r",
                    route_name,
                    target_url,
                )
                continue

            if route_name in self.providers:
                logger.warning("Duplicate route_name %s, overwrite previous target.", route_name)

            # 路由级 pseudo_non_stream 优先；未配置则继承全局值
            route_pseudo = item.get("pseudo_non_stream")
            if isinstance(route_pseudo, bool):
                pseudo_non_stream = route_pseudo
            else:
                pseudo_non_stream = self.pseudo_non_stream

            self.providers[route_name] = ProviderRoute(
                route_name,
                target_url,
                proxy_url,
                session=self.session,
                debug=self.debug,
                request_timeout=self.request_timeout,
                pseudo_non_stream=pseudo_non_stream,
                extract_args=self.extract_args,
                fix_retries=self.fix_retries,
                tool_error_patterns=self.tool_error_patterns,
            )

    async def _dispatch(self, req: web.Request) -> web.Response:
        route_name = req.match_info.get("route_name", "").strip("/")
        sub_path = req.match_info.get("sub_path", "")
        start_time = time.perf_counter()

        provider = self.providers.get(route_name)
        if provider is None:
            if self.debug:
                path_text = f"/{route_name}"
                if sub_path:
                    path_text = f"{path_text}/{sub_path.strip('/')}"
                logger.info("Streamify debug route miss: %s %s", req.method.upper(), path_text)
            return web.json_response(
                {
                    "error": {
                        "message": f"Unknown route_name '{route_name}'.",
                        "type": "route_not_found",
                    },
                    "available_routes": sorted(self.providers.keys()),
                },
                status=404,
            )

        path_text = f"/{route_name}"
        if sub_path:
            path_text = f"{path_text}/{sub_path.strip('/')}"

        if self.debug:
            try:
                if req.method.upper() in {"POST", "PUT", "PATCH"}:
                    raw = await req.read()
                    if raw:
                        try:
                            body_obj = json.loads(raw)
                            sanitized = _sanitize_for_log(body_obj)
                            detail = json.dumps(sanitized, ensure_ascii=False)
                        except Exception:
                            detail = f"<{len(raw)} bytes, non-JSON>"
                    else:
                        detail = "<empty body>"
                else:
                    qs = dict(req.query)
                    detail = json.dumps(qs, ensure_ascii=False) if qs else "<no query>"
                logger.info(
                    "Streamify >>> %s %s -> %s\n%s",
                    req.method.upper(), path_text, provider.target_url, detail,
                )
            except Exception as _exc:
                logger.warning("Streamify debug: 记录请求日志失败: %s", _exc)

        try:
            response = await provider.dispatch(req, sub_path)
            if self.debug:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "Streamify debug handled: %s %s -> %s status=%s elapsed=%dms",
                    req.method.upper(),
                    path_text,
                    provider.target_url,
                    response.status,
                    elapsed_ms,
                )
                try:
                    if isinstance(response, web.Response) and response.body:
                        content_type = response.content_type or ""
                        if "json" in content_type or "text" in content_type:
                            body_text = response.body.decode("utf-8", errors="replace")
                            try:
                                resp_obj = json.loads(body_text)
                                sanitized_resp = _sanitize_for_log(resp_obj)
                                resp_detail = json.dumps(sanitized_resp, ensure_ascii=False)
                            except Exception:
                                resp_detail = body_text[:2000]
                        else:
                            resp_detail = f"<{len(response.body)} bytes, {content_type}>"
                    else:
                        resp_detail = "<streaming response>"
                    logger.info(
                        "Streamify <<< %s %s status=%s elapsed=%dms\n%s",
                        req.method.upper(), path_text, response.status, elapsed_ms, resp_detail,
                    )
                except Exception as _exc:
                    logger.warning("Streamify debug: 记录响应日志失败: %s", _exc)
            return response
        except Exception as exc:
            logger.exception("Proxy dispatch failed for route %s: %s", route_name, exc)
            return web.json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "proxy_internal_error",
                    }
                },
                status=500,
            )

    def get_route_infos(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for route_name in sorted(self.providers.keys()):
            provider = self.providers[route_name]
            infos.append(
                {
                    "route_name": route_name,
                    "target_url": provider.target_url,
                    "proxy_url": provider.proxy_url,
                }
            )
        return infos

    async def start(self) -> None:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        self._load_providers()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await self.site.start()

        logger.info("Streamify proxy started at http://127.0.0.1:%s", self.port)
        if self.debug:
            logger.info("Streamify debug logging enabled.")
        if not self.providers:
            logger.warning("No provider routes configured.")
        else:
            for info in self.get_route_infos():
                proxy_text = info["proxy_url"] or "(none)"
                logger.info(
                    "Route /%s -> %s (proxy: %s)",
                    info["route_name"],
                    info["target_url"],
                    proxy_text,
                )

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            self.site = None

        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

        logger.info("Streamify proxy stopped.")
