from typing import Any, List

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .proxy import StreamifyProxy


@register(
    "astrbot_plugin_streamify_router",
    "LiJinHao999",
    "LLM非流请求稳流网关，将非流请求转为流式，防止部分包装了cloudflare的中转因无响应超时自动关闭，解决神秘的 'NoneType' object has no attribute 'get' 问题",
    "0.1.0",
    "https://github.com/LiJinHao999/astrbot_plugin_streamify_router",
)
class StreamifyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.proxy = None

    def _get_provider_list(self) -> List[Any]:
        providers = self.config.get("providers", [])
        return providers if isinstance(providers, list) else []

    def _resolve_providers(self):
        providers = self._get_provider_list()

        # Backward compatibility for old single-target config.
        if not providers:
            target = str(self.config.get("target_base_url", "")).strip()
            if target:
                providers = [
                    {
                        "route_name": "default",
                        "target_url": target,
                        "proxy_url": "",
                    }
                ]
        return providers

    def _is_debug_enabled(self) -> bool:
        value = self.config.get("debug", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _resolve_request_timeout(self) -> float:
        default_timeout = 120.0
        value = self.config.get("request_timeout", default_timeout)
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid request_timeout=%r, fallback to %.1fs.", value, default_timeout
            )
            return default_timeout

        if timeout <= 0:
            logger.warning(
                "Non-positive request_timeout=%r, fallback to %.1fs.",
                value,
                default_timeout,
            )
            return default_timeout
        return timeout

    @staticmethod
    def _build_local_route_base(port: int, route_name: str) -> str:
        name = route_name.strip().strip("/")
        if not name:
            name = "<route_name>"
        return f"http://127.0.0.1:{port}/{name}"

    def _sync_provider_forward_urls(self, port: int) -> None:
        providers = self._get_provider_list()

        changed = False
        for item in providers:
            if not isinstance(item, dict):
                continue

            route_name = str(item.get("route_name", "")).strip()
            expected = self._build_local_route_base(port, route_name)
            if item.get("forward_url") != expected:
                item["forward_url"] = expected
                changed = True

        if changed:
            self.config["providers"] = providers
            try:
                self.config.save_config()
            except Exception as exc:
                logger.warning("Failed to save generated provider forward_url: %s", exc)

    async def initialize(self):
        if not self.config.get("enabled", True):
            logger.info("Streamify proxy disabled")
            return

        port = self.config.get("port", 23456)
        debug = self._is_debug_enabled()
        request_timeout = self._resolve_request_timeout()
        self._sync_provider_forward_urls(port)
        providers = self._resolve_providers()

        self.proxy = StreamifyProxy(
            port=port,
            providers=providers,
            debug=debug,
            request_timeout=request_timeout,
        )
        await self.proxy.start()

    @filter.command("streamify")
    async def status(self, event: AstrMessageEvent):
        """查看稳流代理状态"""
        if not self.proxy or not self.proxy.runner:
            yield event.plain_result("稳流代理未运行")
            return

        port = self.config.get("port", 23456)
        routes = self.proxy.get_route_infos()

        if not routes:
            yield event.plain_result(
                f"稳流代理运行中\n监听: http://127.0.0.1:{port}\n路由: (未配置)"
            )
            return

        debug = self._is_debug_enabled()
        lines = [
            "稳流代理运行中",
            f"监听: http://127.0.0.1:{port}",
            f"调试日志: {'on' if debug else 'off'}",
            "路由列表:",
        ]
        for info in routes:
            proxy_url = info["proxy_url"] if info["proxy_url"] else "(none)"
            local_base = self._build_local_route_base(port, info["route_name"])
            lines.append(
                f"- /{info['route_name']} -> {info['target_url']} (proxy: {proxy_url})"
            )
            lines.append(f"  base_url: {local_base}")

        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        if self.proxy:
            await self.proxy.stop()
