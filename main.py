from typing import Any, List

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .proxy import StreamifyProxy


@register(
    "astrbot_plugin_streamify_router",
    "LiJinHao999",
    "Astrbot 非侵入式 LLM 稳流网关(使用aiohttp)，提升 LLM 稳定性与使用体验，提供工具调用强化与假非流转发支持，无感强化工具调用体验，静默重试&修复function calling请求体(尤其是gemini 工具调用空回问题)，无感解决非流式请求超时截断问题(429)，如果你在使用LLM(目前尤其是gemini)的过程中频繁遇到这类报错，可以尝试一下这个插件",
    "1.0.0",
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
        # 向后兼容读取配置项
        pseudo_non_stream = bool(
            self.config.get("pseudo_non_stream", self.config.get("enabled", True))
        )
        fix_retries = int(
            self.config.get("fix_retries", self.config.get("gemini_fix_retries", 1))
        )
        extract_args = bool(
            self.config.get("extract_args", self.config.get("gemini_extract_args", False))
        )

        if not pseudo_non_stream and not extract_args:
            logger.info("Streamify: 假非流与 FC 增强均已禁用，代理不启动")
            return

        port = self.config.get("port", 23456)
        debug = self._is_debug_enabled()
        request_timeout = self._resolve_request_timeout()
        self._sync_provider_forward_urls(port)
        providers = self._resolve_providers()

        raw_patterns = self.config.get("tool_error_patterns", None)
        tool_error_patterns = raw_patterns if isinstance(raw_patterns, list) else None

        self.proxy = StreamifyProxy(
            port=port,
            providers=providers,
            debug=debug,
            request_timeout=request_timeout,
            pseudo_non_stream=pseudo_non_stream,
            extract_args=extract_args,
            fix_retries=fix_retries,
            tool_error_patterns=tool_error_patterns,
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
