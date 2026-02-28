from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

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

    def _resolve_providers(self):
        providers = self.config.get("providers", []) or []

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

    async def initialize(self):
        if not self.config.get("enabled", True):
            logger.info("Streamify proxy disabled")
            return

        port = self.config.get("port", 6190)
        providers = self._resolve_providers()

        self.proxy = StreamifyProxy(port=port, providers=providers)
        await self.proxy.start()

    @filter.command("streamify")
    async def status(self, event: AstrMessageEvent):
        """查看稳流代理状态"""
        if not self.proxy or not self.proxy.runner:
            yield event.plain_result("稳流代理未运行")
            return

        port = self.config.get("port", 6190)
        routes = self.proxy.get_route_infos()

        if not routes:
            yield event.plain_result(
                f"稳流代理运行中\n监听: http://127.0.0.1:{port}\n路由: (未配置)"
            )
            return

        lines = ["稳流代理运行中", f"监听: http://127.0.0.1:{port}", "路由列表:"]
        for info in routes:
            proxy_url = info["proxy_url"] if info["proxy_url"] else "(none)"
            lines.append(
                f"- /{info['route_name']} -> {info['target_url']} (proxy: {proxy_url})"
            )

        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        if self.proxy:
            await self.proxy.stop()
