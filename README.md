# astrbot_plugin_streamify_router

LLM 非流请求稳流网关（AstrBot 插件）。

将非流式请求自动转为流式拉取并聚合回完整响应，降低长耗时请求导致的连接超时风险；原本就是流式请求时直接透传 SSE。

## 主要能力

- 多提供商路由：通过 `providers` 列表配置多个转发目标。
- 路径分发：按 `/{route_name}/...` 区分不同目标。
- 自动识别 API 格式（无需 `provider_type`）：
  - OpenAI Chat Completions: `/v1/chat/completions`
  - Claude Messages: `/v1/messages`
  - Gemini Generate Content: `:generateContent` / `:streamGenerateContent`
  - OpenAI Responses: `/v1/responses`
- 每个路由可独立配置 `proxy_url`。

## 使用方式

1. 安装并启用插件。
2. 在插件配置中填写 `providers`。
3. 在 AstrBot provider 中将 `base_url` 设为 `http://localhost:<端口>/<路由名>`。

## 配置项

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `true` | 是否启用代理 |
| `port` | int | `23456` | 本地监听端口 |
| `providers` | template_list | - | 多路由配置列表（`route_name`/`target_url`/`proxy_url`） |

## 指令

- `/streamify`：查看代理运行状态和已注册路由。
