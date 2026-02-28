# astrbot_plugin_streamify_router

## 注意:此项目为纯AIGC(gpt-5.3-codex)创建的插件，可能未经合适的性能检查或存在一些未知的问题，如果你有遇到类似的问题，可以使用此插件处理，或自行fork，但也请注意风险

LLM 非流请求稳流网关（AstrBot 插件）。

将非流请求转为流式请求，拼接响应后返回(假非流)，防止部分包装了cloudflare的中转因非流请求在2min内无响应，而超时自动关闭(429)，从而触发神秘的 'NoneType' object has no attribute 'get' 问题；原本就是流式请求时直接透传 SSE。

## 主要能力

- 多提供商路由：通过 `providers` 列表配置多个转发目标。
- 路径分发：按 `/{route_name}/...` 区分不同目标。
- 自动识别 API 格式（无需 `provider_type`）：
  - OpenAI Chat Completions: `/v1/chat/completions`
  - Claude Messages: `/v1/messages`
  - Gemini Generate Content: `:generateContent` / `:streamGenerateContent`
  - OpenAI Responses: `/v1/responses`
- 每个路由可独立配置 `proxy_url`。
- 可选调试日志：开启后输出每次请求的处理状态与耗时。

## 更健壮的处理

- 流式解析改为按行处理 SSE，避免分块边界导致 UTF-8 多字节字符被截断后丢字。
- 增加全局 `request_timeout` 超时控制，覆盖连接和读取阶段，防止上游长时间挂起导致协程堆积。
- 对 `providers` 配置做统一类型归一化，配置异常时自动回退，减少运行期分支和空值问题。
- OpenAI Chat 非流聚合增强工具调用兼容：同时支持 `delta.tool_calls`、`delta.function_call`、`choice.message.tool_calls`、`choice.message.function_call`，并兼容缺失 `index` 的分片聚合，降低工具调用中断概率。
- Gemini 非流聚合保留完整 `content.parts`，支持文本与 `functionCall` 等非文本 part 合并，不再只拼接文本。

## 使用方式

1. 安装并启用插件。
2. 在插件配置中填写 `providers` , 设置转发端点
3. 在 AstrBot provider 中 `base_url` 对应设为 `http://localhost:<端口，默认23456>/<路由名>`。

## 配置项

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `true` | 是否启用代理 |
| `debug` | bool | `false` | 是否输出调试日志（成功请求也会记录） |
| `port` | int | `23456` | 本地监听端口 |
| `request_timeout` | float | `120.0` | 上游请求全局超时（秒），用于连接/读取超时控制 |
| `providers` | template_list | - | 多路由配置列表（`route_name`/`target_url`/`forward_url`/`proxy_url`） |

## 指令

- `/streamify`：查看代理运行状态和已注册路由。
