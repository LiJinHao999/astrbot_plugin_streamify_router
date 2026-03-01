# astrbot_plugin_streamify_router

Astrbot 无感稳流网关（AstrBot 插件）。

## 常见问题

如果你有遇到以下问题:

1. gemini function calling(尤其是gemini3.0模型) / 其它模型 fuction calling 工具调用失败或参数空回

```bash
Tool `astrbot_execute_shell` Result: error: Tool handler parameter mismatch,          
  please check the handler definition. Handler parameters: command: str, background: bool = False, env: dict = {} 
```

2. 在使用插件的过程中，超过两分钟时自动超时、截断，上游返回429并超时……

```bash
'NoneType' object has no attribute 'get'  或 LLM 响应错误: All chat models failed: Exception: 请求失败, 返回的 candidates 为空

上游429超时，无法自动重试
```

插件将在本地启动一个转发修复端口来解决这类出于模型问题/上游问题所产生的兼容性问题

只需要你**填写需要转发的url，并更换astrbot provider的url为转发的url**

通常是http://127.0.0.1:23456/yourendpoint (保存插件设置后，你应该填写的url示例会自动显示在设置项中)

**如果还有什么难以处理的神秘问题，欢迎提issue讨论，作者看见会回复和讨论的ovo**

## 处理思路介绍

### 无感修复 Function Calling 报错 & 提示词注入修复

**Layer1 : 预处理&记忆拦截**

由于插件属于网关层，网关层收到消息将会在astrbot之前，可以提前拦截，但网关无法自行判断工具调用是否是错误的

因此，Layer1根据：tool中是否有必填的参数项没填(required)与插件缓存记忆中的tool是否会因为没填参数项报错来提前拦截&重试，扼杀出现错误的可能性。

**Layer2 : 根据astrbot响应中是否有报错响应来进行重试，关键词正则匹配(如不能.*为空、error:、(?i)missing \d+ required).可以在配置项中自行设置。**

提示词注入修复：发送一条含有用户消息、llm试图使用的工具、这个工具的介绍和参数介绍的消息(仅包含报错工具，最小化上下文处理)，然后解析这条简单消息的响应

**Layer3 : 如果重试、提示词调用都失败，注入一条工具调用失败的消息，遵从事实，防止模型产生莫名其妙的幻觉**

### 伪装非流

网关接收到非流请求，但在内部将非流请求转为流式请求，拼接响应后返回(假非流)，即可防止部分包装了cloudflare的中转因非流请求在2min内无响应，而超时自动关闭(429)

而原本就是流式请求时直接透传 SSE。

### 特性支持

- 多提供商路由：通过 `providers` 列表配置多个转发目标。
- 路径分发：按 `/{route_name}/...` 区分不同目标。
- 自动识别 API 格式（无需 `provider_type`）：
  - OpenAI Chat Completions: `/v1/chat/completions`
  - Claude Messages: `/v1/messages`
  - Gemini Generate Content: `:generateContent` / `:streamGenerateContent`
  - OpenAI Responses: `/v1/responses`
- 每个路由可独立配置 `proxy_url`，是否开启假非流转发
- 可选调试日志：开启后输出每次请求的处理状态与耗时。

## 使用方式

1. 安装并启用插件。
2. 在插件配置中填写 `providers` , 设置转发端点
3. 在 AstrBot provider 中 `base_url` 对应设为 `http://localhost:<端口>，默认23456>/<路由名>`。

## 指令

- `/streamify`：查看代理运行状态和已注册路由。

## 配置项

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `port` | 本地监听端口 | `23456` |
| `request_timeout` | 上游请求超时（秒） | `120.0` |
| `pseudo_non_stream` | 是否启用全局假非流覆盖 | `true` |
| `tool_error_patterns` | 额外的工具错误识别正则列表 | `[]` |
| `fix_retries` | 最多重试次数 | `1` |
| `extract_args` | 启用 FC 增强（工具参数 JSON 提取） | `false` |
| `debug` | 启用调试日志 | `false` |
| `providers` | 提供商路由列表（`template_list`） | - |
| `providers[].route_name` | 路由名称 | - |
| `providers[].target_url` | 上游 API 基础 URL | `https://api.openai.com` |
| `providers[].forward_url` | 生成的本地转发 URL（自动生成） | `""` |
| `providers[].proxy_url` | HTTP 代理 URL（可选） | `""` |
| `providers[].pseudo_non_stream` | 路由级别假非流覆盖 | `true` |
