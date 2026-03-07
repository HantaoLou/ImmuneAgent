# -*- coding: utf-8 -*-
"""
GLM OpenAI 兼容代理服务器

这个代理服务器将 OpenAI 格式的 API 请求转发到智谱 GLM API。
用于在 OpenCode 沙盒内提供 GLM 支持，无需修改 OpenCode 源码。

使用方式:
1. 在沙盒内启动: python glm_proxy.py --port 8080 --api-key YOUR_GLM_KEY
2. OpenCode 配置: OPENAI_BASE_URL=http://localhost:8080/v1
"""

import json
import asyncio
import argparse
import os
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 智谱 GLM API 配置
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

# 模型映射: OpenAI 模型名 -> GLM 模型名
MODEL_MAPPING = {
    "gpt-4o": "glm-4-flash",
    "gpt-4o-mini": "glm-4-flash",
    "gpt-4": "glm-4-plus",
    "gpt-4-turbo": "glm-4-plus",
    "gpt-3.5-turbo": "glm-4-flash",
    # 直接使用 GLM 模型名也支持
    "glm-4": "glm-4",
    "glm-4-flash": "glm-4-flash",
    "glm-4-plus": "glm-4-plus",
    "glm-4-air": "glm-4-air",
}


@dataclass
class ProxyConfig:
    """代理配置"""
    port: int = 8080
    api_key: Optional[str] = None
    debug: bool = False


class GLMProxy:
    """智谱 GLM OpenAI 兼容代理"""
    
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供智谱 API Key (通过 --api-key 或 ZHIPU_API_KEY 环境变量)")
        
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 分钟超时
        
    def _get_glm_model(self, openai_model: str) -> str:
        """将 OpenAI 模型名映射到 GLM 模型名"""
        return MODEL_MAPPING.get(openai_model, "glm-4-flash")
    
    def _convert_messages(self, openai_messages: list) -> list:
        """转换消息格式 (OpenAI -> GLM)"""
        glm_messages = []
        
        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # 处理不同消息类型
            if role == "system":
                glm_messages.append({"role": "system", "content": content})
            elif role == "user":
                # 处理多模态内容
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)
                glm_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                glm_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # 工具调用结果
                glm_messages.append({"role": "tool", "content": content})
        
        return glm_messages
    
    def _convert_tools(self, openai_tools: list) -> list:
        """转换工具格式 (OpenAI -> GLM)"""
        glm_tools = []
        
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                glm_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    }
                })
        
        return glm_tools
    
    async def _call_glm_api(
        self, 
        model: str, 
        messages: list, 
        tools: list = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        """调用智谱 GLM API"""
        
        glm_model = self._get_glm_model(model)
        glm_messages = self._convert_messages(messages)
        
        payload = {
            "model": glm_model,
            "messages": glm_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = self._convert_tools(tools)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        url = f"{ZHIPU_API_BASE}/chat/completions"
        
        if self.config.debug:
            logger.info(f"调用 GLM API: model={glm_model}, messages_count={len(glm_messages)}")
        
        response = await self.client.post(
            url,
            headers=headers,
            json=payload,
            timeout=300.0,
        )
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"GLM API 错误: {response.status_code} - {error_text}")
            raise Exception(f"GLM API 错误: {response.status_code}")
        
        return response.json()
    
    async def _stream_glm_api(
        self,
        model: str,
        messages: list,
        tools: list = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """流式调用智谱 GLM API"""
        
        glm_model = self._get_glm_model(model)
        glm_messages = self._convert_messages(messages)
        
        payload = {
            "model": glm_model,
            "messages": glm_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        if tools:
            payload["tools"] = self._convert_tools(tools)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        url = f"{ZHIPU_API_BASE}/chat/completions"
        
        if self.config.debug:
            logger.info(f"流式调用 GLM API: model={glm_model}")
        
        async with self.client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=300.0,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"GLM API 错误: {response.status_code} - {error_text}")
                raise Exception(f"GLM API 错误: {response.status_code}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        # 解析 GLM 响应并转换为 OpenAI 格式
                        glm_data = json.loads(data)
                        openai_data = self._convert_stream_response(glm_data)
                        yield f"data: {json.dumps(openai_data)}\n\n"
                    except json.JSONDecodeError:
                        continue
    
    def _convert_response(self, glm_response: dict, model: str) -> dict:
        """转换响应格式 (GLM -> OpenAI)"""
        choices = []
        for choice in glm_response.get("choices", []):
            message = choice.get("message", {})
            choices.append({
                "index": choice.get("index", 0),
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            })
        
        usage = glm_response.get("usage", {})
        
        return {
            "id": glm_response.get("id", f"chatcmpl-{hash(str(glm_response))}"),
            "object": "chat.completion",
            "created": glm_response.get("created", 0),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
    
    def _convert_stream_response(self, glm_data: dict) -> dict:
        """转换流式响应格式"""
        choices = []
        for choice in glm_data.get("choices", []):
            delta = choice.get("delta", {})
            choices.append({
                "index": choice.get("index", 0),
                "delta": {
                    "role": delta.get("role"),
                    "content": delta.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason"),
            })
        
        return {
            "id": glm_data.get("id", ""),
            "object": "chat.completion.chunk",
            "created": glm_data.get("created", 0),
            "model": glm_data.get("model", ""),
            "choices": choices,
        }


async def run_proxy_server(config: ProxyConfig):
    """运行代理服务器"""
    from aiohttp import web
    
    proxy = GLMProxy(config)
    
    async def handle_models(request):
        """返回可用模型列表"""
        models = [
            {"id": model, "object": "model", "owned_by": "zhipu"}
            for model in MODEL_MAPPING.keys()
        ]
        return web.json_response({
            "object": "list",
            "data": models,
        })
    
    async def handle_chat_completions(request):
        """处理聊天补全请求"""
        try:
            body = await request.json()
            
            model = body.get("model", "gpt-4o")
            messages = body.get("messages", [])
            tools = body.get("tools")
            stream = body.get("stream", False)
            max_tokens = body.get("max_tokens", 4096)
            temperature = body.get("temperature", 0.7)
            
            if stream:
                # 流式响应
                response = web.StreamResponse()
                response.headers["Content-Type"] = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = "keep-alive"
                await response.prepare(request)
                
                async for chunk in proxy._stream_glm_api(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    await response.write(chunk.encode("utf-8"))
                    await response.drain()
                
                return response
            else:
                # 非流式响应
                glm_response = await proxy._call_glm_api(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                openai_response = proxy._convert_response(glm_response, model)
                return web.json_response(openai_response)
                
        except Exception as e:
            logger.error(f"处理请求错误: {e}")
            return web.json_response(
                {"error": {"message": str(e), "type": "proxy_error"}},
                status=500,
            )
    
    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    
    logger.info(f"启动 GLM 代理服务器: http://0.0.0.0:{config.port}")
    logger.info(f"OpenAI 兼容端点: http://localhost:{config.port}/v1")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", config.port)
    await site.start()
    
    # 保持运行
    while True:
        await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description="GLM OpenAI 兼容代理服务器")
    parser.add_argument("--port", type=int, default=8080, help="代理服务器端口")
    parser.add_argument("--api-key", type=str, help="智谱 API Key")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    config = ProxyConfig(
        port=args.port,
        api_key=args.api_key,
        debug=args.debug,
    )
    
    asyncio.run(run_proxy_server(config))


if __name__ == "__main__":
    main()

