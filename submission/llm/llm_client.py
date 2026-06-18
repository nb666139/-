"""
GridSynergy — LLM调用客户端 (LLMClient)
支持OpenAI兼容API的调用封装，含Demo模式下的模拟响应。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from config import get_config


class LLMClient:
    """
    LLM API调用客户端

    支持所有OpenAI兼容的API接口（OpenAI、vLLM、DeepSeek、Qwen等）。
    在Demo模式下，返回模拟的调度方案而不实际调用API。
    """

    def __init__(self) -> None:
        """初始化LLM客户端"""
        self._config = get_config()

        self._api_key: str = self._config.llm.api_key
        self._base_url: str = self._config.llm.base_url
        self._model: str = self._config.llm.model
        self._temperature: float = self._config.llm.temperature
        self._max_tokens: int = self._config.llm.max_tokens
        self._timeout: int = self._config.llm.timeout
        self._max_retries: int = self._config.llm.max_retries

        self._client: Any = None  # OpenAI client instance

        # 尝试初始化OpenAI客户端
        if self._api_key:
            self._init_client()

    def _init_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        except ImportError:
            print("[LLMClient] 警告：openai库未安装，将使用Demo模式")
            self._api_key = ""
        except Exception as e:
            print(f"[LLMClient] 警告：初始化OpenAI客户端失败：{e}，将使用Demo模式")
            self._api_key = ""

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """
        调用LLM Chat Completion。

        参数:
            system_prompt: 系统提示词（角色设定）
            user_message: 用户消息
            temperature: 生成温度（None则使用配置值）

        返回:
            LLM的文本响应
        """
        if not self._api_key or self._client is None:
            return self._chat_demo(system_prompt, user_message)

        if temperature is None:
            temperature = self._temperature

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self._max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLMClient] API调用失败 (尝试 {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print("[LLMClient] 所有重试失败，回退到Demo模式")
                    return self._chat_demo(system_prompt, user_message)

        return ""

    def chat_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        调用LLM的Function Calling / Tool Use 功能。

        参数:
            system_prompt: 系统提示词
            user_message: 用户消息
            tools: OpenAI格式的工具定义列表

        返回:
            Tool调用结果字典，或文本响应
        """
        if not self._api_key or self._client is None:
            # Demo模式下返回模拟的工具调用
            return {
                "role": "assistant",
                "content": "Demo模式：工具调用模拟响应",
                "tool_calls": [],
            }

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    tools=tools,
                    tool_choice="auto",
                )

                choice = response.choices[0]
                result: dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                }

                if choice.message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                return result

            except Exception as e:
                print(f"[LLMClient] API调用失败 (尝试 {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "role": "assistant",
                        "content": f"API调用失败: {e}",
                        "tool_calls": [],
                    }

        return {"role": "assistant", "content": "", "tool_calls": []}

    def _chat_demo(self, system_prompt: str, user_message: str) -> str:
        """
        Demo模式下的模拟响应。
        生成一个合理的调度方案JSON，不调用LLM API。
        """
        # 从用户消息中提取关键信息
        total_load: float = 250.0
        wind_forecast: float = 60.0
        solar_forecast: float = 30.0

        # 尝试从消息中解析负荷和新源预测
        import re
        load_match = re.search(r'总负荷[：:]\s*([\d.]+)', user_message)
        if load_match:
            total_load = float(load_match.group(1))

        wind_match = re.search(r'风电[^:：]*?([\d.]+)\s*MW', user_message)
        if wind_match:
            wind_forecast = float(wind_match.group(1))

        solar_match = re.search(r'光伏[^:：]*?([\d.]+)\s*MW', user_message)
        if solar_match:
            solar_forecast = float(solar_match.group(1))

        # 计算净负荷
        renewable_output: float = wind_forecast + solar_forecast
        net_load: float = total_load * 1.03 - renewable_output

        # 经济调度：按边际成本排序
        generator_cost: dict[str, float] = {
            "G1": 20.0, "G2": 22.0, "G3": 35.0,
            "G4": 30.0, "G5": 28.0, "G6": 40.0,
        }
        generator_capacity: dict[str, float] = {
            "G1": 80.0, "G2": 80.0, "G3": 50.0,
            "G4": 55.0, "G5": 30.0, "G6": 40.0,
        }

        # 按成本排序
        sorted_gens: list[str] = sorted(generator_cost, key=generator_cost.get)  # type: ignore[arg-type]
        unit_commitment: dict[str, dict[str, Any]] = {}
        remaining: float = max(0.0, net_load)
        total_cap: float = sum(generator_capacity[g] for g in sorted_gens)

        for gen_name in sorted_gens:
            cap = generator_capacity[gen_name]
            share = min(cap * 0.85, remaining * cap / max(total_cap, 1))
            share = max(share, cap * 0.2)
            remaining -= share
            unit_commitment[gen_name] = {
                "status": "on",
                "output_mw": round(share, 2),
            }

        # 构建响应
        plan: dict[str, Any] = {
            "summary": f"LLM模拟方案：总负荷{total_load:.1f}MW，新能源{renewable_output:.1f}MW，{len(unit_commitment)}台机组在线",
            "unit_commitment": unit_commitment,
            "topology_switches": {f"L{i}": "closed" for i in range(1, 42)},
            "renewable_curtailment": {"wind_mw": 0.0, "solar_mw": 0.0},
            "constraints_check": {
                "power_balance": True,
                "voltage_ok": True,
                "line_loading_ok": True,
            },
            "expected_cost": round(
                sum(u["output_mw"] * generator_cost.get(g, 30.0) for g, u in unit_commitment.items()), 2
            ),
        }

        # 返回JSON格式（带代码块包裹以模拟LLM输出风格）
        return f"```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```"
