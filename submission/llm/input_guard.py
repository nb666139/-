"""
GridSynergy — 输入校验模块 (Input Guard)

对用户自然语言输入进行 LLM 语义校验，判断是否与电网调度相关。
无关输入将被拒绝，避免无效调用浪费 API 配额和系统资源。
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import get_config
from llm.llm_client import LLMClient

# ============================================================================
# Prompt 模板
# ============================================================================

RELEVANCE_CHECK_SYSTEM_PROMPT = """你是一个电力系统调度领域的输入过滤专家。

你的任务是判断用户的输入是否与"新能源电网调度"相关。

## 相关主题（以下任一匹配即为相关）
- 电网调度、发电计划、机组组合、经济调度
- 新能源消纳：风电、光伏、可再生能源出力调整
- 电力系统安全：N-1故障、拓扑重构、负荷分配
- 电力市场：VPP博弈、多主体协同、电力交易
- 电网运行状态：负荷变化、出力波动、备用容量
- 电力系统优化：成本最小化、运行效率

## 不相关主题（拒绝）
- 写诗、写文章、翻译、闲聊
- 其他领域的技术问题（如软件开发、化学、医学等）
- 没有任何调度/电力相关语义的纯数学计算
- 代码生成、编程相关

## 输出格式
请严格按以下JSON格式输出，不要包含其他文字：
```json
{"relevant": true, "reason": "简短理由(最多20字)"}
```
"""


class InputGuard:
    """输入校验守卫，调用 LLM 判断用户输入与电网调度是否相关。"""

    def __init__(self) -> None:
        self._config = get_config()
        self._llm_client: LLMClient | None = None
        if not self._config.demo_mode:
            self._llm_client = LLMClient()

    def check(self, user_input: str) -> dict[str, Any]:
        """
        校验用户输入。

        参数:
            user_input: 用户原始自然语言输入

        返回:
            {"relevant": bool, "reason": str, "method": "llm"|"keyword"|"demo"}
        """
        # 快速关键词预筛 — 明显无关的直接拒绝，节省 API 调用
        quick_check = self._keyword_check(user_input)
        if quick_check is not None:
            return quick_check

        # Demo 模式下的保守策略 — 放过
        if self._config.demo_mode:
            return {"relevant": True, "reason": "Demo模式默认放行", "method": "demo"}

        # LLM 语义校验
        return self._llm_check(user_input)

    def _keyword_check(self, user_input: str) -> dict[str, Any] | None:
        """
        关键词快速预筛。

        如果明确不相关 → 直接拒绝 (返回 dict)
        如果明确相关 → 直接放行 (返回 dict)
        如果无法确定 → 返回 None (交给 LLM)
        """
        text = user_input.lower()

        # 白名单关键词（明确与电网调度相关）
        whitelist = [
            "电网", "调度", "发电", "机组", "风电", "光伏", "太阳能", "新能源",
            "消纳", "负荷", "功率", "n-1", "故障", "拓扑", "vpp", "博弈",
            "潮流", "电压", "线路", "安全", "备用", "出力", "储能",
            "日前", "实时", "计划", "供电", "电力", "成本", "经济调度",
        ]
        for kw in whitelist:
            if kw in text:
                return {"relevant": True, "reason": f"关键词匹配: {kw}", "method": "keyword"}

        # 黑名单关键词（明显不可能与调度有关）
        blacklist = [
            "写诗", "写文章", "翻译", "笑话", "聊天", "讲故事", "你是谁",
            "你好", "天气", "股票", "电影", "音乐", "游戏", "做饭",
            "python", "代码", "编程", "debug", "bug",
        ]
        for kw in blacklist:
            if kw in text:
                return {
                    "relevant": False,
                    "reason": f"输入与电网调度无关 (检测到无关关键词)",
                    "method": "keyword",
                }

        # 纯问候/极短/无意义的输入
        if len(text.strip()) < 4 and not any(k in text for k in ["电", "风", "光", "能"]):
            return {
                "relevant": False,
                "reason": "输入过短且未包含电网相关关键词",
                "method": "keyword",
            }

        # 无法确定 → 交给 LLM
        return None

    def _llm_check(self, user_input: str) -> dict[str, Any]:
        """使用 LLM 进行语义级校验。"""
        assert self._llm_client is not None, "LLM client 未初始化"

        try:
            response = self._llm_client.chat(
                system_prompt=RELEVANCE_CHECK_SYSTEM_PROMPT,
                user_message=f'请判断以下输入是否与电网调度相关：\n"{user_input}"\n\n按JSON格式输出。',
            )
            parsed = self._parse_json(response)
            if parsed and "relevant" in parsed:
                return {
                    "relevant": bool(parsed["relevant"]),
                    "reason": str(parsed.get("reason", "LLM校验")),
                    "method": "llm",
                }
        except Exception as e:
            print(f"  [InputGuard] LLM 校验异常: {e}")

        # 异常回退：保守放行
        return {"relevant": True, "reason": "校验异常, 保守放行", "method": "fallback"}

    def _parse_json(self, response: str) -> dict[str, Any] | None:
        """从 LLM 响应中提取 JSON。"""
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return None
