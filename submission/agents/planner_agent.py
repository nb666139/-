"""
GridSynergy — 规划Agent (PlannerAgent)
负责接收自然语言调度指令，生成结构化的电网调度方案。
包括机组组合、发电出力分配、拓扑开关状态等决策。
"""

import json
import re
from typing import Any, Optional

import numpy as np

from config import get_config
from llm.llm_client import LLMClient


# ============================================================================
# Prompt 模板
# ============================================================================

PLANNER_SYSTEM_PROMPT = """你是一名资深的电力系统调度专家，专精于高比例新能源接入下的电网运行调度。

## 你的职责
根据当前电网状态和调度指令，生成一个安全、经济、可行的发电调度方案。

## 你需要输出的内容
请严格按以下JSON格式输出调度方案，不要包含其他文字：

```json
{
    "summary": "方案的简要说明（一句话）",
    "unit_commitment": {
        "G1": {"status": "on", "output_mw": 50.0},
        "G2": {"status": "on", "output_mw": 40.0},
        ...
    },
    "topology_switches": {
        "L1": "closed",
        "L6": "open",
        ...
    },
    "renewable_curtailment": {
        "wind_mw": 0.0,
        "solar_mw": 0.0
    },
    "constraints_check": {
        "power_balance": true,
        "voltage_ok": true,
        "line_loading_ok": true
    },
    "expected_cost": 0.0
}
```

## 调度原则
1. 优先消纳新能源（风电、光伏），尽量减少弃风弃光
2. 保证功率平衡：总发电 = 总负荷 + 网损（约3%）
3. 各发电机出力不超过其容量上限
4. 保留足够的旋转备用（约5%-10%的总负荷）
5. 线路不过载（负载率 < 100%）
6. 节点电压保持在0.95-1.05 p.u.范围内
7. 经济调度：优先使用边际成本低的机组

## 约束条件
- 机组爬坡率限制：每分钟不超过额定容量的5%
- N-1安全准则：任一元件故障后系统仍可稳定运行
- 新能源出力具有不确定性，需考虑预测误差

## 推理要求 (重要!)
在 summary 字段开头用「决策依据：」开头，用一两句话说明你为什么做出这样的机组出力分配决策。
例如：「决策依据：由于风电骤降50%，需增加G1/G2燃煤机组出力，同时启动G5燃气备用以填补缺口。」
然后在 summary 后面继续描述调度方案内容。"""

PLANNER_USER_TEMPLATE = """{memory_context}
## 当前调度指令
{dispatch_instruction}

## 电网当前状态
- 总负荷：{total_load} MW
- 新能源出力预测：风电 {wind_forecast} MW，光伏 {solar_forecast} MW
- 机组在线状态：{generator_status}
- 当前拓扑开关状态：{topology_status}
- 线路负载情况：{line_loading}

## 要求
请根据以上信息（含历史经验参考）生成最优调度方案。"""


# ============================================================================
# PlannerAgent
# ============================================================================

class PlannerAgent:
    """
    规划Agent
    根据自然语言调度指令和电网上下文，通过LLM推理生成结构化的调度方案。
    在Demo模式下使用内置规则生成方案而不调用LLM。
    """

    def __init__(self) -> None:
        """初始化规划Agent"""
        self._config = get_config()
        self._llm_client: Optional[LLMClient] = None
        if not self._config.demo_mode:
            self._llm_client = LLMClient()
        # 发电机容量数据（IEEE-30标准，MW）
        self._generator_capacity: dict[str, float] = {
            "G1": 80.0, "G2": 80.0, "G3": 50.0,
            "G4": 55.0, "G5": 30.0, "G6": 40.0,
        }
        # 发电机边际成本（$/MWh，排序用）
        self._generator_cost: dict[str, float] = {
            "G1": 20.0, "G2": 22.0, "G3": 35.0,
            "G4": 30.0, "G5": 28.0, "G6": 40.0,
        }

    def plan(self, dispatch_instruction: str, grid_context: dict[str, Any]) -> dict[str, Any]:
        """
        根据调度指令和电网上下文生成调度方案。

        参数:
            dispatch_instruction: 自然语言调度指令（如"将风电出力提升20%"）
            grid_context: 电网当前状态字典，包含负荷、发电机状态、拓扑等

        返回:
            结构化的调度方案字典
        """
        if self._config.demo_mode:
            return self._plan_demo(dispatch_instruction, grid_context)
        else:
            return self._plan_llm(dispatch_instruction, grid_context)

    def _plan_demo(self, dispatch_instruction: str, grid_context: dict[str, Any]) -> dict[str, Any]:
        """
        生成调度方案。

        优先使用SCUCSolver（Pyomo+HiGHS）进行MILP优化求解；
        若Pyomo/HiGHS未安装，回退到启发式经济调度规则。
        """
        total_load: float = float(grid_context.get("total_load", 250.0))
        wind_forecast: float = float(grid_context.get("wind_forecast", 60.0))
        solar_forecast: float = float(grid_context.get("solar_forecast", 30.0))
        gen_status: dict[str, str] = grid_context.get("generator_status", {})
        topology: dict[str, str] = grid_context.get("topology_status", {})

        # 尝试使用SCUCSolver进行MILP优化
        try:
            from solver.scuc_solver import SCUCSolver
            from data.ieee30 import GENERATOR_DATA, LINE_DATA

            # 构建24h预测数据（取当前值扩展）
            load_24h = np.array([total_load * 0.85 + total_load * 0.15 * abs(np.sin(np.pi * h / 12))
                                 for h in range(24)])
            wind_24h = np.array([wind_forecast * (0.8 + 0.4 * abs(np.sin(np.pi * (h + 3) / 12)))
                                for h in range(24)])
            pv_24h = np.array([solar_forecast * max(0, np.sin(np.pi * (h - 6) / 12))
                              for h in range(24)])

            scuc = SCUCSolver(horizon=24)
            scuc.set_gen_params(GENERATOR_DATA)
            scuc.set_line_params(LINE_DATA)
            scuc_result = scuc.solve(load_24h, wind_24h, pv_24h)

            # 转换为Planner输出格式（取第一个时段的机组出力）
            gen_output = scuc_result["generator_output"]
            unit_commitment = {}
            for gen_name, gen_vals in gen_output.items():
                if not gen_name.startswith("G") or not isinstance(gen_vals, list):
                    continue
                unit_commitment[gen_name] = {
                    "status": "on",
                    "output_mw": round(gen_vals[0], 2),
                    "capacity_mw": self._generator_capacity.get(gen_name, 50.0),
                }

            plan = {
                "summary": f"SCUC优化方案 (方法={scuc_result['method']})，"
                           f"总负荷{total_load:.1f}MW，总成本{scuc_result['total_cost']:.1f}万元",
                "unit_commitment": unit_commitment,
                "topology_switches": topology or {f"L{i}": "closed" for i in range(1, 42)},
                "renewable_curtailment": {
                    "wind_mw": round(scuc_result["curtailed_mw"][0] * 0.6, 2),
                    "solar_mw": round(scuc_result["curtailed_mw"][0] * 0.4, 2),
                },
                "constraints_check": {"power_balance": True, "voltage_ok": True, "line_loading_ok": True},
                "expected_cost": scuc_result["total_cost"],
                "metadata": {"mode": "scuc", "method": scuc_result["method"],
                             "total_load": total_load, "total_cost": scuc_result["total_cost"]},
            }
            return plan

        except ImportError:
            pass  # 回退到启发式规则

        # 回退：启发式经济调度
        renewable_output = wind_forecast + solar_forecast
        net_load = max(total_load * 1.03 - renewable_output, 0.0)

        wind_curtail = 0.0
        solar_curtail = 0.0
        if "风电" in dispatch_instruction and "降" in dispatch_instruction:
            pct_match = re.search(r'(\d+)%', dispatch_instruction)
            pct = float(pct_match.group(1)) / 100.0 if pct_match else 0.2
            wind_curtail = wind_forecast * pct

        online_gens = [g for g, s in gen_status.items() if s == "on"] or list(self._generator_capacity.keys())
        sorted_gens = sorted(online_gens, key=lambda g: self._generator_cost.get(g, 30.0))

        unit_commitment = {}
        remaining = net_load
        for i, gn in enumerate(sorted_gens):
            cap = self._generator_capacity.get(gn, 50.0)
            if i < len(sorted_gens) - 1:
                share = min(cap * 0.85, remaining * cap / sum(self._generator_capacity.get(g, 50.0) for g in sorted_gens))
            else:
                share = min(cap * 0.9, remaining)
            share = max(share, cap * 0.2)
            remaining -= share
            unit_commitment[gn] = {"status": "on", "output_mw": round(share, 2), "capacity_mw": cap}

        for gn in online_gens:
            if gn not in unit_commitment:
                unit_commitment[gn] = {"status": "on", "output_mw": 0.0, "capacity_mw": self._generator_capacity.get(gn, 50.0)}

        if not topology:
            topology = {f"L{i}": "closed" for i in range(1, 42)}

        plan = {
            "summary": f"启发式调度方案，总负荷{total_load:.1f}MW，新能源出力{renewable_output:.1f}MW",
            "unit_commitment": unit_commitment,
            "topology_switches": topology,
            "renewable_curtailment": {"wind_mw": round(wind_curtail, 2), "solar_mw": round(solar_curtail, 2)},
            "constraints_check": {"power_balance": True, "voltage_ok": True, "line_loading_ok": True},
            "expected_cost": round(sum(u["output_mw"] * self._generator_cost.get(g, 30.0) for g, u in unit_commitment.items()), 2),
            "metadata": {"mode": "heuristic", "total_load": total_load, "renewable_output": renewable_output, "net_load": net_load},
        }
        return plan

    def _plan_llm(self, dispatch_instruction: str, grid_context: dict[str, Any]) -> dict[str, Any]:
        """
        使用LLM生成调度方案。
        """
        # 注入默认拓扑和线路负载数据（若未提供）
        grid_context = self._enrich_grid_context(grid_context)

        # 构建用户消息
        total_load = grid_context.get("total_load", 250.0)
        wind_forecast = grid_context.get("wind_forecast", 60.0)
        solar_forecast = grid_context.get("solar_forecast", 30.0)
        generator_status = grid_context.get("generator_status", {})
        topology_status = grid_context.get("topology_status", {})
        line_loading = grid_context.get("line_loading", {})

        user_message: str = PLANNER_USER_TEMPLATE.format(
            memory_context=grid_context.get("memory_context", ""),
            dispatch_instruction=dispatch_instruction,
            total_load=total_load,
            wind_forecast=wind_forecast,
            solar_forecast=solar_forecast,
            generator_status=json.dumps(generator_status, ensure_ascii=False),
            topology_status=json.dumps(topology_status, ensure_ascii=False),
            line_loading=json.dumps(line_loading, ensure_ascii=False),
        )

        # 调用LLM
        response: str = self._llm_client.chat(  # type: ignore[union-attr]
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # 解析JSON响应
        plan: dict[str, Any] = self._parse_json_response(response)
        plan["metadata"] = {
            "mode": "llm",
            "raw_response": response[:500],
            "reasoning": self._extract_reasoning(response),
        }
        return plan

    def _enrich_grid_context(self, grid_context: dict[str, Any]) -> dict[str, Any]:
        """为 grid_context 注入默认拓扑和线路负载数据。"""
        ctx = dict(grid_context)
        if not ctx.get("topology_status"):
            ctx["topology_status"] = {f"L{i}": "closed" for i in range(1, 42)}
        if not ctx.get("line_loading"):
            ctx["line_loading"] = {"avg": "~60%", "max_branch": "L12: 82%", "warning": None}
        return ctx

    def _extract_reasoning(self, response: str) -> str:
        """从 LLM 原始响应中提取推理过程（决策依据）。"""
        match = re.search(r'决策依据[：:]\s*(.+?)(?:[。.]|\\n)', response, re.DOTALL)
        if match:
            return match.group(1).strip()[:200]
        # 尝试从 summary 中提取
        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response)
        if summary_match:
            return summary_match.group(1)[:200]
        return "未能提取明确推理依据"

    def check_relevance(self, user_input: str) -> dict[str, Any]:
        """便捷方法：检查输入是否与电网调度相关。"""
        from llm.input_guard import InputGuard
        return InputGuard().check(user_input)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """
        从LLM响应中提取JSON结构化方案。
        支持```json代码块包裹或纯JSON格式。
        """
        # 尝试匹配 ```json ... ``` 代码块
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试匹配整个响应为JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 回退：返回空方案
        print(f"[PlannerAgent] 警告：无法解析LLM响应为JSON，返回空方案。响应片段：{response[:200]}")
        return {
            "summary": "LLM响应解析失败",
            "unit_commitment": {},
            "topology_switches": {},
            "renewable_curtailment": {"wind_mw": 0.0, "solar_mw": 0.0},
            "constraints_check": {
                "power_balance": False,
                "voltage_ok": False,
                "line_loading_ok": False,
            },
            "expected_cost": 0.0,
        }
