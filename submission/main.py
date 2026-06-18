"""
GridSynergy — 基于多智能体协同决策的新能源电网自主调度系统

主入口：完整的4-Agent协同调度流水线。
流程：自然语言指令 → Planner → Validator → Negotiator → Memory → 输出
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# 确保项目路径在sys.path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, SystemConfig
from agents.planner_agent import PlannerAgent
from agents.validator_agent import ValidatorAgent
from agents.negotiator_agent import NegotiatorAgent
from agents.memory_agent import MemoryAgent, MemoryEntry
from env.scenario_generator import ScenarioGenerator
from solver.power_flow import PowerFlowSolver
from solver.safety_checker import SafetyChecker
from llm.llm_client import LLMClient
import numpy as np


# ============================================================================
# GridSynergy 主系统
# ============================================================================

class GridSynergySystem:
    """
    GridSynergy 多智能体协同调度系统

    整合四个核心Agent和控制流，实现完整的自主调度流水线。
    """

    def __init__(self) -> None:
        """初始化GridSynergy系统"""
        self._config = get_config()

        # 初始化各Agent
        self.planner: PlannerAgent = PlannerAgent()
        self.validator: ValidatorAgent = ValidatorAgent()
        self.negotiator: NegotiatorAgent = NegotiatorAgent()
        self.memory: MemoryAgent = MemoryAgent()

        # 初始化求解器
        self.power_flow: PowerFlowSolver = PowerFlowSolver()
        self.safety_checker: SafetyChecker = SafetyChecker()

        # 场景生成器
        self.scenario_gen: ScenarioGenerator = ScenarioGenerator()

        # LLM客户端（非Agent专属，直接调用）
        self.llm_client: LLMClient | None = None
        if not self._config.demo_mode:
            self.llm_client = LLMClient()

        # 运行日志
        self._run_log: list[dict[str, Any]] = []

    def run(self, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        执行一次完整的调度流水线。

        流水线步骤：
        1. Planner: 生成候选调度方案
        2. Validator: 验证方案安全性
        3. Negotiator: 多VPP经济博弈
        4. Memory: 存储经验、提供参考

        参数:
            scenario: 预定义的场景；为None则自动生成

        返回:
            完整的调度结果字典
        """
        print("=" * 70)
        print("  GridSynergy — 新能源电网自主调度系统")
        print("  " + ("Demo模式（无需API Key）" if self._config.demo_mode else "LLM模式"))
        print("=" * 70)

        # ---- 步骤0: 场景准备 ----
        if scenario is None:
            scenario = self.scenario_gen.generate_normal_scenario()

        dispatch_instruction: str = scenario["dispatch_instruction"]
        grid_context: dict[str, Any] = scenario["grid_context"]

        print(f"\n[场景] {scenario.get('type', 'normal')}")
        print(f"[指令] {dispatch_instruction[:120]}...")

        # ---- 步骤0.5: Memory检索相似场景 ----
        print(f"\n{'─' * 50}")
        print("[Memory] 检索历史相似场景...")
        scene_features: np.ndarray = self._extract_scene_features(grid_context)
        memory_context: str = self.memory.get_context(scene_features)
        print(f"  -> 检索到 {len(self.memory.retrieve(scene_features))} 条相似经验")

        # 如果有相似经验，注入到grid_context
        if memory_context and "暂无" not in memory_context:
            grid_context["historical_reference"] = memory_context

        # ---- 步骤1: Planner生成方案 ----
        print(f"\n{'─' * 50}")
        print("[Planner] 生成调度方案...")

        debate_round: int = 0
        max_debate_rounds: int = self._config.agent.max_debate_rounds
        dispatch_plan: dict[str, Any] = {}
        validation_result: dict[str, Any] = {}

        while debate_round < max_debate_rounds:
            debate_round += 1
            print(f"  辩论轮次 {debate_round}/{max_debate_rounds}")

            # 如果之前有验证反馈，追加到指令中
            if debate_round > 1 and validation_result:
                feedback: str = (
                    f"\n\n【上一轮验证反馈】\n"
                    f"安全评分：{validation_result.get('safety_score', 0):.1f}/100\n"
                    f"越限详情：{json.dumps(validation_result.get('details', {}), ensure_ascii=False, indent=2)}\n"
                    f"请修正以上问题，重新生成调度方案。"
                )
                dispatch_instruction_round = dispatch_instruction + feedback
            else:
                dispatch_instruction_round = dispatch_instruction

            dispatch_plan = self.planner.plan(dispatch_instruction_round, grid_context)
            print(f"  -> 方案摘要: {dispatch_plan.get('summary', 'N/A')}")
            print(f"  -> 预期成本: {dispatch_plan.get('expected_cost', 'N/A')}")

            # ---- 步骤2: Validator验证 ----
            print(f"\n{'─' * 50}")
            print("[Validator] 验证调度方案...")

            # 构建电网模型用于验证
            grid_model_for_validation: dict[str, Any] = self._build_validation_grid_model(
                grid_context, dispatch_plan
            )

            validation_result = self.validator.validate(dispatch_plan, grid_model_for_validation)

            print(f"  -> 电压评分: {validation_result['details']['voltage']['score']:.1f}")
            print(f"  -> 线路负载评分: {validation_result['details']['line_loading']['score']:.1f}")
            print(f"  -> N-1安全评分: {validation_result['details']['n1_security']['score']:.1f}")
            print(f"  -> 频率评分: {validation_result['details']['frequency']['score']:.1f}")
            print(f"  -> ★ 综合安全评分: {validation_result['safety_score']:.1f}/100")

            # 判断是否需要回退
            if self.validator.rollback(validation_result["safety_score"]):
                print(f"  ⚠ 安全评分低于阈值({self._config.agent.safety_threshold})，需要重新规划...")
                continue
            else:
                print(f"  ✅ 安全评分达标，方案通过！")
                break

        if debate_round >= max_debate_rounds and validation_result:
            safety_score = validation_result.get("safety_score", 0)
            if safety_score < self._config.agent.safety_threshold:
                print(f"  ⚠ 达到最大辩论轮次，安全评分仍不达标({safety_score:.1f})。")
                print(f"  系统将采用最优可用方案，并标记风险。")

        # ---- 步骤2.5: 安全校验器深入检查 ----
        print(f"\n{'─' * 50}")
        print("[SafetyChecker] 深度安全校验...")

        n1_analysis = self.safety_checker.n1_contingency_analysis(
            grid_context, dispatch_plan
        )
        print(f"  -> N-1事故分析: {n1_analysis['passed']}/{n1_analysis['total_contingencies']} 通过")

        total_checks: dict[str, Any] = {
            "n1_analysis": n1_analysis,
            "voltage_violations": validation_result.get("details", {}).get("voltage", {}).get("violations", []),
            "line_violations": validation_result.get("details", {}).get("line_loading", {}).get("violations", []),
            "frequency_details": validation_result.get("details", {}).get("frequency", {}).get("details", {}),
        }
        safety_score_final: float = self.safety_checker.compute_safety_score(total_checks)
        print(f"  -> 综合安全评分（深度校验）: {safety_score_final:.1f}/100")

        # ---- 步骤3: Negotiator经济博弈 ----
        print(f"\n{'─' * 50}")
        print("[Negotiator] 多VPP经济博弈...")

        # 构建博弈所需的联合状态
        multi_vpp_state: dict[str, Any] = self._build_negotiation_state(
            grid_context, dispatch_plan
        )

        negotiation_result: dict[str, Any] = self.negotiator.negotiate(multi_vpp_state)

        print(f"  -> 博弈轮次: {negotiation_result['round']}")
        print(f"  -> 纳什均衡收敛: {negotiation_result['equilibrium_reached']}")
        print(f"  -> 总收益: {negotiation_result['total_profit']}")
        for vpp_id, schedule in negotiation_result["dispatch_schedule"].items():
            print(f"    {vpp_id}: 出力={schedule['output_mw']}MW, 报价={schedule['bid_price']}$/MWh, 收益={schedule['profit']}")

        # ---- 步骤4: Memory存储经验 ----
        print(f"\n{'─' * 50}")
        print("[Memory] 存储本次调度经验...")

        is_success: bool = validation_result.get("safety_score", 0) >= self._config.agent.safety_threshold
        entry: MemoryEntry = MemoryEntry(
            scene_features=scene_features,
            dispatch_instruction=dispatch_instruction,
            dispatch_plan=dispatch_plan,
            validation_result=validation_result,
            safety_score=validation_result.get("safety_score", 0.0),
            is_success=is_success,
        )
        self.memory.store(entry)
        print(f"  -> 经验已存储 (成功={is_success})")
        print(f"  -> 记忆库统计: {json.dumps(self.memory.get_statistics(), ensure_ascii=False)}")

        # ---- 整合最终结果 ----
        print(f"\n{'=' * 70}")
        print("  ✅ GridSynergy调度流水线完成")
        print("=" * 70)

        final_result: dict[str, Any] = {
            "system": "GridSynergy v1.0",
            "mode": "demo" if self._config.demo_mode else "llm",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": {
                "type": scenario.get("type", "normal"),
                "instruction": dispatch_instruction,
                "hour": scenario.get("hour", 0),
            },
            "grid_context": {
                "total_load": grid_context.get("total_load", 0),
                "wind_forecast": grid_context.get("wind_forecast", 0),
                "solar_forecast": grid_context.get("solar_forecast", 0),
            },
            "dispatch_plan": dispatch_plan,
            "validation": {
                "safety_score": safety_score_final,
                "debate_rounds": debate_round,
                "details": validation_result.get("details", {}),
                "n1_analysis": {
                    "total": n1_analysis.get("total_contingencies", 0),
                    "passed": n1_analysis.get("passed", 0),
                    "failed": n1_analysis.get("failed", 0),
                    "pass_rate": n1_analysis.get("pass_rate", 0.0),
                },
            },
            "negotiation": {
                "equilibrium_reached": negotiation_result["equilibrium_reached"],
                "total_profit": negotiation_result["total_profit"],
                "schedule": negotiation_result["dispatch_schedule"],
            },
            "memory": self.memory.get_statistics(),
        }

        self._run_log.append(final_result)
        return final_result

    def run_batch(self, num_scenarios: int = 5) -> list[dict[str, Any]]:
        """
        批量运行多个场景，评估系统综合表现。

        参数:
            num_scenarios: 场景数量

        返回:
            各场景结果列表
        """
        results: list[dict[str, Any]] = []
        scenario_types: list[str] = ["normal", "normal", "n1_fault", "wind_ramp", "normal"]

        for i in range(min(num_scenarios, len(scenario_types))):
            stype = scenario_types[i]
            print(f"\n{'#' * 70}")
            print(f"#  场景 {i + 1}/{num_scenarios}: {stype}")
            print(f"{'#' * 70}")

            if stype == "normal":
                scenario = self.scenario_gen.generate_normal_scenario()
            elif stype == "n1_fault":
                scenario = self.scenario_gen.generate_n1_fault_scenario()
            elif stype == "wind_ramp":
                scenario = self.scenario_gen.generate_wind_ramp_scenario()
            else:
                scenario = self.scenario_gen.generate_normal_scenario()

            result = self.run(scenario)
            results.append(result)

        # ---- 汇总报告 ----
        self._print_summary(results)
        return results

    def _print_summary(self, results: list[dict[str, Any]]) -> None:
        """打印批量运行汇总"""
        print(f"\n{'#' * 70}")
        print("#  批量运行汇总报告")
        print(f"{'#' * 70}")

        scores = [r["validation"]["safety_score"] for r in results]
        profits = [r["negotiation"]["total_profit"] for r in results]

        print(f"  场景总数: {len(results)}")
        print(f"  平均安全评分: {np.mean(scores):.1f}")
        print(f"  最低安全评分: {np.min(scores):.1f}")
        print(f"  最高安全评分: {np.max(scores):.1f}")
        print(f"  平均总收益: {np.mean(profits):.1f}")
        print(f"  记忆库条目: {results[-1]['memory']['total_entries']}")

    def _extract_scene_features(self, grid_context: dict[str, Any]) -> np.ndarray:
        """
        从电网上下文中提取场景特征向量。
        用于Memory Agent的相似场景检索。
        """
        features: list[float] = [
            float(grid_context.get("total_load", 0)) / 500.0,
            float(grid_context.get("wind_forecast", 0)) / 200.0,
            float(grid_context.get("solar_forecast", 0)) / 100.0,
            float(grid_context.get("market_price", 50.0)) / 100.0,
        ]

        # 线路平均负载率
        line_loadings: dict[str, float] = grid_context.get("line_loading", {})
        if line_loadings:
            features.append(np.mean(list(line_loadings.values())) / 100.0)
        else:
            features.append(0.5)

        # 在线机组比例
        gen_status: dict[str, str] = grid_context.get("generator_status", {})
        total_gens = len(gen_status)
        online_gens = sum(1 for s in gen_status.values() if s == "on")
        features.append(online_gens / max(total_gens, 1))

        # 补齐到配置的特征维度
        features_array = np.array(features, dtype=np.float32)
        target_dim = self._config.memory.feature_dim
        if len(features_array) < target_dim:
            features_array = np.pad(features_array, (0, target_dim - len(features_array)))
        elif len(features_array) > target_dim:
            features_array = features_array[:target_dim]

        return features_array

    def _build_validation_grid_model(
        self, grid_context: dict[str, Any], dispatch_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """构建用于验证的电网模型"""
        unit_commitment = dispatch_plan.get("unit_commitment", {})
        generator_output: dict[str, float] = {}
        for gen_name, gen_data in unit_commitment.items():
            if isinstance(gen_data, dict):
                generator_output[gen_name] = float(gen_data.get("output_mw", 0.0))

        return {
            "node_voltages": grid_context.get("node_voltages", {}),
            "line_loadings": grid_context.get("line_loading", {}),
            "total_load": grid_context.get("total_load", 250.0),
            "wind_forecast": grid_context.get("wind_forecast", 60.0),
            "solar_forecast": grid_context.get("solar_forecast", 30.0),
            "generator_output": generator_output,
            "topology_status": grid_context.get("topology_status", {}),
        }

    def _build_negotiation_state(
        self, grid_context: dict[str, Any], dispatch_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """构建Negotiator需要的联合状态"""
        unit_commitment = dispatch_plan.get("unit_commitment", {})

        vpp_states: dict[str, Any] = {}
        for vpp_id, gen_data in unit_commitment.items():
            if isinstance(gen_data, dict):
                output = float(gen_data.get("output_mw", 0.0))
                capacity = float(gen_data.get("capacity_mw", 50.0))
                vpp_states[vpp_id] = {
                    "current_output": output,
                    "remaining_capacity": max(0.0, capacity - output),
                    "renewable_available": 0.0,  # 常规机组无新能源可用
                }

        return {
            "global_load": float(grid_context.get("total_load", 250.0)),
            "market_price": float(grid_context.get("market_price", 50.0)),
            "renewable_forecast": float(grid_context.get("wind_forecast", 0.0))
                                  + float(grid_context.get("solar_forecast", 0.0)),
            "vpp_states": vpp_states,
        }

    def save_results(self, filepath: str) -> None:
        """保存运行结果到JSON文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._run_log, f, ensure_ascii=False, indent=2)
        print(f"\n[输出] 结果已保存至: {filepath}")


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    """Main entry point"""
    config = get_config()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     GridSynergy — 新能源电网自主调度系统                     ║
║     基于多智能体协同决策 (LLM + MADRL)                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    system = GridSynergySystem()

    # ---- Demo: 单场景运行 ----
    print("\n[Demo] 运行正常调度场景...\n")

    # 场景1: 正常调度
    scenario_gen = ScenarioGenerator(seed=42)
    scenario = scenario_gen.generate_normal_scenario(hour=12)   # 中午12点，负荷高峰
    result = system.run(scenario)

    # 场景2: N-1故障
    print("\n\n")
    scenario_fault = scenario_gen.generate_n1_fault_scenario(hour=14)
    result_fault = system.run(scenario_fault)

    # 保存结果
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dispatch_results.json")
    system.save_results(output_path)

    # 保存记忆库
    memory_path = os.path.join(output_dir, "memory_store.json")
    system.memory.save_to_file(memory_path)
    print(f"[输出] 记忆库已保存至: {memory_path}")

    print(f"\n{'=' * 70}")
    print(f"  运行完成！共执行 {len(system._run_log)} 个场景")
    print(f"  结果文件: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
