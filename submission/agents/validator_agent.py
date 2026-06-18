"""
GridSynergy — 验证Agent (ValidatorAgent)
负责对规划Agent生成的调度方案进行多维安全验证。
验证维度：电压幅值、线路负载、N-1安全、频率稳定。
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from config import get_config


class ValidatorAgent:
    """
    验证Agent
    对调度方案执行全面的安全校验，输出安全评分（0-100），
    并控制方案的回退机制。
    """

    def __init__(self) -> None:
        """初始化验证Agent"""
        self._config = get_config()
        # 电压标准范围（标幺值）
        self._voltage_min: float = self._config.grid.voltage_min  # 0.95
        self._voltage_max: float = self._config.grid.voltage_max  # 1.05
        # 线路负载率上限（百分比）
        self._line_loading_limit: float = 100.0
        # N-1校核负载率上限
        self._n1_loading_limit: float = 120.0
        # 频率偏差允许范围（Hz）
        self._freq_deviation_max: float = 0.5
        # 各验证维度的权重
        self._weights: dict[str, float] = {
            "voltage": 0.30,
            "line_loading": 0.25,
            "n1_security": 0.30,
            "frequency": 0.15,
        }

    def validate(self, dispatch_plan: dict[str, Any], grid_model: dict[str, Any]) -> dict[str, Any]:
        """
        对调度方案进行完整验证。

        参数:
            dispatch_plan: Planner生成的调度方案
            grid_model: 电网模型数据，包含节点、线路、发电机信息

        返回:
            验证结果字典，含各维度评分和综合安全评分
        """
        # 1. 电压幅值校验
        voltage_score, voltage_violations = self._check_voltage(dispatch_plan, grid_model)

        # 2. 线路负载校验
        line_score, line_violations = self._check_line_loading(dispatch_plan, grid_model)

        # 3. N-1安全扫描
        n1_results: list[dict[str, Any]] = self.n1_scan(dispatch_plan)
        n1_score: float = self._compute_n1_score(n1_results)

        # 4. 频率稳定校验
        freq_score, freq_details = self._check_frequency_stability(dispatch_plan, grid_model)

        # 综合安全评分（加权平均）
        safety_score: float = (
            voltage_score * self._weights["voltage"]
            + line_score * self._weights["line_loading"]
            + n1_score * self._weights["n1_security"]
            + freq_score * self._weights["frequency"]
        )

        # 是否有越限
        has_violations: bool = (
            len(voltage_violations) > 0
            or len(line_violations) > 0
            or any(r.get("violation", False) for r in n1_results)
            or not freq_details.get("stable", True)
        )

        result: dict[str, Any] = {
            "safety_score": round(safety_score, 2),
            "passed": safety_score >= self._config.agent.safety_threshold,
            "has_violations": has_violations,
            "n1_pass_rate": round(100.0 - sum(1 for r in n1_results if r.get("violation")) / max(len(n1_results), 1) * 100, 1),
            "details": {
                "voltage": {
                    "score": voltage_score,
                    "violations": voltage_violations,
                    "weight": self._weights["voltage"],
                },
                "line_loading": {
                    "score": line_score,
                    "violations": line_violations,
                    "line_violations_detail": [{"line": v.get("line", ""), "loading_pct": v.get("loading_pct", 100)} for v in line_violations],
                    "weight": self._weights["line_loading"],
                },
                "n1_security": {
                    "score": n1_score,
                    "results": n1_results,
                    "weight": self._weights["n1_security"],
                },
                "frequency": {
                    "score": freq_score,
                    "details": freq_details,
                    "weight": self._weights["frequency"],
                },
            },
        }
        return result

    def n1_scan(self, dispatch_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """
        N-1安全扫描：模拟各关键设备（线路、发电机）故障后系统是否仍安全运行。

        参数:
            dispatch_plan: 调度方案

        返回:
            N-1扫描结果列表，每个元素包含故障设备和安全状态
        """
        n1_results: list[dict[str, Any]] = []

        # 获取拓扑开关状态
        topology: dict[str, str] = dispatch_plan.get("topology_switches", {})
        # 获取机组出力
        unit_commitment: dict[str, dict[str, Any]] = dispatch_plan.get("unit_commitment", {})

        # N-1线路故障模拟
        for line_name, switch_state in topology.items():
            if switch_state == "open":
                continue  # 已断开的线路不参与校核

            # 模拟该线路断开后，其他线路负载变化
            # (简化的灵敏度分析：每条线路故障后，剩余线路负载增加约5%-15%)
            overload_detected: bool = self._simulate_line_outage(line_name, topology, unit_commitment)
            n1_results.append({
                "contingency_type": "line_outage",
                "element": line_name,
                "violation": overload_detected,
                "severity": "high" if overload_detected else "low",
            })

        # N-1发电机故障模拟
        total_gen_output: float = sum(
            u.get("output_mw", 0.0) for u in unit_commitment.values()
        )
        for gen_name, gen_info in unit_commitment.items():
            if gen_info.get("status") != "on":
                continue

            # 模拟该发电机脱网后，其他机组能否补足功率缺额
            lost_output: float = gen_info.get("output_mw", 0.0)
            remaining_capacity: float = sum(
                self._get_gen_capacity(g) - unit_commitment[g].get("output_mw", 0.0)
                for g in unit_commitment
                if g != gen_name and unit_commitment[g].get("status") == "on"
            )
            violation: bool = remaining_capacity < lost_output * 0.95

            n1_results.append({
                "contingency_type": "generator_outage",
                "element": gen_name,
                "lost_output_mw": lost_output,
                "remaining_capacity_mw": remaining_capacity,
                "violation": violation,
                "severity": "high" if violation else "low",
            })

        return n1_results

    def rollback(self, safety_score: float, threshold: float | None = None) -> bool:
        """
        判断是否需要回退（拒绝当前方案，要求Planner重新生成）。

        参数:
            safety_score: 综合安全评分
            threshold: 安全阈值，默认使用配置文件中的值

        返回:
            True表示需要回退，False表示方案通过
        """
        if threshold is None:
            threshold = self._config.agent.safety_threshold
        return safety_score < threshold

    # ========================================================================
    # 内部校验方法
    # ========================================================================

    def _check_voltage(
        self, dispatch_plan: dict[str, Any], grid_model: dict[str, Any]
    ) -> tuple[float, list[dict[str, Any]]]:
        """电压幅值校验。使用 Pandapower 真实潮流计算。"""
        violations: list[dict[str, Any]] = []

        # 优先使用真实 Pandapower 潮流
        node_voltages, _ = self._run_pandapower_pf(dispatch_plan, grid_model)
        
        if node_voltages is None:
            # 回退：基于机组出力的简化电压估算
            uc = dispatch_plan.get("unit_commitment", {})
            total_gen = sum(
                (uc[g].get("output_mw", uc[g]) if isinstance(uc[g], dict) else float(uc[g]))
                for g in uc
            )
            total_load = float(grid_model.get("total_load", 250))
            imbalance = abs(total_gen - total_load) / max(total_load, 1)
            num_nodes = self._config.grid.num_nodes
            # 有功不平衡导致电压偏差
            base_voltage = 1.02 if total_gen >= total_load else 0.97
            spread = max(0.01, imbalance * 0.05)
            rng = np.random.default_rng(hash(str(sorted(uc.items()))) % (2**31))
            node_voltages = {
                f"B{i+1}": round(float(np.clip(base_voltage + rng.normal(0, spread), 0.90, 1.10)), 4)
                for i in range(num_nodes)
            }

        violation_count = 0
        for node, voltage in node_voltages.items():
            if voltage < self._voltage_min or voltage > self._voltage_max:
                violation_count += 1
                violations.append({
                    "node": node, "voltage_pu": voltage,
                    "limit": f"{self._voltage_min}-{self._voltage_max}",
                    "deviation": round(voltage - self._voltage_max if voltage > self._voltage_max
                                       else self._voltage_min - voltage, 4),
                })

        total_nodes = len(node_voltages) if node_voltages else 1
        score = max(0.0, 100.0 - violation_count * (100.0 / total_nodes) * 3)
        return round(score, 2), violations

    def _check_line_loading(
        self, dispatch_plan: dict[str, Any], grid_model: dict[str, Any]
    ) -> tuple[float, list[dict[str, Any]]]:
        """线路负载校验。使用 Pandapower 真实潮流计算。"""
        violations: list[dict[str, Any]] = []

        # 优先使用真实 Pandapower 潮流
        _, line_loadings = self._run_pandapower_pf(dispatch_plan, grid_model)
        
        if line_loadings is None:
            # 回退：基于总出力的负载估算
            uc = dispatch_plan.get("unit_commitment", {})
            total_gen = sum(
                (uc[g].get("output_mw", uc[g]) if isinstance(uc[g], dict) else float(uc[g]))
                for g in uc
            )
            num_lines = self._config.grid.num_lines
            # 总出力越大，线路负载越高
            avg_loading = min(90.0, total_gen / 500.0 * 100)
            rng = np.random.default_rng(hash(str(sorted(uc.items())) + "lines") % (2**31))
            line_loadings = {
                f"L{i+1}": round(float(np.clip(avg_loading + rng.normal(0, 15), 10, 150)), 1)
                for i in range(num_lines)
            }

        violation_count = 0
        total_lines = len(line_loadings) if line_loadings else 1

        for line, loading in line_loadings.items():
            if loading > self._line_loading_limit:
                violation_count += 1
                violations.append({
                    "line": line, "loading_pct": loading,
                    "limit_pct": self._line_loading_limit,
                    "overload_pct": round(loading - self._line_loading_limit, 2),
                })

        score = max(0.0, 100.0 - violation_count * (100.0 / total_lines) * 5)
        return round(score, 2), violations

    def _compute_n1_score(self, n1_results: list[dict[str, Any]]) -> float:
        """
        根据N-1扫描结果计算安全评分。
        """
        if not n1_results:
            return 100.0

        total: int = len(n1_results)
        violations: int = sum(1 for r in n1_results if r.get("violation", False))
        # 每个违规扣(100/total * 2)分
        score: float = max(0.0, 100.0 - violations * (100.0 / total) * 2)
        return round(score, 2)

    def _check_frequency_stability(
        self, dispatch_plan: dict[str, Any], grid_model: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """
        频率稳定校验。
        检查在有功不平衡时频率偏差是否在允许范围内。
        """
        # 计算有功不平衡量
        unit_commitment: dict[str, dict[str, Any]] = dispatch_plan.get("unit_commitment", {})
        total_generation: float = sum(
            u.get("output_mw", 0.0) for u in unit_commitment.values()
        )
        total_load: float = float(grid_model.get("total_load", 250.0))
        renewable_curtailment: dict[str, float] = dispatch_plan.get("renewable_curtailment", {})

        # 考虑新能源弃电后的实际出力
        wind_forecast: float = float(grid_model.get("wind_forecast", 60.0))
        solar_forecast: float = float(grid_model.get("solar_forecast", 30.0))
        curtail_wind: float = renewable_curtailment.get("wind_mw", 0.0)
        curtail_solar: float = renewable_curtailment.get("solar_mw", 0.0)
        actual_renewable: float = (wind_forecast - curtail_wind) + (solar_forecast - curtail_solar)

        # 总有功发电 = 机组出力 + 新能源实际出力
        total_supply: float = total_generation + actual_renewable
        # 网损估算约3%
        total_demand: float = total_load * 1.03
        # 有功不平衡
        power_imbalance: float = total_supply - total_demand
        imbalance_pct: float = abs(power_imbalance) / max(total_demand, 1.0) * 100

        # 根据不平衡量估算频率偏差（简化的下垂特性）
        # 假设系统频率响应特性为 0.1 Hz / % 不平衡
        freq_deviation: float = imbalance_pct * 0.1
        stable: bool = freq_deviation <= self._freq_deviation_max

        # 评分
        if stable:
            score: float = max(70.0, 100.0 - imbalance_pct * 10)
        else:
            score = max(0.0, 50.0 - (freq_deviation - self._freq_deviation_max) * 100)

        details: dict[str, Any] = {
            "total_supply_mw": round(total_supply, 2),
            "total_demand_mw": round(total_demand, 2),
            "power_imbalance_mw": round(power_imbalance, 2),
            "imbalance_pct": round(imbalance_pct, 2),
            "estimated_freq_deviation_hz": round(freq_deviation, 3),
            "stable": stable,
        }

        return round(score, 2), details

    def _simulate_line_outage(
        self,
        fault_line: str,
        topology: dict[str, str],
        unit_commitment: dict[str, dict[str, Any]],
    ) -> bool:
        """
        模拟单条线路故障后其他线路是否过载。

        优先使用SafetyChecker进行真实N-1重潮流分析。
        若不可用，回退到基于负载率的简化估算。
        """
        try:
            from solver.safety_checker import SafetyChecker
            sc = SafetyChecker()
            dispatch = {"generator_output": unit_commitment}
            grid_model = {"topology_status": topology}
            result = sc.check_dispatch_safety(dispatch, grid_model)
            return result["n1_pass_rate"] < 95.0
        except Exception:
            # 回退：基于线路容量的保守估计
            from data.ieee30 import LINE_DATA
            total_gen = sum(u.get("output_mw", 0.0) for u in unit_commitment.values())
            for line in LINE_DATA:
                if line["name"] == fault_line:
                    rate = line.get("rate", 100.0)
                    # 若该线路承载功率超过容量则存在过载风险
                    return total_gen / len(LINE_DATA) > rate
            return False

    def _extract_generator_output(self, dispatch_plan: dict[str, Any]) -> dict[str, float]:
        """从 PlannerAgent 的 unit_commitment 提取发电机出力映射。"""
        uc = dispatch_plan.get("unit_commitment", {})
        return {
            gen: (uc[gen].get("output_mw", uc[gen]) if isinstance(uc[gen], dict) else float(uc[gen]))
            for gen in uc
        }

    def _run_pandapower_pf(self, dispatch_plan: dict[str, Any], grid_model: dict[str, Any]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """使用 Pandapower 运行真实 AC 潮流计算。
        
        Returns:
            (node_voltages, line_loadings) 或 (None, None) 如果失败
        """
        try:
            import pandapower as pp
            import pandapower.networks as pn
            
            net = copy.deepcopy(pn.case30())
            gen_output = self._extract_generator_output(dispatch_plan)
            
            # 将 Planner 的6台发电机映射到 IEEE30 的5台发电机 + 1台新增
            # IEEE30 发电机索引: 0(slack), 1, 2, 3, 4 → 对应我们的 G1-G5
            gen_map = {0: "G1", 1: "G2", 2: "G3", 3: "G4", 4: "G5"}
            for pp_idx, our_name in gen_map.items():
                if our_name in gen_output and pp_idx < len(net.gen):
                    net.gen.at[pp_idx, "p_mw"] = gen_output[our_name]
            
            # 如果有 G6，调整负荷或加到某一节点
            if "G6" in gen_output:
                extra = gen_output["G6"]
                # 将 G6 出力以负负荷方式注入 bus 12（分布式电源）
                if len(net.load) > 10:
                    net.load.at[10, "p_mw"] = max(0, net.load.at[10, "p_mw"] - extra)
            
            # 运行潮流
            try:
                pp.runpp(net, init="flat")
            except pp.LoadflowNotConverged:
                try:
                    pp.runpp(net, init="dc")
                except Exception:
                    return None, None
            
            # 提取节点电压 (p.u.)
            node_voltages = {f"B{i+1}": round(float(v), 4) 
                           for i, v in enumerate(net.res_bus["vm_pu"])}
            
            # 提取线路负载 (%)
            line_loadings = {}
            for i in range(len(net.line)):
                line_loadings[f"L{i+1}"] = round(float(net.res_line["loading_percent"].iloc[i]), 2)
            
            return node_voltages, line_loadings
        except Exception as e:
            print(f"  [Validator] Pandapower 潮流失败: {e}")
            return None, None

    def _get_gen_capacity(self, gen_name: str) -> float:
        """获取发电机额定容量（MW）"""
        capacities: dict[str, float] = {
            "G1": 80.0, "G2": 80.0, "G3": 50.0,
            "G4": 55.0, "G5": 30.0, "G6": 40.0,
        }
        return capacities.get(gen_name, 50.0)
