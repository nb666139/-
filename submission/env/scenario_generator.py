"""
GridSynergy — 场景生成器 (ScenarioGenerator)
生成训练和测试场景，包括正常运行、N-1故障、风电骤降、光伏波动等场景。
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from config import get_config


class ScenarioGenerator:
    """
    场景生成器

    生成多种电网运行场景用于测试调度系统的鲁棒性。
    支持：
    - 正常运行场景（日负荷曲线 + 新能源时序）
    - N-1故障场景（线路/发电机故障）
    - 风电骤降场景（风速突变）
    - 光伏波动场景（云层遮挡）
    """

    def __init__(self, seed: int = 42) -> None:
        """初始化场景生成器"""
        self._config = get_config()
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # 典型日负荷曲线（24小时，归一化至均值1.0）
        self._daily_load_profile: np.ndarray = np.array([
            0.65, 0.60, 0.58, 0.56, 0.58, 0.65,
            0.75, 0.88, 0.95, 0.98, 1.00, 0.97,
            0.93, 0.95, 0.98, 1.00, 1.05, 1.08,
            1.10, 1.05, 0.98, 0.92, 0.82, 0.72,
        ], dtype=np.float64)

        # 典型风电日出力曲线（归一化）
        self._wind_profile: np.ndarray = np.array([
            0.85, 0.88, 0.90, 0.87, 0.82, 0.78,
            0.72, 0.68, 0.65, 0.62, 0.60, 0.58,
            0.56, 0.55, 0.54, 0.56, 0.60, 0.65,
            0.72, 0.78, 0.82, 0.85, 0.88, 0.86,
        ], dtype=np.float64)

        # 典型光伏日出力曲线（归一化，中午峰值）
        self._solar_profile: np.ndarray = np.array([
            0.00, 0.00, 0.00, 0.00, 0.02, 0.10,
            0.25, 0.45, 0.65, 0.80, 0.90, 0.95,
            0.92, 0.85, 0.75, 0.55, 0.35, 0.15,
            0.05, 0.01, 0.00, 0.00, 0.00, 0.00,
        ], dtype=np.float64)

        # 基础负荷（MW）
        self._base_load: float = 283.4

        # 新能源装机容量
        self._wind_capacity: float = 100.0
        self._solar_capacity: float = 60.0

    def generate_normal_scenario(self, hour: int | None = None) -> dict[str, Any]:
        """
        生成正常运行场景。

        参数:
            hour: 指定小时（0-23），None则随机选择

        返回:
            包含电网状态的场景字典
        """
        if hour is None:
            hour = self._rng.integers(0, 24)

        # 获取该小时的负荷和新能源出力
        load_factor: float = self._daily_load_profile[hour]
        wind_factor: float = self._wind_profile[hour]
        solar_factor: float = self._solar_profile[hour]

        # 加入随机波动（±10%）
        load: float = self._base_load * load_factor * self._rng.uniform(0.9, 1.1)
        wind: float = self._wind_capacity * wind_factor * self._rng.uniform(0.85, 1.15)
        solar: float = self._solar_capacity * solar_factor * self._rng.uniform(0.8, 1.2)

        # 构建电网上下文
        grid_context: dict[str, Any] = self._build_base_grid_context(
            total_load=round(load, 2),
            wind_forecast=round(wind, 2),
            solar_forecast=round(solar, 2),
        )

        # 自然语言调度指令
        instructions: list[str] = [
            f"当前为小时{hour}:00，请按照经济调度原则生成最优发电计划。"
            f"风电预测{wind:.1f}MW，光伏预测{solar:.1f}MW，总负荷{load:.1f}MW。",
            f"小时{hour}:00正常调度，请优先消纳新能源，保持电网安全稳定运行。",
            f"生成小时{hour}:00的调度方案，确保功率平衡和N-1安全。",
        ]

        return {
            "type": "normal",
            "hour": hour,
            "dispatch_instruction": self._rng.choice(instructions),
            "grid_context": grid_context,
            "load_factor": round(load_factor, 3),
            "wind_factor": round(wind_factor, 3),
            "solar_factor": round(solar_factor, 3),
        }

    def generate_n1_fault_scenario(self, hour: int | None = None) -> dict[str, Any]:
        """
        生成N-1故障场景。
        随机选择一条线路或一台发电机模拟故障。

        返回:
            包含故障信息的场景字典
        """
        base_scenario: dict[str, Any] = self.generate_normal_scenario(hour)

        # 随机选择故障类型
        fault_type: str = self._rng.choice(["line_outage", "generator_trip"])

        if fault_type == "line_outage":
            fault_line: str = f"L{self._rng.integers(1, 42)}"
            fault_desc: str = (
                f"【紧急】线路 {fault_line} 发生永久性故障跳闸！"
                f"请重新规划调度方案，确保系统在N-1状态下安全运行。"
            )
            fault_info: dict[str, Any] = {
                "type": "line_outage",
                "element": fault_line,
                "severity": "high",
            }
        else:
            fault_gen: str = self._rng.choice(["G1", "G2", "G3", "G4", "G5", "G6"])
            fault_desc = (
                f"【紧急】发电机 {fault_gen} 发生故障脱网！"
                f"请重新规划调度方案，调配其他机组补充功率缺额。"
            )
            fault_info = {
                "type": "generator_trip",
                "element": fault_gen,
                "lost_capacity": self._rng.uniform(30.0, 80.0),
                "severity": "high",
            }

        # 更新指令和上下文
        base_scenario["dispatch_instruction"] = fault_desc
        base_scenario["type"] = "n1_fault"
        base_scenario["fault_info"] = fault_info

        # 在拓扑中标记故障线路
        if fault_type == "line_outage":
            base_scenario["grid_context"]["topology_status"][fault_line] = "open"

        return base_scenario

    def generate_wind_ramp_scenario(self, hour: int | None = None) -> dict[str, Any]:
        """
        生成风电骤降场景。
        模拟风速突然大幅下降，风电出力短时间内急剧减少。

        返回:
            包含风电骤降信息的场景字典
        """
        base_scenario: dict[str, Any] = self.generate_normal_scenario(hour)

        wind_forecast: float = float(base_scenario["grid_context"]["wind_forecast"])
        # 风电骤降至原先的30%-50%
        ramp_ratio: float = self._rng.uniform(0.3, 0.5)
        new_wind: float = round(wind_forecast * ramp_ratio, 2)
        lost_wind: float = round(wind_forecast - new_wind, 2)

        ramp_desc: str = (
            f"【紧急】风电出力骤降！预测风速在15分钟内急剧下降，"
            f"风电出力从 {wind_forecast:.1f}MW 降至 {new_wind:.1f}MW，"
            f"功率缺额约 {lost_wind:.1f}MW。"
            f"请紧急调度常规机组增加出力，或启动需求侧响应。"
        )

        base_scenario["dispatch_instruction"] = ramp_desc
        base_scenario["type"] = "wind_ramp"
        base_scenario["grid_context"]["wind_forecast"] = new_wind
        base_scenario["ramp_info"] = {
            "original_wind": wind_forecast,
            "new_wind": new_wind,
            "lost_wind": lost_wind,
            "ramp_pct": round(ramp_ratio * 100, 1),
        }

        return base_scenario

    def generate_scenario_batch(
        self, scenario_types: list[str] | None = None, num_each: int = 5
    ) -> list[dict[str, Any]]:
        """
        批量生成多种类型场景。

        参数:
            scenario_types: 场景类型列表，None则生成全部类型
            num_each: 每种类型生成的数量

        返回:
            场景列表
        """
        if scenario_types is None:
            scenario_types = ["normal", "n1_fault", "wind_ramp"]

        scenarios: list[dict[str, Any]] = []
        for scenario_type in scenario_types:
            for _ in range(num_each):
                hour: int = self._rng.integers(0, 24)
                if scenario_type == "normal":
                    scenarios.append(self.generate_normal_scenario(hour))
                elif scenario_type == "n1_fault":
                    scenarios.append(self.generate_n1_fault_scenario(hour))
                elif scenario_type == "wind_ramp":
                    scenarios.append(self.generate_wind_ramp_scenario(hour))

        return scenarios

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _build_base_grid_context(
        self,
        total_load: float,
        wind_forecast: float,
        solar_forecast: float,
    ) -> dict[str, Any]:
        """
        构建基础电网上下文数据。
        """
        # 发电机初始状态
        generator_status: dict[str, str] = {
            "G1": "on", "G2": "on", "G3": "on",
            "G4": "on", "G5": "on", "G6": "on",
        }

        # 线路拓扑（默认全部闭合）
        topology_status: dict[str, str] = {
            f"L{i}": "closed" for i in range(1, 42)
        }

        # 估算线路负载（基于总负荷的简化分配）
        total_gen_capacity: float = sum([80, 80, 50, 55, 30, 40])
        line_loading: dict[str, float] = {}
        for i in range(1, 42):
            # 各线路负载率基于总负荷的随机分布
            base_rate: float = (total_load / (total_gen_capacity * 3))  # 3倍容量留有裕度
            line_loading[f"L{i}"] = round(
                base_rate * 100 * self._rng.uniform(0.4, 1.2), 1
            )

        # 估算节点电压
        node_voltages: dict[str, float] = {}
        for i in range(30):
            node_voltages[f"B{i+1}"] = round(self._rng.normal(1.0, 0.015), 4)

        return {
            "total_load": total_load,
            "wind_forecast": wind_forecast,
            "solar_forecast": solar_forecast,
            "generator_status": generator_status,
            "topology_status": topology_status,
            "line_loading": line_loading,
            "node_voltages": node_voltages,
            "market_price": round(self._rng.uniform(40.0, 60.0), 2),
        }
