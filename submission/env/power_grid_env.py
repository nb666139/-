"""
GridSynergy — 电网模拟环境 (PowerGridEnv)
模拟IEEE-30节点电网的基本行为，包括新能源出力的随机性和负荷波动。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config import get_config


class PowerGridEnv:
    """
    IEEE-30节点电网模拟环境

    提供 reset/step/get_state 标准接口，模拟电网运行状态。
    包含：
    - 新能源出力随机性（风速/光照波动）
    - 负荷波动的时序模式
    - 发电机出力调整
    - 拓扑重构
    """

    def __init__(self, seed: int | None = 42) -> None:
        """
        初始化电网环境。

        参数:
            seed: 随机种子
        """
        self._config = get_config()
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # 电网参数
        self.num_nodes: int = self._config.grid.num_nodes
        self.num_generators: int = self._config.grid.num_generators
        self.num_lines: int = self._config.grid.num_lines

        # 发电机数据
        self._generator_names: list[str] = ["G1", "G2", "G3", "G4", "G5", "G6"]
        self._generator_capacity: dict[str, float] = {
            "G1": 80.0, "G2": 80.0, "G3": 50.0,
            "G4": 55.0, "G5": 30.0, "G6": 40.0,
        }
        self._generator_type: dict[str, str] = {
            "G1": "thermal", "G2": "thermal", "G3": "thermal",
            "G4": "thermal", "G5": "thermal", "G6": "thermal",
        }
        self._generator_status: dict[str, str] = {}
        self._generator_output: dict[str, float] = {}

        # 拓扑数据
        self._topology: dict[str, str] = {}   # line_name -> "closed"/"open"
        self._line_limits: dict[str, float] = {}  # line_name -> max_loading_mw

        # 负荷数据
        self._node_loads: dict[str, float] = {}   # node_name -> load_mw
        self._total_load: float = 0.0

        # 新能源数据
        self._wind_capacity: float = 100.0   # 风电装机容量 MW
        self._solar_capacity: float = 60.0   # 光伏装机容量 MW
        self._wind_output: float = 0.0
        self._solar_output: float = 0.0
        self._wind_forecast: float = 0.0
        self._solar_forecast: float = 0.0

        # 市场电价
        self._market_price: float = 50.0   # $/MWh

        # 时间步
        self._time_step: int = 0

        # 初始化
        self._init_grid_model()

    def _init_grid_model(self) -> None:
        """初始化电网模型静态数据"""
        rng = self._rng

        # 发电机初始状态（全部在线）
        for gen_name in self._generator_names:
            self._generator_status[gen_name] = "on"
            # 初始出力设为容量的60%-80%
            cap = self._generator_capacity[gen_name]
            self._generator_output[gen_name] = round(cap * rng.uniform(0.55, 0.75), 2)

        # 线路初始化（IEEE-30标准有41条线路）
        for i in range(1, self.num_lines + 1):
            line_name: str = f"L{i}"
            self._topology[line_name] = "closed"
            # 线路额定容量 50-150 MVA
            self._line_limits[line_name] = round(rng.uniform(50.0, 150.0), 1)

        # 节点负荷初始化（30节点系统总负荷约283 MW，在此基础上有±10%波动）
        base_loads: list[float] = [
            0.0, 21.7, 2.4, 7.6, 94.2, 0.0, 22.8, 30.0, 0.0, 5.8,
            0.0, 11.2, 0.0, 6.2, 8.2, 3.5, 9.0, 3.2, 9.5, 2.2,
            17.5, 0.0, 3.2, 8.7, 0.0, 3.5, 0.0, 0.0, 2.4, 10.6,
        ]
        for i in range(self.num_nodes):
            node_name: str = f"B{i + 1}"
            self._node_loads[node_name] = round(
                base_loads[i] * rng.uniform(0.9, 1.1), 2
            )

        self._total_load = round(sum(self._node_loads.values()), 2)

        # 新能源初始出力
        self._wind_output = round(self._wind_capacity * rng.uniform(0.3, 0.7), 2)
        self._solar_output = round(self._solar_capacity * rng.uniform(0.2, 0.8), 2)
        self._wind_forecast = self._wind_output
        self._solar_forecast = self._solar_output

    # ========================================================================
    # 环境标准接口
    # ========================================================================

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """
        重置环境到初始状态。

        返回:
            当前环境状态字典
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._time_step = 0
        self._init_grid_model()
        return self.get_state()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        执行一个调度动作，环境向前推进一步。

        参数:
            action: 调度动作字典
                - unit_commitment: 各发电机出力调整
                - topology_switches: 开关操作
                - renewable_curtailment: 新能源弃电

        返回:
            (新状态, 奖励, 是否终止, 额外信息)
        """
        self._time_step += 1

        # 1. 应用发电机出力调整
        unit_commitment: dict[str, dict[str, Any]] = action.get("unit_commitment", {})
        for gen_name, gen_action in unit_commitment.items():
            if gen_name in self._generator_output:
                new_output: float = float(gen_action.get("output_mw", 0.0))
                # 应用爬坡率限制（每分钟5%容量）
                max_ramp: float = self._generator_capacity[gen_name] * 0.05
                prev_output: float = self._generator_output[gen_name]
                if abs(new_output - prev_output) > max_ramp:
                    new_output = prev_output + max_ramp * np.sign(new_output - prev_output)
                self._generator_output[gen_name] = max(0.0, min(new_output, self._generator_capacity[gen_name]))

        # 2. 应用拓扑调整
        topology_switches: dict[str, str] = action.get("topology_switches", {})
        for line_name, switch_state in topology_switches.items():
            if line_name in self._topology:
                self._topology[line_name] = switch_state

        # 3. 应用新能源弃电
        curtailment: dict[str, float] = action.get("renewable_curtailment", {})
        curtail_wind: float = curtailment.get("wind_mw", 0.0)
        curtail_solar: float = curtailment.get("solar_mw", 0.0)
        self._wind_output = max(0.0, self._wind_forecast - curtail_wind)
        self._solar_output = max(0.0, self._solar_forecast - curtail_solar)

        # 4. 更新新能源预测（引入随机性）
        wind_change: float = self._rng.normal(0.0, 5.0)
        solar_change: float = self._rng.normal(0.0, 2.0)
        self._wind_forecast = max(0.0, min(self._wind_capacity, self._wind_forecast + wind_change))
        self._solar_forecast = max(0.0, min(self._solar_capacity, self._solar_forecast + solar_change))

        # 5. 更新负荷（引入波动）
        load_change_pct: float = self._rng.normal(0.0, 0.02)  # 2%标准差
        for node in self._node_loads:
            self._node_loads[node] *= (1.0 + load_change_pct)
            self._node_loads[node] = round(self._node_loads[node], 2)
        self._total_load = round(sum(self._node_loads.values()), 2)

        # 6. 更新市场电价
        price_change: float = self._rng.normal(0.0, 2.0)
        self._market_price = max(20.0, min(100.0, self._market_price + price_change))

        # 7. 计算奖励（综合供电质量和经济性）
        reward: float = self._compute_reward()

        # 8. 判断终止
        done: bool = self._time_step >= 24  # 模拟24小时

        return self.get_state(), reward, done, {"time_step": self._time_step}

    def get_state(self) -> dict[str, Any]:
        """
        获取当前电网完整状态。

        返回:
            包含所有状态变量的字典
        """
        return {
            "time_step": self._time_step,
            "total_load": self._total_load,
            "generator_status": dict(self._generator_status),
            "generator_output": dict(self._generator_output),
            "generator_capacity": dict(self._generator_capacity),
            "topology_status": dict(self._topology),
            "line_limits": dict(self._line_limits),
            "line_loadings": self._estimate_line_loadings(),
            "node_voltages": self._estimate_node_voltages(),
            "wind_forecast": round(self._wind_forecast, 2),
            "solar_forecast": round(self._solar_forecast, 2),
            "wind_capacity": self._wind_capacity,
            "solar_capacity": self._solar_capacity,
            "market_price": round(self._market_price, 2),
            "node_loads": dict(self._node_loads),
            "total_generation": round(
                sum(self._generator_output.values()) + self._wind_output + self._solar_output, 2
            ),
        }

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _estimate_line_loadings(self) -> dict[str, float]:
        """
        估算线路负载率（简化潮流模型）。
        实际项目中应使用完整潮流计算。
        """
        line_loadings: dict[str, float] = {}
        total_gen: float = sum(self._generator_output.values()) + self._wind_output + self._solar_output
        base_loading: float = self._total_load / max(total_gen, 1.0)

        for line_name, limit in self._line_limits.items():
            # 简化：各线路负载与总传输需求成比例，加随机噪声
            raw_loading_mw: float = limit * base_loading * self._rng.uniform(0.5, 0.95)
            line_loadings[line_name] = round(raw_loading_mw / limit * 100, 1)

        return line_loadings

    def _estimate_node_voltages(self) -> dict[str, float]:
        """
        估算节点电压（标幺值，简化模型）。
        """
        voltages: dict[str, float] = {}
        for i in range(self.num_nodes):
            node_name: str = f"B{i + 1}"
            # 电压在0.96-1.04之间正态分布，基准值1.00
            voltage: float = self._rng.normal(1.0, 0.015)
            voltage = max(0.93, min(1.07, voltage))
            voltages[node_name] = round(voltage, 4)

        # 理想情况（均衡态）下调高电压中位值
        for node in voltages:
            voltages[node] = round(voltages[node] * 1.002, 4)

        return voltages

    def _compute_reward(self) -> float:
        """
        计算环境奖励。

        奖励 = 经济性奖励 + 安全性奖励 - 惩罚项
        """
        # 经济性：总发电成本
        total_cost: float = 0.0
        cost_coeff: dict[str, float] = {
            "G1": 20.0, "G2": 22.0, "G3": 35.0,
            "G4": 30.0, "G5": 28.0, "G6": 40.0,
        }
        for gen_name, output in self._generator_output.items():
            total_cost += output * cost_coeff.get(gen_name, 30.0)

        # 新能源成本最低
        total_cost += self._wind_output * 5.0 + self._solar_output * 3.0

        # 经济性奖励（成本越低越好）
        economic_reward: float = -total_cost / 1000.0  # 缩放

        # 安全性奖励：基于线路负载和电压
        line_loadings: dict[str, float] = self._estimate_line_loadings()
        overload_penalty: float = sum(
            max(0.0, ld - 90.0) / 10.0 for ld in line_loadings.values()
        )
        voltage_penalty: float = sum(
            abs(v - 1.0) * 100.0 for v in self._estimate_node_voltages().values()
            if v < 0.95 or v > 1.05
        )

        # 出力平衡奖励
        gen_total: float = sum(self._generator_output.values()) + self._wind_output + self._solar_output
        balance_reward: float = -abs(gen_total - self._total_load * 1.03) / 10.0

        reward: float = economic_reward + balance_reward - overload_penalty - voltage_penalty
        return round(reward, 4)

    # ========================================================================
    # 便捷属性
    # ========================================================================

    @property
    def total_load(self) -> float:
        return self._total_load

    @property
    def wind_forecast(self) -> float:
        return self._wind_forecast

    @property
    def solar_forecast(self) -> float:
        return self._solar_forecast

    @property
    def market_price(self) -> float:
        return self._market_price

    @property
    def time_step(self) -> int:
        return self._time_step
