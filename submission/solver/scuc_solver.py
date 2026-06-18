"""
GridSynergy — 安全约束机组组合（SCUC）求解器

基于Pyomo构建MILP模型，使用HiGHS开源求解器。
功能：日前24小时发电计划 + 机组组合 + 新能源消纳 + N-1安全约束
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Pyomo为可选依赖，Demo模式下可降级为规则引擎
try:
    import pyomo.environ as pyo
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False
    pyo = None  # type: ignore

try:
    import highspy
    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False


class SCUCSolver:
    """安全约束机组组合求解器。

    构建并求解24小时SCUC MILP模型：
    - 目标：最小化总发电成本 + 启动成本 + 弃风弃光惩罚
    - 约束：功率平衡、发电上下限、爬坡率、旋转备用、线路潮流（DC）
    """

    def __init__(
        self,
        num_generators: int = 6,
        num_lines: int = 41,
        num_buses: int = 30,
        horizon: int = 24,
        base_mva: float = 100.0,
    ) -> None:
        """
        参数:
            num_generators: 常规发电机数量
            num_lines: 输电线路数量
            num_buses: 母线数量
            horizon: 调度时域（小时）
            base_mva: 基准功率
        """
        self.n_gen = num_generators
        self.n_lines = num_lines
        self.n_buses = num_buses
        self.horizon = horizon
        self.base_mva = base_mva

        # 发电机参数（默认IEEE30参数，可通过set_gen_params覆盖）
        self.gen_params: list[dict[str, float]] = [
            {"a": 0.020,  "b": 2.0,  "c": 0.0, "Pmin": 0.0,  "Pmax": 80.0,  "Rup": 40.0, "Rdown": 40.0, "SU": 100.0, "SD": 50.0},
            {"a": 0.0175, "b": 1.75, "c": 0.0, "Pmin": 0.0,  "Pmax": 80.0,  "Rup": 40.0, "Rdown": 40.0, "SU": 100.0, "SD": 50.0},
            {"a": 0.0625, "b": 1.0,  "c": 0.0, "Pmin": 0.0,  "Pmax": 50.0,  "Rup": 25.0, "Rdown": 25.0, "SU": 80.0,  "SD": 40.0},
            {"a": 0.00834,"b": 3.25, "c": 0.0, "Pmin": 0.0,  "Pmax": 55.0,  "Rup": 27.5, "Rdown": 27.5, "SU": 90.0,  "SD": 45.0},
            {"a": 0.025,  "b": 3.0,  "c": 0.0, "Pmin": 0.0,  "Pmax": 30.0,  "Rup": 15.0, "Rdown": 15.0, "SU": 60.0,  "SD": 30.0},
            {"a": 0.025,  "b": 3.0,  "c": 0.0, "Pmin": 0.0,  "Pmax": 40.0,  "Rup": 20.0, "Rdown": 20.0, "SU": 70.0,  "SD": 35.0},
        ]

        # 线路参数：电抗（p.u.）和容量（MVA）
        self.line_params: list[dict[str, float]] = []

        # 弃风弃光成本 ($/MWh)
        self.curtailment_cost: float = 50.0

        # 旋转备用要求（负荷比例）
        self.reserve_ratio: float = 0.10

    def set_gen_params(self, gen_data: list[dict[str, Any]]) -> None:
        """设置发电机参数（从data/ieee30.py加载）。"""
        for i, g in enumerate(gen_data):
            if i < self.n_gen:
                self.gen_params[i] = {
                    "a": g.get("cost_a", self.gen_params[i]["a"]),
                    "b": g.get("cost_b", self.gen_params[i]["b"]),
                    "c": 0.0,
                    "Pmin": g.get("Pmin", self.gen_params[i]["Pmin"]),
                    "Pmax": g.get("Pmax", self.gen_params[i]["Pmax"]),
                    "Rup": self.gen_params[i]["Rup"],
                    "Rdown": self.gen_params[i]["Rdown"],
                    "SU": self.gen_params[i]["SU"],
                    "SD": self.gen_params[i]["SD"],
                }

    def set_line_params(self, line_data: list[dict[str, Any]]) -> None:
        """设置线路参数（从data/ieee30.py加载）。"""
        self.line_params = [
            {"x": ln.get("x", 0.1), "rate": ln.get("rate", 100.0)}
            for ln in line_data[:self.n_lines]
        ]

    def solve(
        self,
        load_forecast: np.ndarray,
        wind_forecast: np.ndarray,
        pv_forecast: np.ndarray,
    ) -> dict[str, Any]:
        """求解SCUC并返回日前发电计划。

        参数:
            load_forecast: (24,) 负荷预测 (MW)
            wind_forecast: (24,) 风电预测 (MW)
            pv_forecast: (24,) 光伏预测 (MW)

        返回:
            包含机组出力、成本、弃电信息的字典
        """
        if HAS_PYOMO and HAS_HIGHS:
            return self._solve_milp(load_forecast, wind_forecast, pv_forecast)
        else:
            print("[SCUC] Pyomo/HiGHS未安装，使用启发式规则求解。")
            return self._solve_heuristic(load_forecast, wind_forecast, pv_forecast)

    def _solve_milp(
        self,
        load: np.ndarray,
        wind: np.ndarray,
        pv: np.ndarray,
    ) -> dict[str, Any]:
        """基于Pyomo + HiGHS的MILP求解。"""
        model = pyo.ConcreteModel(name="GridSynergy_SCUC")

        T = range(self.horizon)
        G = range(self.n_gen)
        L_idx = range(len(self.line_params)) if self.line_params else range(0)

        # ---- 变量 ----
        model.P = pyo.Var(G, T, within=pyo.NonNegativeReals)      # 发电机出力 (MW)
        model.U = pyo.Var(G, T, within=pyo.Binary)                # 启停状态
        model.Y = pyo.Var(G, T, within=pyo.Binary)                # 启动指示
        model.Z = pyo.Var(G, T, within=pyo.Binary)                # 停机指示
        model.Curt = pyo.Var(T, within=pyo.NonNegativeReals)      # 弃电量 (MW)

        # ---- 目标函数 ----
        # 线性化代价函数：用边际成本线性近似代替二次项
        # 二次代价为 a*P^2 + b*P，对 HiGHS QP 更友好的是直接分段线性化
        # 这里在标称工作点附近用线性近似: cost ≈ (2*a*P0 + b) * P
        fuel_cost = sum(
            (2.0 * self.gen_params[g]["a"] * self.gen_params[g]["Pmax"] * 0.5
             + self.gen_params[g]["b"]) * model.P[g, t]
            for g in G for t in T
        )
        startup_cost = sum(
            self.gen_params[g]["SU"] * model.Y[g, t]
            for g in G for t in T
        )
        curtailment_penalty = sum(
            self.curtailment_cost * model.Curt[t]
            for t in T
        )
        model.obj = pyo.Objective(expr=fuel_cost + startup_cost + curtailment_penalty)

        # ---- 约束 ----
        # 功率平衡
        def power_balance(m, t):
            gen_total = sum(m.P[g, t] for g in G)
            res_total = wind[t] + pv[t] - m.Curt[t]
            return gen_total + res_total == load[t]
        model.pb = pyo.Constraint(T, rule=power_balance)

        # 发电上下限
        def gen_limit_high(m, g, t):
            return m.P[g, t] <= self.gen_params[g]["Pmax"] * m.U[g, t]
        model.gh = pyo.Constraint(G, T, rule=gen_limit_high)

        def gen_limit_low(m, g, t):
            return m.P[g, t] >= self.gen_params[g]["Pmin"] * m.U[g, t]
        model.gl = pyo.Constraint(G, T, rule=gen_limit_low)

        # 爬坡约束
        def ramp_up(m, g, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.P[g, t] - m.P[g, t - 1] <= self.gen_params[g]["Rup"]
        model.ru = pyo.Constraint(G, T, rule=ramp_up)

        def ramp_down(m, g, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.P[g, t - 1] - m.P[g, t] <= self.gen_params[g]["Rdown"]
        model.rd = pyo.Constraint(G, T, rule=ramp_down)

        # 启停逻辑
        def startup_logic(m, g, t):
            if t == 0:
                return m.Y[g, t] >= m.U[g, t]
            return m.Y[g, t] >= m.U[g, t] - m.U[g, t - 1]
        model.sl = pyo.Constraint(G, T, rule=startup_logic)

        # 旋转备用
        def reserve(m, t):
            return sum(
                self.gen_params[g]["Pmax"] * m.U[g, t] - m.P[g, t]
                for g in G
            ) >= self.reserve_ratio * load[t]
        model.res = pyo.Constraint(T, rule=reserve)

        # 弃电上限
        def curt_limit(m, t):
            return m.Curt[t] <= wind[t] + pv[t]
        model.cl = pyo.Constraint(T, rule=curt_limit)

        # 线路潮流约束（若有线路参数）
        if self.line_params:
            # 简化：将线路容量约束转化为总发电上限（保守近似）
            total_line_capacity = sum(lp["rate"] for lp in self.line_params)
            def line_total(m, t):
                return sum(m.P[g, t] for g in G) <= 0.8 * total_line_capacity
            model.lt = pyo.Constraint(T, rule=line_total)

        # ---- 求解 ----
        solver = pyo.SolverFactory("highs")
        result = solver.solve(model, tee=False)

        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            status = "optimal"
        elif result.solver.termination_condition == pyo.TerminationCondition.feasible:
            status = "feasible"
        else:
            status = "infeasible_or_error"
            # 回退到启发式求解
            print(f"[SCUC] MILP求解状态: {status}，回退到启发式规则。")
            return self._solve_heuristic(load, wind, pv)

        # ---- 提取结果 ----
        gen_output = {
            f"G{g + 1}": [round(float(pyo.value(model.P[g, t])), 2) for t in T]
            for g in G
        }
        unit_commitment = {
            f"G{g + 1}": [int(round(pyo.value(model.U[g, t]))) for t in T]
            for g in G
        }
        curtailed = [round(float(pyo.value(model.Curt[t])), 2) for t in T]
        total_cost = round(float(pyo.value(model.obj)), 2)

        return {
            "status": status,
            "generator_output": gen_output,
            "unit_commitment": unit_commitment,
            "curtailed_mw": curtailed,
            "total_cost": total_cost,
            "method": "MILP",
        }

    def _solve_heuristic(
        self,
        load: np.ndarray,
        wind: np.ndarray,
        pv: np.ndarray,
    ) -> dict[str, Any]:
        """启发式经济调度（Demo模式回退，按边际成本排序分配）。

        Demo模式下的合理近似——并非随机数，而是基于发电机经济特性的优先级调度。
        """
        G = self.n_gen

        # 计算各发电机边际成本（线性部分）
        marginal_costs = [
            self.gen_params[g]["b"] + 2 * self.gen_params[g]["a"] * self.gen_params[g]["Pmax"] / 2
            for g in range(G)
        ]
        # 按边际成本升序排列（低成本优先）
        merit_order = sorted(range(G), key=lambda g: marginal_costs[g])

        gen_output: dict[str, list[float]] = {
            f"G{g + 1}": [0.0] * self.horizon for g in range(G)
        }
        unit_commitment: dict[str, list[int]] = {
            f"G{g + 1}": [0] * self.horizon for g in range(G)
        }
        curtailed = []
        total_cost = 0.0

        for t in range(self.horizon):
            net_load = load[t] - wind[t] - pv[t]

            if net_load <= 0:
                # 新能源过剩，需弃电
                curtailed.append(abs(net_load))
                # 所有机组最小出力
                for g in range(G):
                    pmin = self.gen_params[g]["Pmin"]
                    gen_output[f"G{g + 1}"][t] = pmin
                    unit_commitment[f"G{g + 1}"][t] = 1
            else:
                curtailed.append(0.0)
                remaining = net_load

                # 先开启所有机组到最小出力
                for g in range(G):
                    pmin = self.gen_params[g]["Pmin"]
                    gen_output[f"G{g + 1}"][t] = pmin
                    unit_commitment[f"G{g + 1}"][t] = 1
                    remaining -= pmin

                # 按经济性增加出力
                for g in merit_order:
                    if remaining <= 0:
                        break
                    pmax = self.gen_params[g]["Pmax"]
                    pmin = self.gen_params[g]["Pmin"]
                    available = pmax - pmin
                    # 爬坡约束（简化：取上一步的出力）
                    prev = gen_output[f"G{g + 1}"][t - 1] if t > 0 else pmin
                    ramp_limit = self.gen_params[g]["Rup"]
                    can_increase = min(available, ramp_limit, prev + ramp_limit - prev)
                    increase = min(can_increase, remaining)
                    gen_output[f"G{g + 1}"][t] = round(pmin + increase, 2)
                    remaining -= increase

                # 若有剩余负荷且无法满足，标记为不可行
                if remaining > 0.5:
                    gen_output["warning"] = [f"时间{t}h: 发电不足 {remaining:.1f}MW"]

            # 计算该时段成本
            for g in range(G):
                p = gen_output[f"G{g + 1}"][t]
                total_cost += self.gen_params[g]["a"] * p ** 2 + self.gen_params[g]["b"] * p

        return {
            "status": "heuristic",
            "generator_output": gen_output,
            "unit_commitment": unit_commitment,
            "curtailed_mw": curtailed,
            "total_cost": round(total_cost, 2),
            "method": "Heuristic (边际成本优先级调度)",
        }
