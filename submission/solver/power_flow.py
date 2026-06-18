"""
GridSynergy — 潮流计算求解器 (PowerFlowSolver)

支持两种模式：
1. Pandapower真实潮流：完整Newton-Raphson AC潮流计算
2. DC近似 + 统计噪声：Demo模式回退（保留原逻辑作为过渡）
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from config import get_config

# 尝试导入Pandapower
try:
    import pandapower as pp
    HAS_PANDAPOWER = True
except ImportError:
    HAS_PANDAPOWER = False


class PowerFlowSolver:
    """潮流计算求解器。

    优先使用Pandapower进行真实AC潮流计算；
    若Pandapower未安装，回退到DC近似（Demo模式）。
    """

    def __init__(self, pp_net=None) -> None:
        """
        参数:
            pp_net: Pandapower网络对象（可选，None则使用内置IEEE30）
        """
        self._config = get_config()
        self._num_nodes: int = self._config.grid.num_nodes
        self._base_mva: float = self._config.grid.base_power
        self._rng: np.random.Generator = np.random.default_rng(42)

        # Pandapower网络
        self._pp_net = pp_net
        if self._pp_net is None and HAS_PANDAPOWER:
            try:
                import pandapower.networks as pn
                self._pp_net = pn.case30()
            except Exception:
                self._pp_net = None

    def set_network(self, pp_net) -> None:
        """设置Pandapower网络对象。"""
        self._pp_net = pp_net

    def solve_power_flow(self, grid_model: dict[str, Any]) -> dict[str, Any]:
        """执行潮流计算。

        参数:
            grid_model: 电网模型字典（含node_loads、generator_output、topology_status）

        返回:
            潮流计算结果（电压、相角、线路潮流）
        """
        if HAS_PANDAPOWER and self._pp_net is not None:
            try:
                return self._solve_with_pandapower(grid_model)
            except Exception as e:
                print(f"[PowerFlow] Pandapower求解失败({e})，回退到DC近似。")
        return self._solve_dc_approximation(grid_model)

    def _solve_with_pandapower(self, grid_model: dict[str, Any]) -> dict[str, Any]:
        """使用Pandapower运行Newton-Raphson AC潮流。"""
        net = copy.deepcopy(self._pp_net)
        generator_output = grid_model.get("generator_output", {})
        topology = grid_model.get("topology_status", {})
        node_loads = grid_model.get("node_loads", {})

        # 设置发电机出力
        for i in range(len(net.gen)):
            gen_name = f"G{i + 1}"
            if gen_name in generator_output:
                net.gen.loc[i, "p_mw"] = generator_output[gen_name]

        # 设置负荷
        if node_loads:
            for i in range(len(net.load)):
                bus = net.load.loc[i, "bus"]
                load_name = f"B{bus}"
                if load_name in node_loads:
                    net.load.loc[i, "p_mw"] = node_loads[load_name]

        # 处理线路开断
        if topology:
            for i in range(len(net.line)):
                from_bus = int(net.line.loc[i, "from_bus"])
                to_bus = int(net.line.loc[i, "to_bus"])
                line_key = f"L{from_bus}-{to_bus}"
                if topology.get(line_key) == "open":
                    net.line.loc[i, "in_service"] = False

        # 运行Newton-Raphson
        try:
            pp.runpp(net, algorithm="nr", numba=False)
            converged = True
        except pp.LoadflowNotConverged:
            converged = False

        # 提取电压结果
        bus_voltages = {}
        if converged:
            for i in range(len(net.res_bus)):
                v = float(net.res_bus.loc[i, "vm_pu"])
                a = float(net.res_bus.loc[i, "va_degree"])
                bus_voltages[f"B{i + 1}"] = {
                    "voltage_pu": round(v, 4),
                    "angle_deg": round(a, 2),
                }

        # 提取线路潮流（兼容 pandapower 2.x 和 3.x 列名）
        line_flows = {}
        if converged:
            q_col = "q_from_mvar" if "q_from_mvar" in net.res_line.columns else "q_from_mw"
            for i in range(len(net.res_line)):
                from_bus = int(net.line.loc[i, "from_bus"])
                to_bus = int(net.line.loc[i, "to_bus"])
                p_from = float(net.res_line.loc[i, "p_from_mw"])
                q_from = float(net.res_line.loc[i, q_col])
                loading = float(net.res_line.loc[i, "loading_percent"])
                line_flows[f"L{from_bus}-{to_bus}"] = {
                    "from_bus": from_bus,
                    "to_bus": to_bus,
                    "flow_mva": round(np.sqrt(p_from ** 2 + q_from ** 2), 2),
                    "loading_pct": round(loading, 1),
                    "status": "closed",
                }

        return {
            "converged": converged,
            "n_buses": len(net.bus),
            "bus_voltages": bus_voltages,
            "line_flows": line_flows,
            "total_generation_mw": round(float(net.res_gen["p_mw"].sum()), 2) if converged else 0.0,
            "total_load_mw": round(float(net.res_load["p_mw"].sum()), 2) if converged else 0.0,
            "method": "Pandapower AC (Newton-Raphson)",
        }

    def _solve_dc_approximation(self, grid_model: dict[str, Any]) -> dict[str, Any]:
        """DC近似潮流（Demo回退，保留原有逻辑）。"""
        node_loads = grid_model.get("node_loads", {})
        generator_output = grid_model.get("generator_output", {})
        topology = grid_model.get("topology_status", {})

        num_nodes = self._num_nodes

        # 注入功率
        P_injection = np.zeros(num_nodes, dtype=np.float64)
        gen_buses = [0, 1, 4, 7, 10, 12]  # IEEE30发电机母线（0-indexed）
        for i, bus_idx in enumerate(gen_buses):
            if bus_idx < num_nodes:
                gen_name = f"G{i + 1}"
                output = generator_output.get(gen_name, 0.0)
                P_injection[bus_idx] = output / self._base_mva

        # 负荷
        for idx in range(num_nodes):
            node_name = f"B{idx + 1}"
            load = node_loads.get(node_name, 0.0)
            P_injection[idx] -= load / self._base_mva

        # B矩阵
        B = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        from data.ieee30 import LINE_DATA as ieee30_lines
        for ln in ieee30_lines:
            f = ln["from"] - 1
            t = ln["to"] - 1
            if f >= num_nodes or t >= num_nodes:
                continue
            line_name = ln["name"]
            if topology.get(line_name) == "open":
                continue
            x = ln["x"]
            if x == 0:
                x = 0.001  # 变压器电抗非零
            b = 1.0 / x
            B[f, t] = -b
            B[t, f] = -b
            B[f, f] += b
            B[t, t] += b

        ref_idx = 0
        B_reduced = np.delete(np.delete(B, ref_idx, axis=0), ref_idx, axis=1)
        P_reduced = np.delete(P_injection, ref_idx)

        try:
            B_reg = B_reduced + np.eye(len(B_reduced)) * 1e-6
            theta_reduced = np.linalg.solve(B_reg, P_reduced)
        except np.linalg.LinAlgError:
            theta_reduced = np.linalg.lstsq(B_reduced, P_reduced, rcond=None)[0]

        theta_full = np.zeros(num_nodes, dtype=np.float64)
        theta_full[1:] = theta_reduced
        theta_deg = theta_full * 180.0 / np.pi
        voltages_pu = np.ones(num_nodes) + self._rng.normal(0, 0.005, num_nodes)
        voltages_pu = np.clip(voltages_pu, 0.94, 1.06)

        # 线路潮流
        line_flows = {}
        for ln in ieee30_lines[:12]:
            f = ln["from"] - 1
            t = ln["to"] - 1
            if f >= num_nodes or t >= num_nodes:
                continue
            line_name = ln["name"]
            if topology.get(line_name) == "open":
                line_flows[line_name] = {"from_bus": ln["from"], "to_bus": ln["to"],
                                           "flow_mva": 0.0, "status": "open"}
                continue
            x = ln["x"] if ln["x"] != 0 else 0.001
            angle_diff = (theta_deg[f] - theta_deg[t]) * np.pi / 180.0
            flow = angle_diff / x * self._base_mva
            line_flows[line_name] = {
                "from_bus": ln["from"], "to_bus": ln["to"],
                "flow_mva": round(flow, 2), "status": "closed",
            }

        bus_voltages = {
            f"B{i + 1}": {"voltage_pu": round(float(voltages_pu[i]), 4),
                          "angle_deg": round(float(theta_deg[i]), 2)}
            for i in range(num_nodes)
        }

        total_gen = sum(generator_output.values())
        total_load = sum(node_loads.values())

        return {
            "converged": True,
            "n_buses": num_nodes,
            "bus_voltages": bus_voltages,
            "line_flows": line_flows,
            "total_generation_mw": round(total_gen, 2),
            "total_load_mw": round(total_load, 2),
            "power_losses_mw": round(abs(total_gen - total_load), 2),
            "method": "DC Approximation (Demo回退)",
        }

    def check_voltage_violations(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        """检查电压越限（0.95-1.05 p.u.）。"""
        violations = []
        v_min = self._config.grid.voltage_min
        v_max = self._config.grid.voltage_max
        bus_voltages = results.get("bus_voltages", {})
        for bus_name, bus_data in bus_voltages.items():
            voltage = float(bus_data.get("voltage_pu", 1.0))
            if voltage < v_min:
                violations.append({"bus": bus_name, "type": "undervoltage",
                                   "voltage_pu": voltage, "limit": v_min,
                                   "deviation": round(v_min - voltage, 4)})
            elif voltage > v_max:
                violations.append({"bus": bus_name, "type": "overvoltage",
                                   "voltage_pu": voltage, "limit": v_max,
                                   "deviation": round(voltage - v_max, 4)})
        return violations

    def check_line_overload(
        self, results: dict[str, Any], limits: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        """检查线路过载。"""
        violations = []
        line_flows = results.get("line_flows", {})
        for line_name, flow_data in line_flows.items():
            flow_mva = float(flow_data.get("flow_mva", 0.0))
            limit_mva = 100.0
            if limits and line_name in limits:
                limit_mva = limits[line_name]
            loading_pct = flow_mva / limit_mva * 100.0 if limit_mva > 0 else 0
            if loading_pct > 100.0:
                violations.append({"line": line_name, "flow_mva": round(flow_mva, 2),
                                   "limit_mva": limit_mva, "loading_pct": round(loading_pct, 1),
                                   "overload_mva": round(flow_mva - limit_mva, 2)})
        return violations
