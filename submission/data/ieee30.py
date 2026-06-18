"""
IEEE 30节点标准测试系统数据模型

基于 MATPOWER case30.m 数据，提供Pandapower网络构建和参数访问。
数据来源：https://github.com/MATPOWER/matpower/blob/master/data/case30.m

该模块支持两种模式：
1. 有pandapower时：构建真实pp网络对象
2. 无pandapower时：以字典形式提供原始参数（Demo模式回退）
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ============================================================================
# IEEE 30节点系统原始参数（来自MATPOWER case30.m）
# ============================================================================

# 基准值
BASE_MVA: float = 100.0
BASE_KV: float = 135.0

# 发电机数据：bus, Pg(MW), Qg(MVar), Qmax, Qmin, Vg(p.u.), mBase, status, Pmax, Pmin
# 6台发电机（G1-G6），成本函数为二次型：a * P^2 + b * P + c
GENERATOR_DATA: list[dict[str, Any]] = [
    {"bus": 1,  "Pmax": 80.0,  "Pmin": 0.0,  "cost_a": 0.020, "cost_b": 2.0,  "name": "G1"},
    {"bus": 2,  "Pmax": 80.0,  "Pmin": 0.0,  "cost_a": 0.0175,"cost_b": 1.75, "name": "G2"},
    {"bus": 5,  "Pmax": 50.0,  "Pmin": 0.0,  "cost_a": 0.0625,"cost_b": 1.0,  "name": "G3"},
    {"bus": 8,  "Pmax": 55.0,  "Pmin": 0.0,  "cost_a": 0.00834,"cost_b": 3.25, "name": "G4"},
    {"bus": 11, "Pmax": 30.0,  "Pmin": 0.0,  "cost_a": 0.025, "cost_b": 3.0,  "name": "G5"},
    {"bus": 13, "Pmax": 40.0,  "Pmin": 0.0,  "cost_a": 0.025, "cost_b": 3.0,  "name": "G6"},
]

# 发电机爬坡速率（MW/h）
GENERATOR_RAMP_RATES: dict[str, float] = {
    "G1": 40.0, "G2": 40.0, "G3": 25.0,
    "G4": 27.5, "G5": 15.0, "G6": 20.0,
}

# 线路数据：from_bus, to_bus, r(p.u.), x(p.u.), b(p.u.), rateA(MVA)
# IEEE 30节点共有41条线路
LINE_DATA: list[dict[str, Any]] = [
    {"from": 1, "to": 2,  "r": 0.0192, "x": 0.0575, "b": 0.0528, "rate": 130.0, "name": "L1-2"},
    {"from": 1, "to": 3,  "r": 0.0452, "x": 0.1852, "b": 0.0408, "rate": 130.0, "name": "L1-3"},
    {"from": 2, "to": 4,  "r": 0.0570, "x": 0.1737, "b": 0.0368, "rate": 65.0,  "name": "L2-4"},
    {"from": 3, "to": 4,  "r": 0.0132, "x": 0.0379, "b": 0.0084, "rate": 130.0, "name": "L3-4"},
    {"from": 2, "to": 5,  "r": 0.0472, "x": 0.1983, "b": 0.0418, "rate": 130.0, "name": "L2-5"},
    {"from": 2, "to": 6,  "r": 0.0581, "x": 0.1763, "b": 0.0374, "rate": 65.0,  "name": "L2-6"},
    {"from": 4, "to": 6,  "r": 0.0119, "x": 0.0414, "b": 0.0090, "rate": 90.0,  "name": "L4-6"},
    {"from": 5, "to": 7,  "r": 0.0460, "x": 0.1160, "b": 0.0204, "rate": 70.0,  "name": "L5-7"},
    {"from": 6, "to": 7,  "r": 0.0267, "x": 0.0820, "b": 0.0170, "rate": 130.0, "name": "L6-7"},
    {"from": 6, "to": 8,  "r": 0.0120, "x": 0.0420, "b": 0.0090, "rate": 32.0,  "name": "L6-8"},
    {"from": 6, "to": 9,  "r": 0.0,    "x": 0.2080, "b": 0.0,    "rate": 65.0,  "name": "L6-9"},
    {"from": 6, "to": 10, "r": 0.0,    "x": 0.5560, "b": 0.0,    "rate": 32.0,  "name": "L6-10"},
    {"from": 9, "to": 11, "r": 0.0,    "x": 0.2080, "b": 0.0,    "rate": 65.0,  "name": "L9-11"},
    {"from": 9, "to": 10, "r": 0.0,    "x": 0.1100, "b": 0.0,    "rate": 65.0,  "name": "L9-10"},
    {"from": 4, "to": 12, "r": 0.0,    "x": 0.2560, "b": 0.0,    "rate": 65.0,  "name": "L4-12"},
    {"from": 12, "to": 13, "r": 0.0,   "x": 0.1400, "b": 0.0,    "rate": 65.0,  "name": "L12-13"},
    {"from": 12, "to": 14, "r": 0.1231, "x": 0.2559, "b": 0.0,   "rate": 32.0,  "name": "L12-14"},
    {"from": 12, "to": 15, "r": 0.0662, "x": 0.1304, "b": 0.0,   "rate": 32.0,  "name": "L12-15"},
    {"from": 12, "to": 16, "r": 0.0945, "x": 0.1987, "b": 0.0,   "rate": 32.0,  "name": "L12-16"},
    {"from": 14, "to": 15, "r": 0.2210, "x": 0.1997, "b": 0.0,   "rate": 16.0,  "name": "L14-15"},
    {"from": 16, "to": 17, "r": 0.0824, "x": 0.1932, "b": 0.0,   "rate": 16.0,  "name": "L16-17"},
    {"from": 15, "to": 18, "r": 0.1070, "x": 0.2185, "b": 0.0,   "rate": 16.0,  "name": "L15-18"},
    {"from": 18, "to": 19, "r": 0.0639, "x": 0.1292, "b": 0.0,   "rate": 16.0,  "name": "L18-19"},
    {"from": 19, "to": 20, "r": 0.0340, "x": 0.0680, "b": 0.0,   "rate": 32.0,  "name": "L19-20"},
    {"from": 10, "to": 20, "r": 0.0936, "x": 0.2090, "b": 0.0,   "rate": 32.0,  "name": "L10-20"},
    {"from": 10, "to": 17, "r": 0.0324, "x": 0.0845, "b": 0.0,   "rate": 32.0,  "name": "L10-17"},
    {"from": 10, "to": 21, "r": 0.0348, "x": 0.0749, "b": 0.0,   "rate": 32.0,  "name": "L10-21"},
    {"from": 10, "to": 22, "r": 0.0727, "x": 0.1499, "b": 0.0,   "rate": 32.0,  "name": "L10-22"},
    {"from": 21, "to": 22, "r": 0.0116, "x": 0.0236, "b": 0.0,   "rate": 32.0,  "name": "L21-22"},
    {"from": 15, "to": 23, "r": 0.1000, "x": 0.2020, "b": 0.0,   "rate": 16.0,  "name": "L15-23"},
    {"from": 22, "to": 24, "r": 0.1150, "x": 0.1790, "b": 0.0,   "rate": 16.0,  "name": "L22-24"},
    {"from": 23, "to": 24, "r": 0.1320, "x": 0.2700, "b": 0.0,   "rate": 16.0,  "name": "L23-24"},
    {"from": 24, "to": 25, "r": 0.1885, "x": 0.3292, "b": 0.0,   "rate": 16.0,  "name": "L24-25"},
    {"from": 25, "to": 26, "r": 0.2544, "x": 0.3800, "b": 0.0,   "rate": 16.0,  "name": "L25-26"},
    {"from": 25, "to": 27, "r": 0.1093, "x": 0.2087, "b": 0.0,   "rate": 16.0,  "name": "L25-27"},
    {"from": 28, "to": 27, "r": 0.0,    "x": 0.3960, "b": 0.0,   "rate": 65.0,  "name": "L28-27"},
    {"from": 27, "to": 29, "r": 0.2198, "x": 0.4153, "b": 0.0,   "rate": 16.0,  "name": "L27-29"},
    {"from": 27, "to": 30, "r": 0.3202, "x": 0.6027, "b": 0.0,   "rate": 16.0,  "name": "L27-30"},
    {"from": 29, "to": 30, "r": 0.2399, "x": 0.4533, "b": 0.0,   "rate": 16.0,  "name": "L29-30"},
    {"from": 8,  "to": 28, "r": 0.0636, "x": 0.2000, "b": 0.0428, "rate": 32.0, "name": "L8-28"},
    {"from": 6,  "to": 28, "r": 0.0169, "x": 0.0599, "b": 0.0130, "rate": 32.0, "name": "L6-28"},
]

# 负荷数据：bus, Pd(MW), Qd(MVar)，总计约189.2 MW
LOAD_DATA: list[dict[str, Any]] = [
    {"bus": 1,  "Pd": 0.0,   "name": "Load1"},
    {"bus": 2,  "Pd": 21.7,  "name": "Load2"},
    {"bus": 3,  "Pd": 2.4,   "name": "Load3"},
    {"bus": 4,  "Pd": 7.6,   "name": "Load4"},
    {"bus": 5,  "Pd": 94.2,  "name": "Load5"},
    {"bus": 6,  "Pd": 0.0,   "name": "Load6"},
    {"bus": 7,  "Pd": 22.8,  "name": "Load7"},
    {"bus": 8,  "Pd": 30.0,  "name": "Load8"},
    {"bus": 9,  "Pd": 0.0,   "name": "Load9"},
    {"bus": 10, "Pd": 5.8,   "name": "Load10"},
    {"bus": 11, "Pd": 0.0,   "name": "Load11"},
    {"bus": 12, "Pd": 11.2,  "name": "Load12"},
    {"bus": 13, "Pd": 0.0,   "name": "Load13"},
    {"bus": 14, "Pd": 6.2,   "name": "Load14"},
    {"bus": 15, "Pd": 8.2,   "name": "Load15"},
    {"bus": 16, "Pd": 3.5,   "name": "Load16"},
    {"bus": 17, "Pd": 9.0,   "name": "Load17"},
    {"bus": 18, "Pd": 3.2,   "name": "Load18"},
    {"bus": 19, "Pd": 9.5,   "name": "Load19"},
    {"bus": 20, "Pd": 2.2,   "name": "Load20"},
    {"bus": 21, "Pd": 17.5,  "name": "Load21"},
    {"bus": 22, "Pd": 0.0,   "name": "Load22"},
    {"bus": 23, "Pd": 3.2,   "name": "Load23"},
    {"bus": 24, "Pd": 8.7,   "name": "Load24"},
    {"bus": 25, "Pd": 0.0,   "name": "Load25"},
    {"bus": 26, "Pd": 3.5,   "name": "Load26"},
    {"bus": 27, "Pd": 0.0,   "name": "Load27"},
    {"bus": 28, "Pd": 0.0,   "name": "Load28"},
    {"bus": 29, "Pd": 2.4,   "name": "Load29"},
    {"bus": 30, "Pd": 10.6,  "name": "Load30"},
]

# 新能源接入位置（风电）
WIND_BUSES: list[int] = [7, 14, 24]
WIND_CAPACITY: dict[int, float] = {7: 30.0, 14: 25.0, 24: 25.0}  # 总计80MW

# 新能源接入位置（光伏）
PV_BUSES: list[int] = [21, 30]
PV_CAPACITY: dict[int, float] = {21: 20.0, 30: 20.0}  # 总计40MW

# 总负荷（MW）
TOTAL_LOAD: float = sum(item["Pd"] for item in LOAD_DATA)


def build_pandapower_network():
    """构建IEEE 30节点Pandapower网络对象。

    返回:
        pandapower.auxiliary.pandapowerNet 网络对象
    """
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise ImportError(
            "需要安装pandapower: pip install pandapower>=2.14.0"
        )

    # pandapower内置IEEE 30节点案例
    net = pn.case30()

    # 添加新能源场站（风电）
    for bus, capacity in WIND_CAPACITY.items():
        pp.create_sgen(
            net, bus, p_mw=0.0, q_mvar=0.0,
            max_p_mw=capacity, min_p_mw=0.0,
            type="WP", name=f"Wind_{bus}"
        )

    # 添加新能源场站（光伏）
    for bus, capacity in PV_CAPACITY.items():
        pp.create_sgen(
            net, bus, p_mw=0.0, q_mvar=0.0,
            max_p_mw=capacity, min_p_mw=0.0,
            type="PV", name=f"PV_{bus}"
        )

    return net


def get_grid_params() -> dict[str, Any]:
    """以字典形式返回IEEE 30节点系统参数（Demo模式回退用）。

    返回:
        包含发电机、线路、负荷、新能源参数的综合字典
    """
    return {
        "base_mva": BASE_MVA,
        "base_kv": BASE_KV,
        "n_buses": 30,
        "n_generators": len(GENERATOR_DATA),
        "n_lines": len(LINE_DATA),
        "total_load_mw": TOTAL_LOAD,
        "total_wind_mw": sum(WIND_CAPACITY.values()),
        "total_pv_mw": sum(PV_CAPACITY.values()),
        "generators": GENERATOR_DATA,
        "lines": LINE_DATA,
        "loads": LOAD_DATA,
        "wind_buses": WIND_BUSES,
        "wind_capacity": WIND_CAPACITY,
        "pv_buses": PV_BUSES,
        "pv_capacity": PV_CAPACITY,
        "ramp_rates": GENERATOR_RAMP_RATES,
    }
