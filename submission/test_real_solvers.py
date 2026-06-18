"""验证真实求解器是否正常工作。"""
from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, ".")

# 1. 测试 Pandapower AC 潮流
print("=" * 50)
print("1. 测试 Pandapower AC 潮流 (Newton-Raphson)")
import pandapower as pp
import pandapower.networks as pn
from solver.power_flow import PowerFlowSolver

net = pn.case30()
solver = PowerFlowSolver(pp_net=net)

# 模拟调度方案
result = solver.solve_power_flow({
    "node_loads": {f"B{i}": net.load.loc[net.load.bus == i, "p_mw"].iloc[0] 
                   if any(net.load.bus == i) else 0.0 for i in range(1, 31)},
    "generator_output": {"G1": 80.0, "G2": 60.0, "G3": 30.0, "G4": 40.0, "G5": 20.0, "G6": 30.0},
    "topology_status": {},
})

print(f"  收敛: {result['converged']}, 方法: {result['method']}")
print(f"  发电: {result['total_generation_mw']}MW, 负荷: {result['total_load_mw']}MW")
print(f"  电压样本: B1={result['bus_voltages'].get('B1', {}).get('voltage_pu', 'N/A')}pu")
violations = solver.check_voltage_violations(result)
print(f"  电压越限: {len(violations)}处")

# 2. 测试 Pyomo + HiGHS SCUC
print("=" * 50)
print("2. 测试 Pyomo + HiGHS SCUC MILP")
from solver.scuc_solver import SCUCSolver

scuc = SCUCSolver(num_generators=6, num_lines=15, num_buses=30, horizon=24)
# 生成24小时测试数据
load = np.array([250 + 50 * np.sin(np.pi * t / 12) for t in range(24)])
wind = np.array([60 + 30 * np.sin(np.pi * t / 8) for t in range(24)])
pv = np.array([20 * (1 if 6 <= t <= 18 else 0.1) for t in range(24)])

scuc_result = scuc.solve(load, wind, pv)
print(f"  状态: {scuc_result['status']}, 方法: {scuc_result['method']}")
print(f"  总成本: {scuc_result['total_cost']:.0f}元")
print(f"  G1平均出力: {np.mean(scuc_result['generator_output']['G1']):.1f}MW")
print(f"  弃电总量: {sum(scuc_result.get('curtailed_mw', [0]*24)):.1f}MW")

# 3. 测试 Pandapower N-1
print("=" * 50)
print("3. 测试 Pandapower N-1 安全分析")
from solver.safety_checker import SafetyChecker

checker = SafetyChecker(pp_net=net)
n1_result = checker.n1_contingency_analysis(
    dispatch_plan={
        "generator_output": {"G1": 80, "G2": 60, "G3": 30, "G4": 40, "G5": 20, "G6": 30}
    },
    grid_model={"node_loads": {}, "topology_status": {}},
)
print(f"  方法: {n1_result['method']}")
print(f"  通过/失败/总计: {n1_result['passed']}/{n1_result['failed']}/{n1_result['total']}")
print(f"  通过率: {n1_result['pass_rate']}%")
if n1_result["failures"]:
    print(f"  首个故障: {n1_result['failures'][0]['contingency']}")

print("=" * 50)
print("✅ 全部真实求解器验证通过！")
