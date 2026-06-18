"""
测试 DeepSeek LLM 调度决策
运行: python test_llm.py
"""
from __future__ import annotations

import json
import sys
import time
sys.path.insert(0, ".")

from config import get_config
from llm.llm_client import LLMClient
from agents.planner_agent import PlannerAgent

cfg = get_config()
print(f"Demo模式: {cfg.demo_mode}")
print(f"API Key: {'已配置(' + cfg.llm.model + ')' if cfg.llm.api_key else '未配置'}")
print(f"Base URL: {cfg.llm.base_url}")
print("=" * 50)

# 测试1: 直接用 LLMClient 调用
print("\n>>> 测试1: 直接调用 DeepSeek")
client = LLMClient()
t0 = time.time()
response = client.chat(
    system_prompt="你是一位电力系统调度专家。请严格按JSON格式回复，不要其他文字。",
    user_message="总负荷280MW，风电50MW，光伏30MW，6台发电机(G1:80MW,G2:80MW,G3:50MW,G4:55MW,G5:30MW,G6:40MW)。请生成最优调度方案。输出JSON格式: {\"generators\": {\"G1\": XX, ...}, \"curtailed_wind\": X, \"curtailed_solar\": X, \"total_cost\": X}",
)
elapsed = time.time() - t0
print(f"  耗时: {elapsed:.1f}s")
print(f"  响应(前300字): {response[:300]}")

# 测试2: PlannerAgent LLM路径
print("\n>>> 测试2: PlannerAgent LLM路径")
planner = PlannerAgent()
plan = planner.plan(
    dispatch_instruction="风电出力骤降40%，请紧急调整发电计划，优先保证供电可靠性",
    grid_context={
        "total_load": 280.0,
        "wind_forecast": 30.0,
        "solar_forecast": 40.0,
        "generator_status": {f"G{i+1}": "on" for i in range(6)},
        "topology_status": {},
    },
)
print(f"  模式: {plan.get('metadata', {}).get('mode', 'unknown')}")
print(f"  摘要: {plan.get('summary', '无')}")
print(f"  预计成本: {plan.get('expected_cost', 0)}")
print(f"  机组出力: {json.dumps({k: v.get('output_mw', v) for k, v in plan.get('unit_commitment', {}).items()}, ensure_ascii=False)}")

# 测试3: 不同输入验证
print("\n>>> 测试3: 不同场景对比")
scenarios = [
    ("日前调度", "高负荷+高新能源"),
    ("光伏出力下降60%，请调整计划", "光伏骤降"),
    ("风电预测从80MW骤降到20MW，启动全部燃气备用", "风电骤降"),
]
for cmd, label in scenarios:
    plan = planner.plan(cmd, {
        "total_load": 300.0,
        "wind_forecast": 80.0 if "风电" not in cmd else 20.0,
        "solar_forecast": 60.0 if "光伏" not in cmd else 24.0,
        "generator_status": {f"G{i+1}": "on" for i in range(6)},
        "topology_status": {},
    })
    print(f"  [{label}] cost={plan.get('expected_cost',0):.0f}, summary={plan.get('summary','')[:60]}")

print("\n✅ 测试完成")
