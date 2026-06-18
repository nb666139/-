"""
测试 GridSynergy 四Agent 协同链路
运行: python test_collab.py
"""
from __future__ import annotations

import json
import sys
sys.path.insert(0, ".")

from config import get_config
from agents.planner_agent import PlannerAgent
from agents.validator_agent import ValidatorAgent
from agents.negotiator_agent import NegotiatorAgent
from agents.memory_agent import MemoryAgent

cfg = get_config()
AGENT_ICON = {"记忆": "🧠", "规划": "📋", "验证": "🛡️", "博弈": "🤝"}

def log(agent: str, msg: str):
    icon = AGENT_ICON.get(agent, "➡️")
    print(f"  {icon} [{agent}Agent] {msg}")

print("=" * 60)
print("  GridSynergy 四Agent 协同链路测试")
print(f"  LLM: {cfg.llm.model}  |  API: {'已配置' if cfg.llm.api_key else '未配置'}")
print("=" * 60)

# ---- 场景 ----
instruction = "风电出力骤降50%，光伏下降到20MW，负荷280MW，请紧急调整发电计划，优先保证供电可靠性"
total_load, wind, solar = 280.0, 20.0, 20.0

print(f"\n📡 场景: 负荷={total_load}MW, 风电={wind}MW, 光伏={solar}MW")
print(f"📝 指令: {instruction}\n")

# =====================================================
# AGENT 1: MemoryAgent — 检索历史经验
# =====================================================
print("── Agent 1: MemoryAgent ──")
memory = MemoryAgent()

# 先注入一条历史成功经验
memory.store(
    {"total_load": 270, "wind_forecast": 22, "solar_forecast": 25, "instruction": "风电出力骤降"},
    dispatch_plan={
        "summary": "启动燃气机组G5全容量，增加G1出力至80MW，削减非关键负荷",
        "unit_commitment": {"G1": 80, "G2": 70, "G3": 40, "G4": 40, "G5": 30, "G6": 20},
        "expected_cost": 13500,
    },
    safety_score=88.0,
    is_success=True,
)
print("  注入历史经验: 1条 (风电骤降场景, 评分88, 成功)")

# 检索
mem_ctx = memory.retrieve_context(total_load=total_load, wind_forecast=wind, solar_forecast=solar)
has_memory = "历史相似场景" in mem_ctx
print(f"  检索结果: {'✅ ' + str(mem_ctx.count('参考场景')) + ' 条相似场景' if has_memory else '暂无'}")
if has_memory:
    for line in mem_ctx.split("\n")[:8]:
        if line.strip():
            print(f"    {line.strip()}")

grid_context = {
    "total_load": total_load,
    "wind_forecast": wind,
    "solar_forecast": solar,
    "generator_status": {f"G{i}": "on" for i in range(1, 7)},
    "topology_status": {},
    "memory_context": mem_ctx,  # 🔗 注入历史经验
}

# =====================================================
# AGENT 2 & 3: Planner ⇄ Validator 辩论
# =====================================================
print("\n── Agent 2-3: Planner ⇄ Validator 辩论 ──")
planner = PlannerAgent()
validator = ValidatorAgent()

for round_num in range(1, 4):
    if round_num > 1:
        # 构建验证反馈
        issues = []
        details = validation.get("details", {})
        for dim in ["voltage", "line_loading", "n1_security"]:
            d = details.get(dim, {})
            if d.get("violations"):
                issues.append(f"{dim}: {len(d['violations'])}处越限")
        feedback = "验证不通过。存在以下问题: " + "; ".join(issues) if issues else "所有维度通过。"
        instruction_with_fb = f"{instruction}\n\n[上轮验证反馈]\n{feedback}\n请根据反馈修正方案。"
    else:
        instruction_with_fb = instruction

    print(f"\n  --- 辩论第{round_num}轮 ---")
    plan = planner.plan(instruction_with_fb, grid_context)
    mode = plan.get("metadata", {}).get("mode", "unknown")
    print(f"  📋 LLM推理 (mode={mode}): {plan.get('summary', '')[:100]}")

    # 打印机组出力
    uc = plan.get("unit_commitment", {})
    if uc:
        outputs = {g: (uc[g].get("output_mw", uc[g]) if isinstance(uc[g], dict) else uc[g]) for g in uc}
        print(f"    机组出力(MW): {json.dumps(outputs, ensure_ascii=False, default=str)}")

    validation = validator.validate(plan, grid_context)
    score = validation["safety_score"]
    status = "✅ 通过" if validation["passed"] else "❌ 不通过"
    print(f"  🛡️ 验证: {score}/100 {status}")

    if validation["passed"]:
        print(f"  ✅ 方案安全通过! 辩论结束 (共{round_num}轮)")
        break
    elif round_num == 3:
        print(f"  ⚠️ 3轮辩论未通过，选最优方案 (评分{score})")

# =====================================================
# AGENT 4: NegotiatorAgent — 基于 Planner 方案博弈
# =====================================================
print("\n── Agent 4: NegotiatorAgent ──")
negotiator = NegotiatorAgent(num_vpps=4)
neg = negotiator.negotiate(
    multi_vpp_state={
        "global_load": total_load,
        "market_price": 50.0,
        "renewable_forecast": wind + solar,
        "vpp_states": {},
    },
    planner_plan=plan,  # 🔗 传入 Planner 方案
)
print(f"  {'✅' if neg['equilibrium_reached'] else '⚠️'} Nash均衡: {neg['equilibrium_reached']}")
print(f"  💰 总收益: {neg['total_profit']:.1f}元")
if neg.get("vpp_planned"):
    print(f"  📊 Planner→VPP映射: {json.dumps(neg.get('vpp_planned', {}), ensure_ascii=False)}")

# =====================================================
# AGENT 1 (回归): MemoryAgent — 存储本次经验
# =====================================================
print("\n── Agent 1 回归: MemoryAgent 存储 ──")
memory.store(
    {"total_load": total_load, "wind_forecast": wind,
     "solar_forecast": solar, "instruction": instruction},
    dispatch_plan=plan,
    safety_score=validation.get("safety_score", 0),
    is_success=validation.get("passed", False),
)
# 再次检索，确认新经验已入库
mem_ctx2 = memory.retrieve_context(total_load=total_load, wind_forecast=wind, solar_forecast=solar)
count2 = mem_ctx2.count("参考场景") if "历史相似场景" in mem_ctx2 else 0
print(f"  ✅ 经验已存储 (记忆库现有 {count2} 条相似场景)")

# =====================================================
# 汇总
# =====================================================
print("\n" + "=" * 60)
print("  📊 协同链路测试完成")
print(f"  数据流: 记忆Agent→规划Agent⇄验证Agent→博弈Agent→记忆Agent")
print(f"  LLM决策: {'DeepSeek V3' if mode == 'llm' else '规则引擎'}")
print(f"  安全评分: {validation.get('safety_score', 0)}/100")
print(f"  博弈收益: {neg['total_profit']:.1f}元")
print("=" * 60)
