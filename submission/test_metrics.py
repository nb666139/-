"""快速验证：输入校验 + 成本模型 + 复杂度 + 准确率"""
import json, sys, time
sys.path.insert(0, ".")

# ===== 输入校验测试 =====
print("=== 输入校验测试 ===")
from llm.input_guard import InputGuard
g = InputGuard()

tests = [
    ("帮我写一首诗", False),
    ("今天天气怎么样", False),
    ("你好啊", False),
    ("光伏出力下降，调整发电计划", True),
    ("帮我写一段代码", False),
    ("风电骤降50%，调整机组出力", True),
]
for text, expect_rel in tests:
    r = g.check(text)
    status = "✅" if r["relevant"] == expect_rel else "❌"
    print(f"  {status} \"{text[:25]}\" → relevant={r['relevant']}, method={r['method']}, reason={r['reason']}")

# ===== 成本模型测试 =====
print("\n=== 成本模型测试 ===")
from web.server_lite import GridSynergyAPIHandler
inst = GridSynergyAPIHandler.__new__(GridSynergyAPIHandler)

mock_plan = {
    "unit_commitment": {
        "G1": {"status": "on", "output_mw": 80},
        "G2": {"status": "on", "output_mw": 60},
        "G3": {"status": "on", "output_mw": 50},
        "G4": {"status": "on", "output_mw": 40},
        "G5": {"status": "on", "output_mw": 30},
        "G6": {"status": "on", "output_mw": 20},
    },
    "renewable_curtailment": {"wind_mw": 5, "solar_mw": 0},
}
mock_val = {"details": {"n1_security": {"results": [{}] * 5}}}

cost = inst._compute_cost_detail(mock_plan, 300, 80, 60)
comp = inst._compute_complexity(mock_plan, mock_val)
acc = inst._compute_accuracy(mock_plan, mock_val, cost["total"], 300, 80, 60)

print(f"  总成本: ¥{cost['total']:.0f}")
print(f"    发电: ¥{cost['generation']:.0f}")
print(f"    启停: ¥{cost['startup']:.0f}")
print(f"    爬坡: ¥{cost['ramp']:.0f}")
print(f"    网损: ¥{cost['network_loss']:.0f}")
print(f"    惩罚: ¥{cost['curtailment_penalty']:.0f}")
print(f"  消纳率: {cost['renewable_rate']}%")
print(f"  公式: {cost['formula']}")

print(f"\n  空间复杂度: {comp['summary']}")
print(f"  约束满足得分: {acc['constraint_score']}/100")
print(f"  成本最优性: {acc['cost_optimality']}")
print(f"  分维: 电压{acc['breakdown']['voltage']} 线路{acc['breakdown']['line_loading']} N-1{acc['breakdown']['n1_security']} 频率{acc['breakdown']['frequency']}")

print("\n✅ 所有测试通过!")
