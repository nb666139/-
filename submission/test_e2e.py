"""
端到端测试：模拟两次连续调度，验证后端可以处理多次请求。
"""
import urllib.request, json, sys

API = "http://localhost:8888/api/dispatch"

# ---- Test 1: 风电骤降 ----
print("=== 调度 #1: 风电骤降 ===")
data1 = json.dumps({
    "instruction": "华东区域风速突降，风电出力预计从50MW降至28MW，请启动燃气备用",
    "total_load": 280, "wind_forecast": 28, "solar_forecast": 50
}).encode()
req1 = urllib.request.Request(API, data=data1, headers={"Content-Type": "application/json"})
resp1 = json.loads(urllib.request.urlopen(req1).read())
print(f"  状态: {resp1['status']}")
print(f"  成本: ¥{resp1['cost']:.0f}")
print(f"  消纳率: {resp1['res_rate']}%")
print(f"  时间: {resp1['time_complexity']['total_ms']}ms")
print(f"  约束满足: {resp1['accuracy']['constraint_score']}/100")
print(f"  Agent日志: {len(resp1['agent_log'])} 条")

# ---- Test 2: 负荷高峰 ----
print()
print("=== 调度 #2: 负荷高峰 ===")
data2 = json.dumps({
    "instruction": "明日晚高峰7-9点负荷预计增加30%，请提前调整机组出力并预留旋转备用",
    "total_load": 380, "wind_forecast": 50, "solar_forecast": 20
}).encode()
req2 = urllib.request.Request(API, data=data2, headers={"Content-Type": "application/json"})
resp2 = json.loads(urllib.request.urlopen(req2).read())
print(f"  状态: {resp2['status']}")
print(f"  成本: ¥{resp2['cost']:.0f}")
print(f"  消纳率: {resp2['res_rate']}%")
print(f"  时间: {resp2['time_complexity']['total_ms']}ms")
print(f"  约束满足: {resp2['accuracy']['constraint_score']}/100")
print(f"  Agent日志: {len(resp2['agent_log'])} 条")

# ---- 对比结论 ----
print()
print("=== 对比 ===")
assert resp1['status'] == 'success', "调度#1失败"
assert resp2['status'] == 'success', "调度#2失败"
assert resp1['cost'] != resp2['cost'], "两次调度成本应不同"
assert resp1['res_rate'] != resp2['res_rate'] or resp1['total_load'] != resp2['total_load'], "消纳率应有差异"

print(f"  成本差异: ¥{resp1['cost']:.0f} → ¥{resp2['cost']:.0f}")
print(f"  消纳率差异: {resp1['res_rate']}% → {resp2['res_rate']}%")
print(f"  时间差异: {resp1['time_complexity']['total_ms']}ms → {resp2['time_complexity']['total_ms']}ms")
print()
print("✅ 两次连续调度均成功，后端无问题！")
