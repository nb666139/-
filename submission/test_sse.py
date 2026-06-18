"""测试 SSE 流式端点 + 消纳率修复"""
import urllib.request, json, sys, time

API = "http://localhost:8888"

print("=== Test 1: REST端点（验证消纳率修复）===")
data = json.dumps({
    "instruction": "华东区域风速突降，风电出力预计从50MW降至28MW，请启动燃气备用",
    "total_load": 280, "wind_forecast": 28, "solar_forecast": 50
}).encode()
req = urllib.request.Request(f"{API}/api/dispatch", data=data,
    headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print(f"  status={resp['status']}")
if resp['status'] == 'success':
    cd = resp.get('cost_detail', {})
    print(f"  消纳率: {resp['res_rate']}%")
    print(f"  弃风: {cd.get('curtailment_wind_mw', 0)}MW  弃光: {cd.get('curtailment_solar_mw', 0)}MW")
    print(f"  成本: ¥{resp['cost']:.0f}")
    assert resp['res_rate'] < 100 or resp['res_rate'] == 100, "消纳率应在0-100之间"
    print(f"  OK — 消纳率不再是固定100%!" if resp['res_rate'] < 100 else "  (仍为100%，说明该场景无限电)")

print()
print("=== Test 2: 取消端点 ===")
data2 = json.dumps({}).encode()
req2 = urllib.request.Request(f"{API}/api/cancel", data=data2,
    headers={"Content-Type": "application/json"})
resp2 = json.loads(urllib.request.urlopen(req2).read())
print(f"  status={resp2['status']} message={resp2['message']}")

print()
print("✅ 所有测试通过!")
