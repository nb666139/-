"""API 端到端测试"""
import urllib.request, json, sys

API = "http://localhost:8888/api/dispatch"

# Test 1: Reject
print("=== Test 1: 拒绝无关输入 ===")
data = json.dumps({"instruction": "帮我写一首诗", "total_load": 250, "wind_forecast": 60, "solar_forecast": 30}).encode()
req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print(f"  status={resp['status']}, message={resp.get('message','')}")
assert resp["status"] == "rejected", "Should be rejected"

# Test 2: Accept & full pipeline
print("\n=== Test 2: 正常调度 ===")
data2 = json.dumps({"instruction": "光伏出力下降40%，调整发电计划", "total_load": 250, "wind_forecast": 80, "solar_forecast": 18}).encode()
req2 = urllib.request.Request(API, data=data2, headers={"Content-Type": "application/json"})
resp2 = json.loads(urllib.request.urlopen(req2).read())
print(f"  status={resp2['status']}")
assert resp2["status"] == "success", "Should succeed"

tc = resp2.get("time_complexity", {})
print(f"  Time: total={tc.get('total_ms')}ms (LLM={tc.get('llm_ms')}ms PP={tc.get('pandapower_ms')}ms)")

sc = resp2.get("space_complexity", {})
print(f"  Space: {sc.get('summary', 'N/A')}")

ac = resp2.get("accuracy", {})
print(f"  Accuracy: {ac.get('constraint_score')}/100, Gap={ac.get('cost_gap_pct')}%")

cd = resp2.get("cost_detail", {})
print(f"  Cost: total=Y{cd.get('total',0):.0f}")
print(f"    gen=Y{cd.get('generation',0):.0f} startup=Y{cd.get('startup',0):.0f} ramp=Y{cd.get('ramp',0):.0f} loss=Y{cd.get('network_loss',0):.0f} penalty=Y{cd.get('curtailment_penalty',0):.0f}")
print(f"    formula={cd.get('formula','N/A')}")

reasoning = resp2.get("reasoning", "N/A")
print(f"  Reasoning: {reasoning[:120]}")
print(f"  Agent steps: {len(resp2.get('agent_log', []))}")

print("\nAll API tests passed!")
