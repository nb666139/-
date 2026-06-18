"""Test real formula computation with different inputs."""
import json, urllib.request, time

def test_dispatch(label, instruction, total_load, wind, solar):
    data = json.dumps({
        "instruction": instruction,
        "total_load": total_load,
        "wind_forecast": wind,
        "solar_forecast": solar,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8888/api/dispatch",
        data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    elapsed = time.time() - t0
    
    logs = result.get("agent_log", [])
    method_info = "?"
    for l in logs:
        if "SCUC" in l.get("message", "") or "MILP" in l.get("message", ""):
            method_info = f"MILP({l['message'][:60]})"
            break
    
    print(f"\n[{label}] 耗时 {elapsed:.1f}s 方法={method_info}")
    print(f"  cost={result['cost']:.2f}万, res={result['res_rate']}%, n1={result['n1_pass_rate']}%")
    for l in logs:
        prefix = "  ✅" if l["is_result"] else "    "
        print(f"{prefix} [{l['time']}] {l['agent']}: {l['message'][:80]}")
    return result

print("=== 真实求解器对比测试 ===\n")

# Test 1: 日前调度
test_dispatch("日前调度", "日前调度，最小化成本，消纳率不低于95%", 300, 90, 60)

# Test 2: 光伏下降
test_dispatch("光伏骤降", "光伏出力下降40%，请调整计划", 250, 80, 18)

# Test 3: 风电下降
test_dispatch("风电骤降", "风电骤降，出力从50MW降至28MW", 280, 28, 50)

print("\n=== 完成 ===")
