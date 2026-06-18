"""
HypoEvo Web 前端 — Flask + SSE 实时展示 Agent 执行过程

启动: python app.py
访问: http://localhost:5000
"""
import json
import queue
import threading
from flask import Flask, render_template, request, Response, jsonify

from llm_client import LLMClient
from tools import create_default_tools

app = Flask(__name__)

# ============================================================
# 预设任务模板
# ============================================================
TASK_TEMPLATES = [
    {
        "id": "room_temp_sc",
        "title": "室温超导材料发现",
        "question": (
            "寻找一种可能在常压或近常压下实现室温超导的新材料体系。"
            "要求:(1)基于可信的物理机制;(2)理论上Tc>200K;(3)合成方案可行。"
        ),
        "prompt": "验证以下假设:'hBN/MoSSe/hBN异质结中8%面内应变可诱导室温超导(Tc~300K)'",
        "steps": "arxiv_search → materials_db_query → dft_estimator → code_executor → file_writer",
    },
    {
        "id": "2d_mat",
        "title": "新型二维材料设计",
        "description": "设计一种新型二维超导材料",
        "question": "设计一种新型二维超导材料，要求:(1)空气稳定性好;(2)理论Tc>50K;(3)合成路径合理",
        "prompt": "搜索并验证 'LaH10-xFx 氟掺杂氢化物超导' 的理论可行性",
        "steps": "arxiv_search → materials_db_query → dft_estimator → file_writer",
    },
    {
        "id": "custom",
        "title": "自定义任务",
        "question": "",
        "prompt": "",
        "steps": "",
    },
]

TOOLS = {
    "arxiv_search": {"name": "📚 arXiv 论文搜索", "desc": "搜索学术论文数据库"},
    "materials_db_query": {"name": "🗄️  材料数据库查询", "desc": "查询 Materials Project"},
    "dft_estimator": {"name": "🧮 DFT 参数估算", "desc": "McMillan公式估算Tc"},
    "code_executor": {"name": "💻 代码执行器", "desc": "运行Python科学计算"},
    "file_writer": {"name": "📝 文件写入", "desc": "保存报告到磁盘"},
}


# ============================================================
# Agent 执行引擎（支持 SSE 流式输出）
# ============================================================
class StreamingAgent:
    """
    与 ReActAgent 逻辑相同，但通过队列实时推送每步进度
    """

    def __init__(self):
        self.llm = LLMClient()
        self.tools = create_default_tools(output_dir=".")
        self.max_steps = 6

    def run(self, task: str, output_queue: queue.Queue):
        """执行 Agent 循环，每步通过队列推送给前端"""
        msg_count = 0

        def emit(event_type: str, data: dict):
            output_queue.put({"event": event_type, "data": data})

        emit("start", {"task": task, "tools": list(TOOLS.keys())})

        last_obs = ""
        for step in range(1, self.max_steps + 1):
            # 用 LLM 思考下一步
            if step == 1:
                user_msg = f"## 任务\n{task}\n\n请开始思考并行动。"
            else:
                user_msg = f"## 上一轮 Observation\n{last_obs[:800]}\n\n请继续思考。"

            system_prompt = self._build_system_prompt()
            response = self.llm.chat(system_prompt, user_msg)

            thought = self._extract_thought(response)
            tool_name, tool_params = self._extract_action(response)

            emit("thought", {
                "step": step, "content": thought,
                "tool": tool_name, "params": tool_params,
            })

            if tool_name == "finish":
                result = self._extract_result(response)
                emit("finish", {"step": step, "result": result})
                emit("done", {"total_steps": step})
                return

            if not tool_name:
                emit("error", {"step": step, "msg": "未识别工具调用"})
                continue

            # 执行工具
            observation = self.tools.execute(tool_name, tool_params)
            try:
                obs_json = json.loads(observation)
                obs_display = json.dumps(obs_json, ensure_ascii=False, indent=2)[:1500]
            except:
                obs_display = observation[:1500]

            last_obs = obs_display
            emit("observation", {
                "step": step,
                "tool": tool_name,
                "content": obs_display,
            })

            if self._should_done(tool_name, observation):
                emit("done", {"total_steps": step, "last_tool": tool_name})
                return

        emit("done", {"total_steps": self.max_steps, "maxed_out": True})

    def _build_system_prompt(self):
        return f"""你是一个 AI Agent，不是聊天机器人。
你有以下工具可用:
{self.tools.get_descriptions()}

你的工作循环:
1. 思考 (Thought) — 分析当前情况
2. 行动 (Action) — 调用工具
3. 观察 (Observation) — 收到工具结果
4. 回到 1，直到完成任务

格式:
Thought: [你的思考]
ACTION: tool_name
PARAMS: {{"param1": "value1"}}

完成时: ACTION: finish | RESULT: 最终产出"""

    def _extract_thought(self, resp):
        for m in ["Thought:", "思考："]:
            if m in resp:
                return resp.split(m, 1)[1].split("\n")[0].strip()[:200]
        for line in resp.split("\n"):
            if line.strip() and not line.strip().startswith("ACTION"):
                return line.strip()[:200]
        return resp[:200]

    def _extract_action(self, resp):
        name, params = None, {}
        for line in resp.split("\n"):
            if "ACTION:" in line:
                name = line.split(":", 1)[-1].strip().lower()
                if name == "finish":
                    return "finish", {}
        for tn in self.tools._tools:
            if tn in resp.lower():
                name = tn; break
        try:
            s = resp.find("{"); e = resp.rfind("}") + 1
            if s >= 0 and e > s:
                params = json.loads(resp[s:e])
        except: pass
        return name, params

    def _extract_result(self, resp):
        for m in ["RESULT:", "结果："]:
            if m in resp: return resp.split(m, 1)[-1].strip()[:500]
        return resp[-500:]

    def _should_done(self, tool, obs):
        return tool == "file_writer" and '"status": "success"' in obs


# ============================================================
# Flask 路由
# ============================================================
@app.route("/")
def index():
    return render_template("index.html", tasks=TASK_TEMPLATES, tools=TOOLS)


@app.route("/api/task/<task_id>")
def get_task(task_id):
    for t in TASK_TEMPLATES:
        if t["id"] == task_id:
            return jsonify(t)
    return jsonify(TASK_TEMPLATES[0])


@app.route("/api/run", methods=["POST"])
def run_agent():
    """SSE 流式返回 Agent 执行过程"""
    data = request.get_json()
    task = data.get("task", "")
    if not task:
        return jsonify({"error": "no task"}), 400

    def generate():
        q = queue.Queue()
        agent = StreamingAgent()

        def _run():
            try:
                agent.run(task, q)
            except Exception as e:
                q.put({"event": "error", "data": {"msg": str(e)}})
            q.put({"event": "__end__", "data": {}})

        threading.Thread(target=_run, daemon=True).start()

        while True:
            msg = q.get()
            if msg["event"] == "__end__":
                break
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🧬 HypoEvo Web UI")
    print("  访问: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000, threaded=True)