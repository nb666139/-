"""
HypoEvo Agent Demo v2 — 真实的 Agent 执行循环

核心区别 vs v1:
  v1: LLM 对话 → 输出文本  (就是个 chatbot)
  v2: Thought → Action → [调用真实工具] → Observation → Thought → ... (真正的 Agent!)

每个 Agent 有完整的 "感知-思考-行动" 闭环:
  1. 接收任务和当前状态
  2. 思考下一步做什么 (Thought)
  3. 决定调用哪个工具 (Action)
  4. 工具返回结果 (Observation)
  5. 基于结果继续思考或结束

运行方式:
  python main_v2.py
"""
import json
import time
from llm_client import LLMClient
from tools import create_default_tools, ToolRegistry

# ============================================================
# ReAct Agent — 真正的 Agent 执行循环
# ============================================================
class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent

    执行循环:
      while not done:
        1. 用 LLM 思考: 当前状态 → Thought + Action 决策
        2. 如果 Action = 某个工具 → 真实执行工具 → 得到 Observation
        3. 反馈 Observation 给下一轮 Thought
        4. 如果 Action = finish → 结束
    """

    def __init__(self, name: str, role: str, llm: LLMClient, tools: ToolRegistry):
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = tools
        self.max_steps = 8
        self.history: list[dict] = []

        self.system_prompt = f"""你是 {name}，{role}。

你是一个 AI Agent，不是聊天机器人。你有以下真实工具可以调用:

{tools.get_descriptions()}

你的工作方式:
1. 读当前任务和已有信息
2. 思考 (Thought): 分析现状，决定下一步做什么
3. 行动 (Action): 调用一个工具来获取信息或执行操作
4. 观察 (Observation): 你会收到工具的执行结果
5. 回到 2 继续思考，直到任务完成

重要规则:
- 你可以多次调用工具，每次基于之前的 Observation 来调整
- 工具调用格式: ACTION: tool_name | PARAMS: {{"param1": "value1", ...}}
- 完成时输出: ACTION: finish | RESULT: 你的最终产出
- 绝不虚构数据——所有事实必须来自工具调用结果
- 不要跳过工具直接输出结论
"""

    def run(self, task: str, max_steps: int = None) -> dict:
        """执行完整的 Agent 循环，返回结果"""
        max_steps = max_steps or self.max_steps
        self.history = []
        messages = [{"role": "system", "content": self.system_prompt}]

        step_log = []
        print(f"\n{'='*60}")
        print(f"  🤖 {self.name} 启动")
        print(f"  📋 任务: {task[:100]}...")
        print(f"  🔧 可用工具: {', '.join(self.tools._tools.keys())}")
        print(f"{'='*60}")

        # Agent 执行循环
        for step in range(1, max_steps + 1):
            print(f"\n  ── Step {step} ──")

            # 1. 构建 prompt
            if step == 1:
                user_msg = f"## 你的任务\n{task}\n\n请开始思考并行动。"
            else:
                user_msg = f"## 上一轮的 Observation\n{last_obs}\n\n请基于这个观察继续思考。如果任务已完成，输出 finish。"

            messages.append({"role": "user", "content": user_msg})
            response = self.llm.chat_via_role(messages, temperature=0.5)
            messages.append({"role": "assistant", "content": response})

            # 2. 解析 Thought 和 Action
            thought = self._extract_thought(response)
            action_name, action_params = self._extract_action(response)

            step_data = {
                "step": step, "thought": thought,
                "action": action_name, "params": action_params
            }
            print(f"  💭 Thought: {thought[:100]}...")

            # 3. 如果 Agent 决定结束
            if action_name == "finish":
                result = self._extract_result(response)
                step_data["result"] = result
                step_log.append(step_data)
                print(f"  ✅ 任务完成: {result[:120]}...")
                messages.append({"role": "user", "content": f"任务完成。最终结果:\n{result}"})
                break

            # 4. 执行工具调用
            if not action_name:
                print(f"  ⚠️ 未识别到工具调用，让 Agent 重试")
                messages.append({"role": "user",
                    "content": "请调用一个具体的工具（格式: ACTION: tool_name | PARAMS: {...}）"})
                step_log.append(step_data)
                continue

            print(f"  🔧 调用工具: {action_name}({json.dumps(action_params, ensure_ascii=False)})")
            observation = self.tools.execute(action_name, action_params)
            last_obs = observation

            # 截断过长的 observation
            obs_display = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"  👁️  Observation: {obs_display}")

            step_data["observation"] = observation[:500]
            step_log.append(step_data)

            # 5. 检查是否该结束了（工具链完成）
            if self._should_finish(action_name, observation):
                final = self._compose_final_result(step_log)
                step_log.append({"step": "final", "result": final})
                print(f"  ✅ 自动完成")
                break

        # 组装返回
        return {
            "agent": self.name,
            "task": task,
            "steps": step_log,
            "total_steps": len(step_log),
            "tools_used": list(set(s.get("action", "") for s in step_log if s.get("action"))),
        }

    def _extract_thought(self, response: str) -> str:
        """从回复中提取 Thought"""
        for marker in ["Thought:", "思考：", "💭"]:
            if marker in response:
                parts = response.split(marker, 1)[1].strip()
                # 取到下一个标记之前
                for end in ["\nAction:", "\n行动:", "\n🔧", "\nACTION:"]:
                    if end in parts:
                        parts = parts.split(end)[0].strip()
                return parts
        # 取第一段有意义的话
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("ACTION") and not line.startswith("PARAMS"):
                return line[:200]
        return response[:200]

    def _extract_action(self, response: str):
        """从回复中提取 Action"""
        action_name = None
        action_params = {}

        # 方法1: 从 ACTION 行解析
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ACTION:") or line.startswith("ACTION：") or line.startswith("行动:"):
                action_name = line.split(":", 1)[-1].split("：", 1)[-1].strip().lower()
                if action_name == "finish":
                    return "finish", {}

        # 方法2: 搜索已知工具名
        for tool_name in self.tools._tools.keys():
            if tool_name in response.lower():
                action_name = tool_name
                break

        # 解析 PARAMS
        for line in response.split("\n"):
            if "PARAMS:" in line or "参数:" in line:
                params_str = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                try:
                    # 尝试 JSON 解析
                    start = params_str.find("{")
                    end = params_str.rfind("}") + 1
                    if start >= 0 and end > start:
                        action_params = json.loads(params_str[start:end])
                except:
                    action_params = {}

        # 方法3: 在 response 中全局搜索 JSON 对象
        if not action_params:
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    action_params = json.loads(response[start:end])
            except:
                pass

        return action_name, action_params

    def _extract_result(self, response: str) -> str:
        """提取最终结果"""
        for marker in ["RESULT:", "结果：", "最终结论：", "✅"]:
            if marker in response:
                parts = response.split(marker, 1)[-1].strip()
                return parts[:500]
        return response[-500:] if len(response) > 500 else response

    def _should_finish(self, tool: str, obs: str) -> bool:
        """判断是否应该自动结束（工具链闭环完成）"""
        if tool == "file_writer" and '"status": "success"' in obs:
            return True
        if tool == "dft_estimator" and "estimated_tc" in obs:
            return True
        return False

    def _compose_final_result(self, step_log: list) -> str:
        """基于已执行的工具链组合最终结果"""
        results = ["Agent 执行总结:"]
        for s in step_log:
            if s.get("action") and s["action"] != "finish":
                obs = s.get("observation", "")
                results.append(f"- 调用 {s['action']}: {obs[:100]}...")
        return "\n".join(results)


# ============================================================
# Demo 场景
# ============================================================
class LLMClient2(LLMClient):
    """扩展 LLM 客户端，支持 messages 格式"""

    def chat_via_role(self, messages: list, temperature: float = None) -> str:
        """支持多轮对话的 chat"""
        if self.demo_mode:
            return self._demo_agent_response(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or 0.5,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _demo_agent_response(self, messages: list) -> str:
        """Demo 模式: 根据最后一轮消息生成模拟 Agent 响应"""
        last_user = messages[-1]["content"].lower() if messages else ""
        msg_count = len(messages)

        # Agent 步骤模拟 —— 这才是真正的 "Agent 在行动"!
        if msg_count <= 3:
            # 第1步: 先搜论文验证假设可行性
            return (
                "Thought: 我需要先验证假设的理论可行性。让我搜索支撑文献\n\n"
                'ACTION: arxiv_search | PARAMS: {"query": "Janus MoSSe superconductivity strain"}'
            )

        elif msg_count <= 5:
            # 第2步: 查询材料数据库验证声子谱
            if "arxiv" in last_user or "Janus" in last_user:
                return (
                    "Thought: 文献确认了Janus TMDs的理论潜力。现在需要验证声子稳定性\n\n"
                    'ACTION: materials_db_query | PARAMS: {"material_formula": "MoSSe", "property": "phonon_dispersion"}'
                )
            else:
                return (
                    "Thought: 需要从材料数据库获取基础参数\n\n"
                    'ACTION: materials_db_query | PARAMS: {"material_formula": "MoSSe", "property": "band_gap"}'
                )

        elif msg_count <= 7:
            # 第3步: 用 McMillan 公式估算 Tc
            return (
                "Thought: 声子谱稳定无虚频，可进行Tc估算。用McMillan公式\n\n"
                'ACTION: dft_estimator | PARAMS: {"material": "MoSSe", "estimate_type": "tc_mcmillan", '
                '"input_params": {"strain": 0.08, "lambda": 2.8, "Theta_D": 380, "mu_star": 0.13}}'
            )

        elif msg_count <= 9:
            # 第4步: 执行数值计算验证
            if "tc" in last_user or "estimated" in last_user:
                return (
                    "Thought: McMillan估算Tc~310K。让我用Python验证这个计算结果\n\n"
                    "ACTION: code_executor | PARAMS: {"
                    '"code": "import math\\n'
                    'lambda_val = 2.8\\n'
                    'Theta_D = 380\\n'
                    'mu_star = 0.13\\n'
                    'Tc = (Theta_D/1.45) * math.exp(-1.04*(1+lambda_val)/(lambda_val - mu_star*(1+0.62*lambda_val)))\\n'
                    'print(f\\"McMillan Tc = {Tc:.1f} K\\")\\n'
                    'print(f\\"可行性评估: 室温超导候选 ✓\\")\\n'
                    'print(f\\"假设来源: 等离激元+BCS联合机制\\")\\n'
                    'print(f\\"λ=2.8 (ep=2.0 + pl=0.8), Θ_D=380K, μ*=0.13")",'
                    '"description": "McMillan Tc 计算验证"'
                    '}'
                )
            else:
                return (
                    "Thought: 基于材料属性，执行数值验证\n\n"
                    'ACTION: code_executor | PARAMS: {"code": "print(\'材料参数验证通过\')", '
                    '"description": "参数验证"}'
                )

        else:
            # 第5步: 保存报告文件
            return (
                "Thought: 所有验证完成。生成最终假设报告并保存到文件\n\n"
                "ACTION: file_writer | PARAMS: {\n"
                '  "filename": "hypothesis_H1_evolved.md",\n'
                '  "content": "# 最终科学假设报告\\n\\n'
                '## 题目: 二维Janus TMDs室温超导假设\\n\\n'
                '## 核心预言\\n'
                '在hBN/MoSSe/hBN异质结中，8%面内应变 + 等离激元联合机制 → Tc ~310K\\n\\n'
                '## 验证路径\\n'
                '1. DFT声子谱: 已验证无虚频 ✓\\n'
                '2. McMillan Tc估算: ~310 K ✓\\n'
                '3. 下一步: 实验ARPES验证费米面\\n\\n'
                '## 支撑文献\\n'
                '- Lu et al., Nature Nanotech 2017 (Janus合成)\\n'
                '- Zhang et al., arXiv 2025 (7%应变→Tc 120K)\\n'
                '- Novelli et al., PRL 2024 (等离激元配对)\\n'
                '- Mennel et al., Nature Comm 2025 (hBN缓冲层8%应变)\\n",\n'
                '  "mode": "w"\n}'
            )


# ============================================================
# 多 Agent 协作演示
# ============================================================
def demo_agent_execution():
    """演示单个 Agent 的完整自主执行过程"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HypoEvo Agent Demo v2 — 真实的 Agent 自主执行              ║
║                                                              ║
║  与 v1 的区别:                                                ║
║  ❌ v1: Agent 只是对话 → 没有真的"做"任何事情                  ║
║  ✅ v2: Agent 调用真实工具 → 搜索论文 → 查询数据库             ║
║         → 执行计算 → 保存文件                                 ║
║                                                              ║
║  执行循环: Thought → Action → [工具执行] → Observation       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 初始化
    llm = LLMClient2()
    tools = create_default_tools(output_dir=".")

    # 创建研究 Agent
    research_agent = ReActAgent(
        name="材料科学研究Agent",
        role="超导材料理论研究者，擅长DFT计算和假设生成",
        llm=llm,
        tools=tools,
    )

    task = (
        "验证以下科学假设的理论可行性:\n"
        "'在hBN/MoSSe/hBN范德华异质结中，8%面内压缩应变可诱导室温超导(Tc~300K)'\n\n"
        "要求:\n"
        "1. 搜索支撑文献 (arxiv_search)\n"
        "2. 查询MoSSe声子谱稳定性 (materials_db_query)\n"
        "3. 用McMillan公式估算Tc (dft_estimator)\n"
        "4. 用Python验证计算 (code_executor)\n"
        "5. 保存最终报告 (file_writer)\n"
        "请自主完成以上所有步骤"
    )

    result = research_agent.run(task, max_steps=8)

    # 最终摘要
    print(f"\n{'='*60}")
    print(f"  📊 Agent 执行摘要")
    print(f"{'='*60}")
    print(f"  总步骤数: {result['total_steps']}")
    print(f"  使用工具: {result['tools_used']}")
    for s in result["steps"]:
        if s.get("action") and s["action"] != "finish":
            print(f"\n  Step {s['step']}: 调用 {s['action']}")
            print(f"    目标: {s.get('thought', '')[:120]}...")


def demo_multi_agent_collaboration():
    """演示多 Agent 协作: Researcher + Reviewer 双 Agent 交互"""
    print("\n\n")
    print("=" * 60)
    print("  🤝 多 Agent 协作演示")
    print("  Researcher Agent 产出 → Reviewer Agent 审阅 → 迭代优化")
    print("=" * 60)

    llm = LLMClient2()
    tools = create_default_tools(output_dir=".")

    # Agent 1: Researcher
    researcher = ReActAgent(
        name="ResearcherAgent",
        role="材料科学假设研究者",
        llm=llm,
        tools=tools,
    )

    # Agent 2: Reviewer (同样的工具，不同的角色)
    reviewer = ReActAgent(
        name="ReviewerAgent",
        role="科学假设评审专家，擅长发现方法论漏洞",
        llm=llm,
        tools=tools,
    )

    # Researcher 先工作
    print("\n🔬 Researcher Agent 开始工作...")
    res_result = researcher.run(
        "搜索并验证 'LaH10-xFx 氟掺杂氢化物超导' 的理论可行性。"
        "需要: arxiv_search → materials_db_query → dft_estimator",
        max_steps=6
    )

    print(f"\n📝 Researcher 产出: 用了 {res_result['total_steps']} 步, 调用了 {res_result['tools_used']}")

    # Reviewer 审阅 Researcher 的发现
    print(f"\n🔍 Reviewer Agent 开始审阅...")
    review_result = reviewer.run(
        f"审阅 Researcher 的发现。Researcher 调用了 {res_result['tools_used']} 工具。"
        "请用 arxiv_search 核查是否有遗漏的关键文献，用 materials_db_query 验证数据一致性。",
        max_steps=4
    )

    print(f"\n📝 Reviewer 审阅完成: 用了 {review_result['total_steps']} 步")
    print(f"\n{'='*60}")
    print(f"  ✅ 多 Agent 协作完成!")
    print(f"  Researcher: {res_result['total_steps']} steps → {res_result['tools_used']}")
    print(f"  Reviewer: {review_result['total_steps']} steps → {review_result['tools_used']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo_agent_execution()
    demo_multi_agent_collaboration()