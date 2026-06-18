"""
HypoEvo 的五大核心 Agent：
  1. GeneratorAgent   — 假设生成（基于文献和领域知识生成候选假设）
  2. DebaterAgent     — 辩论双方（正/反/评审三种角色扮演）
  3. EvolutionAgent   — 假设进化（综合辩论结果，迭代优化）
  4. MemoryAgent      — 记忆管理（检索历史经验，反馈给其他Agent）
"""
import json
from llm_client import LLMClient
from config import TOURNAMENT_CONFIG


# ============================================================
# 提示词模板
# ============================================================

DOMAIN_PROMPT = """你是一位材料科学领域的研究科学家，专精于超导材料理论设计与第一性原理计算。
你的研究方向：新型超导材料的理论预言——包括氢化物超导体、二维超导体、镍基/铜基超导体等。
你熟悉DFT、Eliashberg方程、BCS理论、以及相关的实验表征手段（输运、ARPES、STM等）。
你需要基于物理直觉、文献知识、以及计算可行性来提出和评估科学假设。"""


GENERATOR_SYSTEM = DOMAIN_PROMPT + """
你是"生成Agent"——你的任务是根据研究问题，生成{count}个候选科学假设。
对每个假设，必须包含：
  - title: 简洁有力的标题
  - framework: 理论基础框架
  - prediction: 具体的理论预言（包含物理量或化学指标）
  - testability: 可验证性分析（理论计算方案 & 实验验证方案）
  - references: 2-3篇支撑文献
输出严格的JSON格式，不输出任何其他内容。"""


PROPONENT_SYSTEM = DOMAIN_PROMPT + """
你是辩论"正方"Agent。你的任务是为一个候选假设进行有力辩护。
维护你被分配到的假设，从理论依据、实验可行性、前人工作支撑三方面论证其价值。
输出JSON，包含: role, favored_hypothesis, arguments (列表), confidence (0-1)。"""


OPPONENT_SYSTEM = DOMAIN_PROMPT + """
你是辩论"反方"Agent。你的任务是批判性地审视假设，寻找其弱点。
你需要是建设性的批评者——指出问题，同时提出改进方向。
输出JSON，包含: role, criticisms (列表), alternative_suggestion (字符串)。"""


REVIEWER_SYSTEM = DOMAIN_PROMPT + """
你是辩论"评审"Agent。你的任务是对正反方辩论后的所有假设进行客观评分。
评分维度: novelty(创新性,1-10), feasibility(可行性,1-10), impact(影响力,1-10)。
给出排名和综合评判，输出JSON格式。"""


EVOLUTION_SYSTEM = DOMAIN_PROMPT + """
你是"进化Agent"。你的任务是综合辩论结果和历史记忆，对最优假设进行改进。
你需要：
  1. 采纳辩论中的有效批评，修正假设中的弱点
  2. 保留原假设的核心创新点
  3. 补充具体的技术方案来解决被指出的问题
  4. 总结此次进化中发现的成功模式和失败模式（供记忆Agent存储）
输出JSON格式。"""


MEMORY_SYSTEM = """你是"记忆Agent"。你的任务是：
  1. 查询历史记忆库中与当前任务相关的经验
  2. 识别当前任务可能踩到的"死胡同"（历史上验证失败的方向）
  3. 推荐历史上验证成功的方法论供复用
输出JSON格式。"""


# ============================================================
# Agent 实现
# ============================================================

class GeneratorAgent:
    """假设生成Agent — 基于研究问题生成候选假设"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, research_question: str,
                 memory_context: str = "",
                 num_hypotheses: int = None) -> dict:
        """生成候选假设"""
        if num_hypotheses is None:
            num_hypotheses = TOURNAMENT_CONFIG["hypothesis_count"]

        system = GENERATOR_SYSTEM.format(count=num_hypotheses)
        user = self._build_prompt(research_question, memory_context)
        response = self.llm.chat(system, user)
        return self._parse(response)

    def _build_prompt(self, question: str, memory: str) -> str:
        parts = [f"## 研究问题\n{question}\n"]
        if memory:
            parts.append(f"\n{memory}")
        parts.append("\n现在请生成假设（严格JSON格式，无其他文字）：")
        return "\n".join(parts)

    @staticmethod
    def _parse(response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return {"hypotheses": [], "error": "parse_failed"}


class DebaterAgent:
    """辩论Agent — 支持正方/反方/评审三种角色"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, hypothesis: dict, research_question: str) -> dict:
        """正方论证"""
        user = f"""## 研究问题\n{research_question}\n
## 当前假设\n{json.dumps(hypothesis, ensure_ascii=False)}\n
请以正方立场进行辩护论证："""
        return self._call(PROPONENT_SYSTEM, user)

    def oppose(self, hypothesis: dict, proponent_args: list,
               research_question: str) -> dict:
        """反方批判"""
        user = f"""## 研究问题\n{research_question}\n
## 当前假设\n{json.dumps(hypothesis, ensure_ascii=False)}\n
## 正方论证\n{json.dumps(proponent_args, ensure_ascii=False)}\n
请以反方立场进行批判性审阅："""
        return self._call(OPPONENT_SYSTEM, user)

    def review(self, hypotheses: list, debates: list,
               research_question: str) -> dict:
        """综合评审"""
        summary = {
            "research_question": research_question,
            "hypotheses_reviewed": [h["id"] for h in hypotheses],
            "debate_records": debates,
        }
        user = f"""## 综合辩论记录\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n
请对所有假设进行综合评分和排名："""
        return self._call(REVIEWER_SYSTEM, user)

    def _call(self, system: str, user: str) -> dict:
        response = self.llm.chat(system, user)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return {"error": "parse_failed", "raw": response[:200]}


class EvolutionAgent:
    """进化Agent — 综合辩论结果，迭代优化假设"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evolve(self, best_hypothesis: dict, review_result: dict,
               debate_records: list, memory_context: str,
               research_question: str) -> dict:
        """进化最优假设"""
        context = {
            "research_question": research_question,
            "original_hypothesis": best_hypothesis,
            "review_verdict": review_result.get("verdict", ""),
            "review_scores": review_result.get("scores", {}),
            "debate_criticisms": [
                r.get("criticisms", []) for r in debate_records
                if r.get("role") == "opponent"
            ],
        }
        user = f"""## 进化任务\n{json.dumps(context, ensure_ascii=False, indent=2)}\n
{memory_context}\n
请综合以上所有信息，对假设进行进化改进："""
        response = self.llm.chat(EVOLUTION_SYSTEM, user)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return {"error": "parse_failed"}


class MemoryAgent:
    """记忆Agent — 管理经验检索与存储建议"""

    def __init__(self, llm: LLMClient, memory):
        self.llm = llm
        self.memory = memory

    def retrieve_and_analyze(self, research_question: str,
                             current_hypotheses: list) -> dict:
        """检索相关记忆并分析"""
        # 从记忆库检索
        relevant = self.memory.query(research_question, top_k=5)
        failures = self.memory.get_failures()
        successes = self.memory.get_successes()

        context = {
            "research_question": research_question,
            "current_hypotheses": [h.get("title", "") for h in current_hypotheses],
            "relevant_memories": relevant,
            "known_dead_ends": failures,
            "proven_strategies": successes,
        }

        user = f"""## 分析任务\n{json.dumps(context, ensure_ascii=False, indent=2)}\n
请基于历史记忆分析当前假设的风险和机遇："""
        response = self.llm.chat(MEMORY_SYSTEM, user)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return {"error": "parse_failed"}

    def learn_from_evolution(self, evolution_result: dict,
                             task_description: str):
        """从进化结果中学习——存入记忆库"""
        memory_contrib = evolution_result.get("memory_contribution", {})

        for pattern in memory_contrib.get("success_patterns", []):
            self.memory.add_success(pattern, task_description,
                                    tags=["evolution", "success"])

        for pattern in memory_contrib.get("failure_patterns", []):
            self.memory.add_failure(pattern, task_description,
                                    tags=["evolution", "failure"])
