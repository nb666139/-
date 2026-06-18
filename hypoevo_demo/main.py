"""
HypoEvo Demo — 完整演示入口

运行方式：
  # Demo 模式（无需 API Key，使用预设模拟数据）
  python main.py

  # 真实 API 模式：
  # 1. 修改 config.py 中 DEMO_MODE = False
  # 2. 填入你的 API Key
  # 3. python main.py
"""
import json
from config import DEMO_MODE
from llm_client import LLMClient
from memory import EvolutionaryMemory
from agents import GeneratorAgent, DebaterAgent, EvolutionAgent, MemoryAgent
from tournament import TournamentOrchestrator


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧬  HypoEvo                                               ║
║   基于多智能体辩论与进化记忆的科学假设自主生成系统            ║
║                                                              ║
║   Core Innovation:                                           ║
║     • Hypothesis Tournament (Co-Scientist, Nature 2026)      ║
║     • Evolutionary Memory   (EvoScientist, arXiv 2026)       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_architecture():
    print("""
  ┌─────────────────────────────────────────────────────┐
  │              HypoEvo 系统架构                        │
  │                                                     │
  │   研究问题 ──→ [记忆Agent] ──→ 检索历史经验          │
  │        │              ↑                ↓             │
  │        ↓              │         ┌──────┴──────┐      │
  │   [生成Agent]         │         │ 进化记忆库   │      │
  │        │              │         │ ✅成功 ❌失败 │      │
  │        ↓              │         └──────┬──────┘      │
  │   [辩论Agent ×3]      │                ↑             │
  │   正方 反方 评审      │                │              │
  │        │              │                │              │
  │        ↓              │                │              │
  │   [进化Agent] ────────┴── 学习 ────────┘              │
  │        │                                              │
  │        ↓                                              │
  │   最终科学假设 + 验证方案                              │
  └─────────────────────────────────────────────────────┘
    """)


def print_result_summary(result: dict):
    """打印结果摘要"""
    print("\n" + "=" * 65)
    print("  📋 最终结果摘要")
    print("=" * 65)

    evolved = result.get("evolved_hypothesis", {})
    review = result.get("review", {})
    mem_stats = result.get("memory_stats", {})

    print(f"\n  🏆 最优假设: {evolved.get('title', 'N/A')}")
    print(f"  📐 理论框架: {evolved.get('framework', 'N/A')}")

    prediction = evolved.get("refined_prediction", evolved.get("prediction", ""))
    print(f"\n  🔮 理论预言: {prediction[:150]}...")

    print(f"\n  📊 评审得分:")
    for hyp_id, scores in review.get("scores", {}).items():
        print(f"     {hyp_id}: novelty={scores.get('novelty','?')}  "
              f"feasibility={scores.get('feasibility','?')}  "
              f"impact={scores.get('impact','?')}  "
              f"total={scores.get('total','?')}")

    print(f"\n  🧬 进化日志 ({len(evolved.get('evolution_log', []))}项改进):")
    for i, log_entry in enumerate(evolved.get("evolution_log", []), 1):
        print(f"     {i}. {log_entry}")

    print(f"\n  💾 记忆库状态:")
    print(f"     总记录: {mem_stats.get('total', 0)}  "
          f"成功: {mem_stats.get('successes', 0)}  "
          f"失败: {mem_stats.get('failures', 0)}  "
          f"死胡同已避开: {mem_stats.get('dead_ends_avoided', 0)}")


def run_second_task(orchestrator: TournamentOrchestrator):
    """
    模拟第二次任务 —— 展示进化记忆的跨任务迁移能力
    第一次运行已积累了经验，第二次应更快、更准
    """
    print("\n" + "=" * 65)
    print("  🔄 第二次任务：展示进化记忆效果")
    print("  （第一次运行已积累经验，第二次应避免已知死胡同）")
    print("=" * 65)

    second_question = (
        "设计一种新型二维超导材料，要求：(1)空气稳定性好；"
        "(2)理论Tc > 50K；(3)合成路径合理"
    )

    result2 = orchestrator.run(second_question, verbose=True)

    # 对比两次任务
    print("\n" + "=" * 65)
    print("  📈 进化记忆效果对比")
    print("=" * 65)
    mem_stats = result2.get("memory_stats", {})
    print(f"  第一次任务积累的经验在第二次中被复用")
    print(f"  死胡同避开数: {mem_stats.get('dead_ends_avoided', 0)}")
    print(f"  记忆库总条目: {mem_stats.get('total', 0)}")
    print(f"  结论：Agent 越用越聪明 ✅")

    return result2


def main():
    print_banner()
    print_architecture()

    mode_str = "🎮 Demo 模式（模拟数据）" if DEMO_MODE else "🌐 真实 API 模式"
    print(f"  运行模式: {mode_str}")
    print()

    # ---- 初始化 ----
    llm = LLMClient()
    memory_store = EvolutionaryMemory()
    generator = GeneratorAgent(llm)
    debater = DebaterAgent(llm)
    evolver = EvolutionAgent(llm)
    memory_agent = MemoryAgent(llm, memory_store)

    orchestrator = TournamentOrchestrator(
        generator, debater, evolver, memory_agent, memory_store
    )

    # ---- 第一次任务 ----
    research_question = (
        "寻找一种可能在常压或近常压下实现室温超导的新材料体系。"
        "要求：(1)基于可信的物理机制（如电声耦合、等离激元、磁涨落等）；"
        "(2)理论上Tc > 200K；(3)合成条件与实验表征方案可行。"
    )

    result1 = orchestrator.run(research_question, verbose=True)
    print_result_summary(result1)

    # ---- 第二次任务（展示进化记忆效果） ----
    run_second_task(orchestrator)

    # ---- 保存结果 ----
    with open("hypoevo_result.json", "w", encoding="utf-8") as f:
        # 只保存最终进化后的假设和关键元数据
        save_data = {
            "evolved_hypothesis": result1.get("evolved_hypothesis", {}),
            "review": result1.get("review", {}),
            "memory_stats": result1.get("memory_stats", {}),
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 结果已保存到 hypoevo_result.json")

    print("\n" + "=" * 65)
    print("  ✨ HypoEvo Demo 演示完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()
