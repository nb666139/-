"""
辩论锦标赛编排器 — HypoEvo 的核心调度引擎

流程：
  1. 记忆检索 → 注入历史经验
  2. 假设生成 → 生成候选假设
  3. 多轮辩论 → 对每个假设进行正/反/评审三方辩论
  4. 假设进化 → 综合辩论结果，迭代优化最优假设
  5. 记忆学习 → 将本次经验存入记忆库
"""
from config import TOURNAMENT_CONFIG


class TournamentOrchestrator:
    """辩论锦标赛编排器"""

    def __init__(self, generator, debater, evolver, memory_agent, memory_store):
        self.generator = generator
        self.debater = debater
        self.evolver = evolver
        self.memory_agent = memory_agent
        self.memory_store = memory_store

    def run(self, research_question: str, verbose: bool = True) -> dict:
        """
        执行完整的 HypoEvo 流水线
        返回完整的结果字典，供展示和分析
        """
        log = lambda msg: print(msg) if verbose else None
        log("\n" + "=" * 65)
        log("  🧬 HypoEvo — 基于多智能体辩论与进化记忆的科学假设生成")
        log("=" * 65)
        log(f"\n📋 研究问题：{research_question}")

        # ---- 阶段 1: 记忆检索 ----
        log("\n" + "-" * 45)
        log("🔍 阶段 1/5：进化记忆检索")
        log("-" * 45)
        memory_state = self.memory_store.summary()
        log(f"  {memory_state}")

        # 先从记忆Agent获取分析（可能在API模式下用，Demo用本地检索）
        memory_context = self.memory_store.get_context_for_task(research_question)

        if memory_context.strip() and "暂无" not in memory_context:
            log("  📖 命中历史经验：")
            for entry in self.memory_store.query(research_question, top_k=3):
                emoji = "❌" if entry["weight"] < 0 else "✅"
                log(f"     {emoji} {entry['content'][:60]}...")
        else:
            log("  📭 无相关历史记忆（首次运行）")

        # ---- 阶段 2: 假设生成 ----
        log("\n" + "-" * 45)
        log("🎯 阶段 2/5：生成候选假设")
        log("-" * 45)
        generation = self.generator.generate(
            research_question, memory_context,
            num_hypotheses=TOURNAMENT_CONFIG["hypothesis_count"]
        )
        hypotheses = generation.get("hypotheses", [])
        for h in hypotheses:
            log(f"  💡 {h['id']}: {h['title']}")

        if not hypotheses:
            log("  ❌ 假设生成失败！")
            return {"error": "hypothesis_generation_failed"}

        # ---- 阶段 3: 多轮辩论 ----
        log("\n" + "-" * 45)
        log("⚔️  阶段 3/5：多智能体辩论锦标赛")
        max_rounds = TOURNAMENT_CONFIG["max_debate_rounds"]

        all_debate_records = []
        for round_num in range(1, max_rounds + 1):
            log(f"\n  — 辩论轮次 {round_num}/{max_rounds} —")
            round_records = []

            for hyp in hypotheses:
                # 正方
                proponent = self.debater.propose(hyp, research_question)
                # 反方
                opponent = self.debater.oppose(
                    hyp, proponent.get("arguments", []), research_question
                )
                record = {
                    "hypothesis_id": hyp["id"],
                    "round": round_num,
                    "proponent": proponent,
                    "opponent": opponent,
                }
                round_records.append(record)

                log(f"  🗣️  {hyp['id']} | 正方置信度: "
                    f"{proponent.get('confidence', '?')} | "
                    f"反方发现问题: {len(opponent.get('criticisms', []))} 个")

            all_debate_records.extend(round_records)

            # 检查辩论是否收敛
            avg_confidence = sum(
                r["proponent"].get("confidence", 0.5) for r in round_records
            ) / max(len(round_records), 1)
            if avg_confidence > TOURNAMENT_CONFIG["convergence_threshold"]:
                log(f"  ✅ 辩论收敛（平均置信度 {avg_confidence:.2f} > "
                    f"{TOURNAMENT_CONFIG['convergence_threshold']}）")
                break

        # ---- 阶段 3b: 评审打分 ----
        log("\n" + "-" * 45)
        log("📊 评审Agent综合打分")
        log("-" * 45)
        review = self.debater.review(hypotheses, all_debate_records,
                                     research_question)
        scores = review.get("scores", {})
        ranking = review.get("ranking", [])
        verdict = review.get("verdict", "")

        for rank, hyp_id in enumerate(ranking, 1):
            s = scores.get(hyp_id, {})
            log(f"  #{rank} {hyp_id}: novelty={s.get('novelty','?')}  "
                f"feasibility={s.get('feasibility','?')}  "
                f"impact={s.get('impact','?')}  "
                f"→ total={s.get('total','?')}")

        log(f"\n  📝 评审意见：{verdict[:100]}...")

        # ---- 阶段 4: 假设进化 ----
        log("\n" + "-" * 45)
        log("🧬 阶段 4/5：假设进化")
        log("-" * 45)

        # 选择排名第一的假设进行进化
        best_hyp_id = ranking[0] if ranking else hypotheses[0]["id"]
        best_hyp = next(h for h in hypotheses if h["id"] == best_hyp_id)

        # 准备记忆上下文
        memory_for_evolution = self.memory_store.get_context_for_task(
            research_question, max_items=5
        )

        evolution = self.evolver.evolve(
            best_hyp, review, all_debate_records,
            memory_for_evolution, research_question
        )
        evolved = evolution.get("evolved_hypothesis", {})
        evolution_log = evolved.get("evolution_log", [])

        log(f"  原始假设: {best_hyp['title']}")
        log(f"  进化后: {evolved.get('title', 'N/A')}")
        log(f"  进化改进项数: {len(evolution_log)}")
        for i, log_entry in enumerate(evolution_log, 1):
            log(f"    {i}. {log_entry[:80]}...")

        # ---- 阶段 5: 记忆学习 ----
        log("\n" + "-" * 45)
        log("💾 阶段 5/5：经验存入进化记忆库")
        log("-" * 45)
        self.memory_agent.learn_from_evolution(
            evolution, f"Task: {research_question[:60]}"
        )
        log(f"  {self.memory_store.summary()}")

        # ---- 组装最终结果 ----
        log("\n" + "=" * 65)
        log("  ✅ HypoEvo 流水线完成！")
        log("=" * 65)

        return {
            "research_question": research_question,
            "memory_context": memory_context,
            "generation": generation,
            "debates": all_debate_records,
            "review": review,
            "evolution": evolution,
            "evolved_hypothesis": evolved,
            "memory_stats": self.memory_store.stats,
        }
