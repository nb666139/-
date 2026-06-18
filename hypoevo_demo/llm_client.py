"""
LLM 客户端 — 支持真实 API 调用和 Demo 模拟模式
"""
import json
from openai import OpenAI
from config import LLM_CONFIG, DEMO_MODE


class LLMClient:
    """统一的 LLM 调用接口"""

    def __init__(self):
        self.demo_mode = DEMO_MODE
        if not self.demo_mode:
            self.client = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
            )
            self.model = LLM_CONFIG["model"]

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = None) -> str:
        """调用 LLM 获取回复"""
        if self.demo_mode:
            return self._demo_response(system_prompt, user_prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature or LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
        )
        return response.choices[0].message.content

    def _demo_response(self, system_prompt: str, user_prompt: str) -> str:
        """Demo 模式：根据 system_prompt 中的人物角色返回模拟数据"""
        # 搜索完整 system_prompt 中的角色关键词
        sp = system_prompt.lower()

        if "生成agent" in sp or "hypothesis_count" in sp:
            return self._demo_generate(user_prompt)
        elif "评审" in sp or "reviewer" in sp:
            return self._demo_reviewer(user_prompt)
        elif "正方" in sp or "proponent" in sp:
            return self._demo_proponent(user_prompt)
        elif "反方" in sp or "opponent" in sp:
            return self._demo_opponent(user_prompt)
        elif "进化agent" in sp or "evolution" in sp:
            return self._demo_evolve(user_prompt)
        elif "记忆agent" in sp or "memory" in sp:
            return self._demo_memory(user_prompt)
        elif "不是聊天机器人" in sp or "AI Agent" in sp or "ai agent" in sp:
            return self._demo_agent_step(user_prompt)
        else:
            return "【Demo】分析结果：这是一个具有潜力的研究方向..."

    @staticmethod
    def _demo_agent_step(user_prompt: str) -> str:
        """Demo 模式：模拟 Agent 每步的 Thought + Action"""
        up = user_prompt.lower()
        # 判断当前是第几步
        if "开始思考" in up or ("step" not in up.lower() and "observation" not in up.lower()):
            # Step 1: 先搜论文
            return (
                "Thought: 我需要先通过文献检索验证假设的理论可行性，"
                "找到支撑证据和潜在风险。\n\n"
                'ACTION: arxiv_search\n'
                'PARAMS: {"query": "Janus MoSSe superconductivity strain 2D materials"}'
            )
        elif "arxiv" in up or "janus" in up or "mose" in up or "2d" in up:
            # Step 2: 有论文结果了，查询材料属性
            return (
                "Thought: 文献确认了Janus TMDs的理论潜力，"
                "现在需要验证MoSSe的声子谱稳定性——这是超导的前提条件。\n\n"
                'ACTION: materials_db_query\n'
                'PARAMS: {"material_formula": "MoSSe", "property": "phonon_dispersion"}'
            )
        elif "mosse" in up or "phonon" in up or "声子" in up:
            # Step 3: 声子稳定，估算 Tc
            return (
                "Thought: 声子谱稳定无虚频，基础条件满足。"
                "现在用McMillan公式估算Tc，考虑等离激元贡献。\n\n"
                'ACTION: dft_estimator\n'
                'PARAMS: {"material": "MoSSe", "estimate_type": "tc_mcmillan", '
                '"input_params": {"strain": 0.08, "lambda": 2.8, "Theta_D": 380, "mu_star": 0.13}}'
            )
        elif "tc" in up or "mcmillan" in up or "estimated" in up or "dft" in up:
            # Step 4: 执行数值验证
            return (
                "Thought: McMillan估算Tc~310K。让我用Python验证这个计算结果。\n\n"
                "ACTION: code_executor\n"
                "PARAMS: {\n"
                '  "code": "import math\\n'
                'lambda_val = 2.8\\n'
                'Theta_D = 380\\n'
                'mu_star = 0.13\\n'
                'Tc = (Theta_D/1.45) * math.exp(-1.04*(1+lambda_val)/(lambda_val - mu_star*(1+0.62*lambda_val)))\\n'
                'print(f\'McMillan Tc = {Tc:.1f} K\')\\n'
                'print(\'可性评估: 室温超导候选 ✓\')\\n'
                'print(\'联合机制: BCS(λ_ep=2.0) + 等离激元(λ_pl=0.8) = λ_total=2.8\')",\n'
                '  "description": "McMillan Tc 计算验证"\n}'
            )
        elif "code" in up or "输出" in up or "执行" in up or "python" in up:
            # Step 5: 保存文件
            return (
                "Thought: 所有验证步骤完成，生成最终科学假设报告并保存。\n\n"
                "ACTION: file_writer\n"
                "PARAMS: {\n"
                '  "filename": "hypothesis_report.md",\n'
                '  "content": "# 科学假设验证报告\\n\\n'
                '## 题目: 二维Janus TMDs室温超导\\n\\n'
                '## 核心预言\\nhBN/MoSSe/hBN异质结中，8%应变+等离激元机制 → Tc~310K\\n\\n'
                '## 验证路径\\n'
                '1. ✅ 文献检索: 找到4篇支撑论文 (Nature Nanotech/PRL/Nature Comm)\\n'
                '2. ✅ 声子谱: MoSSe无虚频，动力学稳定\\n'
                '3. ✅ McMillan估算: Tc≈310K (λ=2.8, Θ_D=380K)\\n'
                '4. ✅ 数值验证: Python计算确认\\n\\n'
                '## 下一步\\n- 实验ARPES验证费米面\\n- DFT详细Eliashberg计算\\n- 器件制备与输运测量",\n'
                '  "mode": "w"\n}'
            )
        else:
            return (
                "Thought: 分析当前信息，下一步需要深入文献研究。\n\n"
                'ACTION: arxiv_search\n'
                'PARAMS: {"query": "superconductivity 2D materials strain engineering"}'
            )

    # ---- Demo 模拟数据 ----

    @staticmethod
    def _demo_generate(user_prompt: str) -> str:
        return json.dumps({
            "hypotheses": [
                {
                    "id": "H1",
                    "title": "二维Janus过渡金属硫族化合物中的室温超导",
                    "framework": "基于BCS理论与电子-声子耦合",
                    "prediction": (
                        "在MoSSe/WSSe异质结中，通过面内压缩应变(>6%)诱导费米面处"
                        "vdH奇点，可使电声耦合常数λ>2.0，实现Tc~300K的超导转变。"
                        "关键优势：二维Janus结构天然破缺面外对称性，无需外电场即可调控。"
                    ),
                    "testability": (
                        "可通过DFT计算验证：(1)声子谱无虚频；(2)Eliashberg函数计算"
                        "λ和ω_log；(3)McMillan公式估算Tc。实验验证需ARPES测费米面+"
                        "输运测量。"
                    ),
                    "references": ["PRL 2024", "Nature Materials 2025"],
                },
                {
                    "id": "H2",
                    "title": "高压氢化物LaH₁₀-yFy中的近室温超导",
                    "framework": "基于化学预压缩效应与氢主导超导机制",
                    "prediction": (
                        "在LaH₁₀中部分氟取代（y=0.5-1.5），通过F⁻的电负性调控氢"
                        "亚晶格的化学压力，可在较低外加压力(<50 GPa)下实现Tc~280K。"
                        "氟取代同时钝化表面，提高空气稳定性。"
                    ),
                    "testability": (
                        "理论：DFT演化结构搜索+电子-声子计算。实验：金刚石对顶砧+"
                        "激光加热合成+电阻/磁化率测量。已有LaH₁₀合成经验可直接复用。"
                    ),
                    "references": ["Nature 2019 (Drozdov)"],
                },
                {
                    "id": "H3",
                    "title": "转角三层石墨烯中关联绝缘态邻近的超导",
                    "framework": "基于平带电子关联与量子几何效应",
                    "prediction": (
                        "在转角三层石墨烯(1.5°-1.8°)的关联绝缘态与半金属态掺杂边界，"
                        "量子几何贡献可增强超流刚度，实现Tc~5K的超导。"
                        "与魔角双层不同，三层体系具有额外的层自由度可调控能带拓扑。"
                    ),
                    "testability": (
                        "理论：连续介质模型+HF+DMFT多体计算。实验：干法转移+"
                        "顶栅静电掺杂+低温输运(稀释制冷机,<50mK)。"
                        "需要超高质量hBN封装器件。"
                    ),
                    "references": ["Science 2022", "Nature Physics 2023"],
                },
            ]
        }, ensure_ascii=False)

    @staticmethod
    def _demo_proponent(user_prompt: str) -> str:
        return json.dumps({
            "role": "proponent",
            "favored_hypothesis": "H1",
            "arguments": [
                "Janus TMDs已有实验合成(Nature Nanotech 2017)，起点高",
                "二维体系声子软化效应强于三维，有利于高超导Tc",
                "应变工程在二维材料中成熟可控，实验可实现>8%应变",
                "最新arXiv预印本(2025.06)已报道WSSe中观测到Tc~120K迹象",
            ],
            "confidence": 0.85,
        }, ensure_ascii=False)

    @staticmethod
    def _demo_opponent(user_prompt: str) -> str:
        return json.dumps({
            "role": "opponent",
            "criticisms": [
                "压缩应变>6%在自由悬浮单层中可行，但器件中受衬底钉扎仅能达3-4%",
                "BCS框架下λ>2.0会同时导致严重的电荷密度波(CDW)不稳定性，可能破坏超导",
                "二维材料库仑屏蔽差，等离激元效应未被计入，可能压制配对",
                "WSSe中120K超导迹象尚未被独立重复验证",
            ],
            "alternative_suggestion": (
                "建议转向H2氢化物路径：LaH₁₀体系已有可信Tc~250K实验记录，"
                "氟掺杂在同一团队2024年JACS论文中概念验证成功"
            ),
        }, ensure_ascii=False)

    @staticmethod
    def _demo_reviewer(user_prompt: str) -> str:
        return json.dumps({
            "role": "reviewer",
            "scores": {
                "H1": {"novelty": 9.0, "feasibility": 5.5, "impact": 9.5, "total": 8.0},
                "H2": {"novelty": 6.5, "feasibility": 8.5, "impact": 8.0, "total": 7.7},
                "H3": {"novelty": 8.0, "feasibility": 6.0, "impact": 7.0, "total": 7.0},
            },
            "ranking": ["H1", "H2", "H3"],
            "verdict": (
                "H1创新性最高，但可行性需加强——建议补充衬底工程方案解决应变瓶颈。"
                "H2可行性最强但创新性不足，可考虑作为保底验证方向。"
                "H3实验难度大(稀释制冷+高质量器件)，适合有经验的实验组。"
                "综合推荐：以H1为主攻，H2为并行验证。"
            ),
        }, ensure_ascii=False)

    @staticmethod
    def _demo_evolve(user_prompt: str) -> str:
        return json.dumps({
            "evolved_hypothesis": {
                "id": "H1-E",
                "title": (
                    "H1进化版：引入衬底解耦缓冲层 + 等离激元介导配对机制的"
                    "二维Janus TMDs室温超导"
                ),
                "framework": "BCS + 等离激元介导配对 + 衬底工程",
                "evolution_log": [
                    "采纳反方意见：用hBN/石墨烯缓冲层解耦衬底，恢复自由悬浮应变能力",
                    "采纳反方意见：补充等离激元贡献，修正Tc估算（从纯BCS λ=2.0→联合机制λ'=2.8）",
                    "采纳正方补充：引用最新Nature Comm 2025关于hBN缓冲层实现>8%应变的实验",
                    "CDW竞争问题：通过载流子掺杂调控，在CDW量子临界点附近寻求最大Tc",
                ],
                "refined_prediction": (
                    "在hBN/MoSSe/hBN范德华异质结中，通过压电应变器实现8%面内压缩应变，"
                    "同时激发声学等离激元参与配对。联合Eliashberg+等离激元框架预测Tc~310K。"
                    "CDW竞争序通过门压调控载流子浓度至~10¹⁴ cm⁻²来压制。"
                ),
                "testability": "改进后可验证性评级：7.5/10",
            },
            "memory_contribution": {
                "success_patterns": [
                    "二维体系下衬底工程是解锁高应变的通用策略",
                    "多配对机制联合考虑显著提升Tc预测——不应只用纯BCS",
                    "预言前必须系统性检查竞争序(CDW/SDW/电荷序)",
                ],
                "failure_patterns": [
                    "假设衬底为刚性约束是常见陷阱——实际界面滑移可达3-5%",
                    "忽略二维材料中的等离激元效应会系统性低估Tc",
                ],
            },
        }, ensure_ascii=False)

    @staticmethod
    def _demo_memory(user_prompt: str) -> str:
        return json.dumps({
            "retrieved_insights": [
                {
                    "type": "success",
                    "content": "量子几何效应在平带体系中可贡献>50%的超流刚度",
                    "source": "上次碳基超导假设任务验证通过",
                    "weight": 0.9,
                },
                {
                    "type": "failure",
                    "content": "纯碳基材料(石墨烯/碳纳米管)中电声耦合不足，Tc普遍<1K",
                    "source": "2025-01材料搜索任务验证失败",
                    "weight": -0.8,
                },
                {
                    "type": "success",
                    "content": "Janus结构的天然内建电场是调控电子结构的高效手段",
                    "source": "2025-03铁电超晶格任务",
                    "weight": 0.7,
                },
            ],
            "recommendation": (
                "历史记忆显示：您的团队在二维体系+应变工程方面成功率高达75%，"
                "建议主攻H1（Janus TMDs），同时避开已经验证失败的纯碳路径。"
                "另外注意：上一次失败的原因是忽视了CDW竞争序——本次已在进化中修正。"
            ),
        }, ensure_ascii=False)
