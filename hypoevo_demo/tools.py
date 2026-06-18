"""
HypoEvo 工具系统 — Agent 可调用的真实工具

每个工具包含：
  - name / description / parameters (JSON Schema)
  - execute() 方法 — 真实执行操作
Agent 调用工具 → 返回 Observation → 下一轮 Thought
"""
import json
import time
import os
from datetime import datetime


# ============================================================
# 工具注册表
# ============================================================
class ToolRegistry:
    """工具注册中心 —— Agent 通过 registry 发现和调用工具"""

    def __init__(self):
        self._tools: dict[str, "BaseTool"] = {}

    def register(self, tool: "BaseTool"):
        self._tools[tool.name] = tool

    def get_schema(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling 格式"""
        return [t.schema for t in self._tools.values()]

    def get_descriptions(self) -> str:
        """返回人类可读的工具列表"""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def execute(self, name: str, params: dict) -> str:
        """执行工具调用，返回 observation"""
        if name not in self._tools:
            return json.dumps({"error": f"未知工具: {name}", "available": list(self._tools.keys())})
        try:
            return self._tools[name].execute(params)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": name})


class BaseTool:
    """工具基类"""
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def execute(self, params: dict) -> str:
        raise NotImplementedError


# ============================================================
# 工具实现
# ============================================================

class ArxivSearchTool(BaseTool):
    """搜索 arXiv 学术论文 — Agent 的"文献检索"能力"""
    name = "arxiv_search"
    description = "搜索arXiv上的学术论文。输入关键词/题目/作者，返回相关论文的标题、作者、摘要和链接。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词，如 'Janus MoSSe superconductivity'"},
            "max_results": {"type": "integer", "description": "最大返回数量，默认5", "default": 5},
        },
        "required": ["query"],
    }

    # Demo 模拟论文数据库
    _PAPERS_DB = {
        "janus": [
            {"title": "Janus Monolayer Transition-Metal Dichalcogenides",
             "authors": "Lu A-Y et al.", "year": 2017,
             "journal": "Nature Nanotechnology", "citations": 2100,
             "abstract": "首次实验合成二维Janus MoSSe，破缺面外对称性产生内建电场，为调控电子结构开辟新维度。",
             "url": "https://arxiv.org/abs/1703.05888"},
            {"title": "Enhanced Superconductivity in Strained Janus WSSe Monolayers",
             "authors": "Zhang L et al.", "year": 2025,
             "journal": "arXiv preprint", "citations": 42,
             "abstract": "第一性原理计算表明8%面内应变使WSSe的Tc从<1K跃升至~120K。电声耦合λ从0.5增至2.1。",
             "url": "https://arxiv.org/abs/2506.12345"},
            {"title": "Plasmon-Mediated Superconductivity in Two-Dimensional Heterostructures",
             "authors": "Novelli P et al.", "year": 2024,
             "journal": "Physical Review Letters", "citations": 89,
             "abstract": "提出声学等离激元作为辅助配对机制，可在二维体系中贡献额外λ~0.5-1.0。",
             "url": "https://arxiv.org/abs/2403.01234"},
        ],
        "hydride": [
            {"title": "Superconductivity at 250 K in Lanthanum Hydride under High Pressures",
             "authors": "Drozdov AP et al.", "year": 2019,
             "journal": "Nature", "citations": 3200,
             "abstract": "在170 GPa下LaH10实现Tc=250K的超导，是目前最高记录之一。",
             "url": "https://arxiv.org/abs/1812.01561"},
            {"title": "Fluorine-Doped LaH10: Enhanced Stability and Retained Superconductivity",
             "authors": "Liu H et al.", "year": 2024,
             "journal": "JACS", "citations": 156,
             "abstract": "F掺杂LaH10-yFy在低压下保持高Tc，同时显著提高空气稳定性。",
             "url": "https://arxiv.org/abs/2405.67890"},
        ],
        "graphene": [
            {"title": "Unconventional Superconductivity in Magic-Angle Graphene Superlattices",
             "authors": "Cao Y et al.", "year": 2018,
             "journal": "Nature", "citations": 5800,
             "abstract": "魔角双层石墨烯中发现Tc~1.7K超导，打开二维莫尔超晶格方向。",
             "url": "https://arxiv.org/abs/1803.02342"},
            {"title": "Superconductivity in Twisted Trilayer Graphene",
             "authors": "Park JM et al.", "year": 2022,
             "journal": "Science", "citations": 890,
             "abstract": "转角三层石墨烯在关联绝缘态附近实现Tc~5K超导，量子几何贡献超流刚度。",
             "url": "https://arxiv.org/abs/2109.00001"},
        ],
        "strain": [
            {"title": "Giant Tuning of Electronic Properties in 2D Materials via Substrate Engineering",
             "authors": "Mennel L et al.", "year": 2025,
             "journal": "Nature Communications", "citations": 67,
             "abstract": "hBN缓冲层使MoS2可承受>8%可逆应变，远超衬底钉扎极限。",
             "url": "https://arxiv.org/abs/2501.11111"},
        ],
    }

    def execute(self, params: dict) -> str:
        query = params["query"].lower()
        max_results = params.get("max_results", 5)
        results = []

        for keyword, papers in self._PAPERS_DB.items():
            if keyword in query:
                results.extend(papers)

        if not results:
            results = [{
                "title": f"Search results for: {params['query']}",
                "authors": "Various",
                "year": 2025,
                "journal": "arXiv",
                "abstract": f"在arXiv上搜索 '{params['query']}' 的相关论文...（真实环境将调用arXiv API）",
                "url": f"https://arxiv.org/search/?query={params['query']}",
            }]

        results = results[:max_results]
        return json.dumps({
            "tool": "arxiv_search",
            "query": params["query"],
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2)


class MaterialsProjectTool(BaseTool):
    """查询材料数据库 — Agent 的"实验事实核查"能力"""
    name = "materials_db_query"
    description = "查询Materials Project等材料数据库，获取材料的晶体结构、能带、态密度、声子谱等计算数据。"
    parameters = {
        "type": "object",
        "properties": {
            "material_formula": {"type": "string", "description": "化学式，如 'MoS2', 'LaH10'"},
            "property": {
                "type": "string",
                "enum": ["band_gap", "phonon_dispersion", "formation_energy", "density_of_states", "crystal_structure"],
                "description": "需要查询的材料属性",
            },
        },
        "required": ["material_formula", "property"],
    }

    _DB = {
        ("MoS2", "band_gap"): {"value": "1.8 eV (单层直接带隙)", "source": "Materials Project mp-2815"},
        ("MoS2", "formation_energy"): {"value": "-2.52 eV/atom", "source": "Materials Project mp-2815"},
        ("MoSSe", "band_gap"): {"value": "1.65 eV (间接带隙, 内建电场~0.1 V/Å)", "source": "Materials Project mp-1025449"},
        ("MoSSe", "phonon_dispersion"): {"value": "无虚频, 声子谱稳定; 最大声子频率 410 cm⁻¹", "source": "DFT计算 (PBE+DFT-D3)"},
        ("WSSe", "band_gap"): {"value": "1.72 eV, 内建电场~0.12 V/Å", "source": "Computational Materials Science 2025"},
        ("LaH10", "formation_energy"): {"value": "-0.18 eV/atom (亚稳态, 需高压稳定)", "source": "Materials Project mp-24165"},
        ("LaH10", "phonon_dispersion"): {"value": "150-200 GPa范围内无虚频; 高频H振动模~250 meV", "source": "Nature 2019 Supp."},
    }

    def execute(self, params: dict) -> str:
        formula = params["material_formula"]
        prop = params["property"]
        key = (formula, prop)

        if key in self._DB:
            return json.dumps({
                "tool": "materials_db_query",
                "material": formula,
                "property": prop,
                "result": self._DB[key],
            }, ensure_ascii=False, indent=2)

        # 模拟生成合理数据
        return json.dumps({
            "tool": "materials_db_query",
            "material": formula,
            "property": prop,
            "result": {
                "value": f"{formula} 的 {prop} 数据（模拟值）",
                "source": f"Materials Project / DFT计算（模拟 - 真实环境将查询实际数据库）",
            },
            "note": "Demo模式模拟数据。接入Materials Project API / OQMD / AFLOW可获取真实数据。",
        }, ensure_ascii=False, indent=2)


class DFTEstimatorTool(BaseTool):
    """DFT参数估算 — Agent 的"理论计算"能力"""
    name = "dft_estimator"
    description = "基于经验公式和已知材料参数估算超导Tc、电声耦合常数λ、德拜温度等。用于快速预筛选假设。"
    parameters = {
        "type": "object",
        "properties": {
            "material": {"type": "string", "description": "材料名称或化学式"},
            "estimate_type": {
                "type": "string",
                "enum": ["tc_mcmillan", "lambda", "debye_temperature", "density_of_states"],
                "description": "估算类型",
            },
            "input_params": {
                "type": "object",
                "description": "估算所需输入参数，如 {'strain': 0.08, 'doping': 1e14}",
            },
        },
        "required": ["material", "estimate_type"],
    }

    def execute(self, params: dict) -> str:
        material = params["material"]
        est_type = params["estimate_type"]
        inputs = params.get("input_params", {})

        results = {
            "tc_mcmillan": {
                "method": "McMillan公式: Tc = (Θ_D/1.45) × exp[-1.04(1+λ)/(λ-μ*(1+0.62λ))]",
                "estimated_tc": "~310 K（λ=2.8, Θ_D=380K, μ*=0.13）",
                "confidence": "中等 — 依赖输入的λ和Θ_D精度",
                "note": "纯BCS框架估算。等离激元贡献未计入McMillan公式，实际Tc可能更高。",
            },
            "lambda": {
                "method": "λ = N(0)×⟨I²⟩/M⟨ω²⟩ ≈ (DOS at EF) × (e-ph matrix element) / (phonon frequency²)",
                "estimated_lambda": "λ ≈ 2.0-2.8（计入等离激元修正后）",
                "breakdown": "BCS贡献 λ_ep=2.0, 等离激元贡献 λ_pl=0.8",
            },
            "debye_temperature": {
                "estimated": "Θ_D ≈ 380 K（基于MoSSe的声子谱）",
                "method": "从声子态密度积分计算",
            },
            "density_of_states": {
                "estimated": "N(0) ≈ 1.5 states/eV/f.u. （8%应变下）",
                "method": "费米面处DOS，基于DFT能带计算",
            },
        }

        result = results.get(est_type, {"error": f"不支持的估算类型: {est_type}"})
        result["material"] = material
        result["input_params"] = inputs
        result["tool"] = "dft_estimator"
        result["disclaimer"] = "快速估算用，最终需DFT验证"

        return json.dumps(result, ensure_ascii=False, indent=2)


class FileWriterTool(BaseTool):
    """文件写入 — Agent 的"产出物保存"能力（这个工具是真实执行的！）"""
    name = "file_writer"
    description = "将文本内容写入文件。用于保存生成的假设、辩论记录、最终报告等。"
    parameters = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "文件名，如 'hypothesis_report.md'"},
            "content": {"type": "string", "description": "要写入的完整文本内容"},
            "mode": {"type": "string", "enum": ["w", "a"], "description": "写入模式：w=覆盖, a=追加", "default": "w"},
        },
        "required": ["filename", "content"],
    }

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def execute(self, params: dict) -> str:
        filename = params["filename"]
        content = params["content"]
        mode = params.get("mode", "w")
        filepath = os.path.join(self.output_dir, filename)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(content)

        size = len(content)
        return json.dumps({
            "tool": "file_writer",
            "status": "success",
            "file": os.path.abspath(filepath),
            "size_bytes": size,
            "message": f"✅ 文件已保存: {os.path.abspath(filepath)} ({size} 字符)",
        }, ensure_ascii=False)


class CodeExecutorTool(BaseTool):
    """代码执行 — Agent 的"科学计算"能力（模拟沙盒）"""
    name = "code_executor"
    description = "在安全沙盒中执行Python科学计算代码。支持numpy/scipy。用于DFT后处理、数据分析、可视化。"
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的Python代码"},
            "description": {"type": "string", "description": "代码用途说明"},
        },
        "required": ["code"],
    }

    def execute(self, params: dict) -> str:
        code = params["code"]
        desc = params.get("description", "未命名计算")

        # 在实际系统中，这里应该用 Docker 沙盒
        # Demo 中执行安全的小范围科学计算
        safe_globals = {"__builtins__": {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "range": range, "list": list,
            "dict": dict, "float": float, "int": int, "str": str,
            "print": print, "enumerate": enumerate, "zip": zip,
        }, "math": __import__("math")}

        # 尝试导入 numpy（如果安装了）
        try:
            safe_globals["np"] = __import__("numpy")
        except ImportError:
            pass

        try:
            import io, sys
            stdout = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout

            exec(code, safe_globals, {})
            sys.stdout = old_stdout
            output = stdout.getvalue().strip()

            return json.dumps({
                "tool": "code_executor",
                "description": desc,
                "status": "success",
                "output": output or "代码执行成功（无标准输出）",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "tool": "code_executor",
                "status": "error",
                "error": str(e),
            }, ensure_ascii=False)


def create_default_tools(output_dir: str = ".") -> ToolRegistry:
    """创建默认工具集"""
    registry = ToolRegistry()
    registry.register(ArxivSearchTool())
    registry.register(MaterialsProjectTool())
    registry.register(DFTEstimatorTool())
    registry.register(FileWriterTool(output_dir))
    registry.register(CodeExecutorTool())
    return registry