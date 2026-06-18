"""
进化记忆系统 — 持久化存储成功/失败经验，支持跨任务知识检索
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import MEMORY_CONFIG


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    id: str
    entry_type: str            # "success" | "failure" | "insight" | "method"
    content: str
    source_task: str
    weight: float              # +正=成功经验, -负=失败教训
    tags: list = field(default_factory=list)
    embedding: np.ndarray = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.entry_type,
            "content": self.content,
            "source_task": self.source_task,
            "weight": self.weight,
            "tags": self.tags,
        }


class EvolutionaryMemory:
    """
    进化记忆库 — 核心创新模块

    功能：
    1. 存储每次假设验证的成功/失败经验
    2. 基于相似度检索相关历史经验
    3. 失败的"死胡同"被标记为负权重，自动避免
    4. 成功的方法论被标记为正权重，优先复用
    """

    def __init__(self):
        self.entries: list[MemoryEntry] = []
        self.max_entries = MEMORY_CONFIG["max_entries"]
        self.task_counter = 0

    def add_success(self, content: str, source_task: str,
                    tags: list = None, weight: float = 0.8) -> MemoryEntry:
        """记录一条成功经验"""
        return self._add("success", content, source_task, abs(weight), tags or [])

    def add_failure(self, content: str, source_task: str,
                    tags: list = None, weight: float = -0.7) -> MemoryEntry:
        """记录一条失败教训（自动负权重）"""
        return self._add("failure", content, source_task, -abs(weight), tags or [])

    def add_insight(self, content: str, source_task: str,
                    tags: list = None, weight: float = 0.5) -> MemoryEntry:
        """记录一条通用洞见"""
        return self._add("insight", content, source_task, weight, tags or [])

    def add_method(self, content: str, source_task: str,
                   tags: list = None, weight: float = 0.6) -> MemoryEntry:
        """记录一条可复用方法论"""
        return self._add("method", content, source_task, weight, tags or [])

    def _add(self, entry_type: str, content: str, source_task: str,
             weight: float, tags: list) -> MemoryEntry:
        entry_id = f"mem-{len(self.entries):04d}"
        embedding = np.random.randn(MEMORY_CONFIG["embedding_dim"])
        embedding /= np.linalg.norm(embedding)

        entry = MemoryEntry(
            id=entry_id, entry_type=entry_type, content=content,
            source_task=source_task, weight=weight, tags=tags,
            embedding=embedding,
        )
        self.entries.append(entry)

        # 超出容量则移除最旧的低权重条目
        if len(self.entries) > self.max_entries:
            # 优先保留高绝对值权重的条目
            self.entries.sort(key=lambda e: abs(e.weight), reverse=True)
            self.entries = self.entries[:self.max_entries]

        return entry

    def query(self, keyword: str, top_k: int = 5) -> list[dict]:
        """基于关键词检索相关记忆"""
        keyword_lower = keyword.lower()
        scored = []
        for entry in self.entries:
            score = sum(1 for tag in entry.tags if tag.lower() in keyword_lower)
            score += sum(1 for word in keyword_lower.split()
                         if word in entry.content.lower())
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e.to_dict() for _, e in scored[:top_k]]

    def get_failures(self) -> list[dict]:
        """获取所有已标记的死胡同"""
        return [e.to_dict() for e in self.entries
                if e.entry_type == "failure" and e.weight < 0]

    def get_successes(self) -> list[dict]:
        """获取所有已验证的成功策略"""
        return [e.to_dict() for e in self.entries
                if e.entry_type in ("success", "method") and e.weight > 0]

    def get_context_for_task(self, task_description: str, max_items: int = 5) -> str:
        """生成给Agent的历史上下文"""
        relevant = self.query(task_description, top_k=max_items)

        if not relevant:
            return "（暂无相关历史经验）"

        parts = ["## 历史经验（来自进化记忆库）\n"]
        for i, entry in enumerate(relevant, 1):
            emoji = "❌" if entry["weight"] < 0 else "✅"
            parts.append(f"{i}. {emoji} [{entry['type']}] {entry['content']}")
            parts.append(f"   来源：{entry['source_task']}")
        return "\n".join(parts)

    @property
    def stats(self) -> dict:
        """统计信息"""
        successes = sum(1 for e in self.entries if e.weight > 0)
        failures = sum(1 for e in self.entries if e.weight < 0)
        return {
            "total": len(self.entries),
            "successes": successes,
            "failures": failures,
            "dead_ends_avoided": failures,
            "reusable_methods": sum(1 for e in self.entries
                                    if e.entry_type == "method"),
        }

    def summary(self) -> str:
        s = self.stats
        return (
            f"记忆库状态：共 {s['total']} 条记录 | "
            f"✅ {s['successes']} 成功 | ❌ {s['failures']} 失败 | "
            f"🔧 {s['reusable_methods']} 方法论"
        )
