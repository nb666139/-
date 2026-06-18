"""
GridSynergy — 记忆Agent (MemoryAgent)
负责存储、检索和管理历史调度经验，为Planner和Validator提供场景参考。
具有经验进化机制：成功经验正加权，失败经验负加权。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config import get_config


# ============================================================================
# MemoryEntry — 单条记忆数据结构
# ============================================================================

@dataclass
class MemoryEntry:
    """
    单条记忆数据结构。

    存储一次完整的调度经验：场景特征、调度方案、验证结果和评估标签。
    """
    # 场景特征向量（归一化）
    scene_features: np.ndarray
    # 调度指令原文
    dispatch_instruction: str
    # 生成的调度方案
    dispatch_plan: dict[str, Any]
    # 验证结果
    validation_result: dict[str, Any]
    # 安全评分
    safety_score: float
    # 是否成功（评分 >= 阈值）
    is_success: bool
    # 经验权重（初始为1.0，进化机制会调整）
    weight: float = 1.0
    # 使用次数（用于衰减计算）
    usage_count: int = 0
    # 创建时间戳
    timestamp: float = field(default_factory=time.time)
    # 经验ID（唯一标识）
    entry_id: str = ""

    def __post_init__(self) -> None:
        """自动生成经验ID"""
        if not self.entry_id:
            import uuid
            self.entry_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化的字典"""
        return {
            "entry_id": self.entry_id,
            "scene_features": self.scene_features.tolist(),
            "dispatch_instruction": self.dispatch_instruction,
            "dispatch_plan": self.dispatch_plan,
            "validation_result": self.validation_result,
            "safety_score": self.safety_score,
            "is_success": self.is_success,
            "weight": self.weight,
            "usage_count": self.usage_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """从字典恢复MemoryEntry"""
        return cls(
            scene_features=np.array(data["scene_features"], dtype=np.float32),
            dispatch_instruction=data["dispatch_instruction"],
            dispatch_plan=data["dispatch_plan"],
            validation_result=data["validation_result"],
            safety_score=data["safety_score"],
            is_success=data["is_success"],
            weight=data.get("weight", 1.0),
            usage_count=data.get("usage_count", 0),
            timestamp=data.get("timestamp", time.time()),
            entry_id=data.get("entry_id", ""),
        )


# ============================================================================
# MemoryAgent
# ============================================================================

class MemoryAgent:
    """
    记忆Agent

    维护一个经验记忆库，支持：
    1. 基于场景特征的相似经验检索
    2. 新经验的存储
    3. 经验的进化（成功正加权 / 失败负加权）
    4. 检索结果整理为自然语言上下文
    """

    def __init__(self) -> None:
        """初始化记忆Agent"""
        self._config = get_config()
        # 记忆库
        self._memory_store: list[MemoryEntry] = []
        # 场景特征维度
        self._feature_dim: int = self._config.memory.feature_dim

    def retrieve(
        self, scene_features: np.ndarray, top_k: int | None = None
    ) -> list[MemoryEntry]:
        """
        检索与当前场景最相似的Top-K条历史经验。

        参数:
            scene_features: 当前场景的特征向量 [feature_dim]
            top_k: 返回条目数，默认使用配置值

        返回:
            按相似度（带权重调整）降序排列的记忆条目列表
        """
        if top_k is None:
            top_k = self._config.memory.top_k

        if not self._memory_store:
            return []

        # 确保特征向量维度一致
        features: np.ndarray = self._normalize_features(scene_features)

        # 计算与所有记忆的余弦相似度 × 经验权重
        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in self._memory_store:
            entry_features: np.ndarray = self._normalize_features(entry.scene_features)

            # 余弦相似度
            dot_product: float = float(np.dot(features, entry_features))
            norm_product: float = float(np.linalg.norm(features) * np.linalg.norm(entry_features))
            cosine_sim: float = dot_product / max(norm_product, 1e-8)

            # 最终得分 = 相似度 × 加权系数 × 衰减因子
            decay: float = self._config.memory.decay_factor ** entry.usage_count
            final_score: float = cosine_sim * entry.weight * decay

            if final_score >= self._config.memory.similarity_threshold:
                scored_entries.append((final_score, entry))

        # 按得分降序排序
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # 更新使用计数
        for _, entry in scored_entries[:top_k]:
            entry.usage_count += 1

        return [entry for _, entry in scored_entries[:top_k]]

    def store(self, entry_or_features, dispatch_plan=None, safety_score=None, is_success=None) -> None:
        """
        存储一条新的调度经验。

        支持两种调用方式：
        1. store(MemoryEntry) — 传入完整 MemoryEntry 对象
        2. store(features_dict, plan, score, success) — 便捷参数调用
        """
        from datetime import datetime
        
        if isinstance(entry_or_features, MemoryEntry):
            entry = entry_or_features
        else:
            # 便捷调用：从字典构建 MemoryEntry
            features = entry_or_features
            if isinstance(features, dict):
                vec = np.array([
                    float(features.get("total_load", 250)) / 500.0,
                    float(features.get("wind_forecast", 60)) / 200.0,
                    float(features.get("solar_forecast", 30)) / 200.0,
                    (features.get("total_load", 250) - features.get("wind_forecast", 60) - features.get("solar_forecast", 30)) / 500.0,
                ], dtype=np.float32)
            else:
                vec = np.asarray(features, dtype=np.float32).flatten()
            
            entry = MemoryEntry(
                scene_features=vec,
                dispatch_instruction=features.get("instruction", "") if isinstance(features, dict) else "",
                dispatch_plan=dispatch_plan or {},
                validation_result={"safety_score": safety_score, "is_success": is_success},
                safety_score=float(safety_score or 0.0),
                is_success=bool(is_success),
            )

        # 检查容量上限，超出则移除权重最低的条目
        if len(self._memory_store) >= self._config.memory.max_entries:
            self._memory_store.sort(key=lambda e: e.weight * (
                self._config.memory.decay_factor ** e.usage_count
            ))
            self._memory_store.pop(0)  # 移除最无价值的

        # 特征向量标准化
        entry.scene_features = self._normalize_features(entry.scene_features)
        self._memory_store.append(entry)

    def get_context(self, scene_features: np.ndarray) -> str:
        """
        将检索到的相似经验整理为可供Planner/Validator参考的自然语言上下文。
        """
        entries: list[MemoryEntry] = self.retrieve(scene_features)

        if not entries:
            return "暂无相似历史场景可供参考。"

        context_parts: list[str] = [f"## 历史相似场景参考（共 {len(entries)} 条）\n"]

        for i, entry in enumerate(entries, 1):
            success_label: str = "✅ 成功" if entry.is_success else "❌ 失败"
            pass_label = "安全通过" if entry.validation_result.get("safety_score", 0) >= 75 else "存在越限"
            context_parts.append(
                f"### 参考场景 {i} [{success_label}，{pass_label}]\n"
                f"- 调度指令: {entry.dispatch_instruction}\n"
                f"- 方案摘要: {entry.dispatch_plan.get('summary', 'N/A')}\n"
                f"- 期望成本: {entry.dispatch_plan.get('expected_cost', 'N/A')}\n"
                f"- 经验权重: {entry.weight:.2f}\n"
            )

        return "\n".join(context_parts)

    def retrieve_context(self, **kwargs) -> str:
        """从字典参数构建特征向量并检索上下文。
        
        支持: retrieve_context(total_load=250, wind_forecast=60, solar_forecast=30)
        """
        vec = np.array([
            float(kwargs.get("total_load", 250)) / 500.0,
            float(kwargs.get("wind_forecast", 60)) / 200.0,
            float(kwargs.get("solar_forecast", 30)) / 200.0,
            (float(kwargs.get("total_load", 250)) - float(kwargs.get("wind_forecast", 60)) - float(kwargs.get("solar_forecast", 30))) / 500.0,
        ], dtype=np.float32)
        return self.get_context(vec)

    def evolve(self) -> None:
        """
        记忆进化机制：
        - 成功经验（is_success=True）：正向加权（boost）
        - 失败经验（is_success=False）：负向加权（penalty）
        - 所有经验随时间衰减
        """
        success_boost: float = self._config.memory.success_boost
        failure_penalty: float = self._config.memory.failure_penalty

        for entry in self._memory_store:
            if entry.is_success:
                entry.weight *= success_boost
            else:
                entry.weight *= failure_penalty

            # 限制权重范围 [0.1, 5.0]
            entry.weight = max(0.1, min(5.0, entry.weight))

    def get_statistics(self) -> dict[str, Any]:
        """获取记忆库统计信息"""
        total: int = len(self._memory_store)
        if total == 0:
            return {"total_entries": 0, "success_rate": 0.0, "avg_safety_score": 0.0}

        success_count: int = sum(1 for e in self._memory_store if e.is_success)
        avg_score: float = float(np.mean([e.safety_score for e in self._memory_store]))

        return {
            "total_entries": total,
            "success_count": success_count,
            "failure_count": total - success_count,
            "success_rate": round(success_count / total * 100, 1),
            "avg_safety_score": round(avg_score, 2),
            "avg_weight": round(float(np.mean([e.weight for e in self._memory_store])), 3),
        }

    def save_to_file(self, filepath: str) -> None:
        """保存记忆库到JSON文件"""
        data: list[dict[str, Any]] = [e.to_dict() for e in self._memory_store]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """从JSON文件加载记忆库"""
        import os
        if not os.path.exists(filepath):
            print(f"[MemoryAgent] 文件不存在：{filepath}")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
        self._memory_store = [MemoryEntry.from_dict(d) for d in data]
        print(f"[MemoryAgent] 从文件加载了 {len(self._memory_store)} 条记忆")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征向量：确保维度和归一化。
        """
        features = np.asarray(features, dtype=np.float32).flatten()

        # 裁剪/填充至统一维度
        if len(features) > self._feature_dim:
            features = features[:self._feature_dim]
        elif len(features) < self._feature_dim:
            features = np.pad(features, (0, self._feature_dim - len(features)))

        # L2归一化
        norm: float = float(np.linalg.norm(features))
        if norm > 1e-8:
            features = features / norm

        return features
