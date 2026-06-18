"""
GridSynergy — 基于多智能体协同决策的新能源电网自主调度系统
系统配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _load_dotenv(env_path: str = ".env") -> None:
    """加载 .env 文件中的环境变量（不覆盖已有环境变量）。"""
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key not in os.environ:
                os.environ[key] = value


# 启动时加载 .env
_load_dotenv()


# ============================================================================
# LLM API 配置
# ============================================================================
@dataclass
class LLMConfig:
    """大语言模型API配置，支持OpenAI兼容接口"""
    # API密钥，通过环境变量 GRIDSYNERGY_API_KEY 设置，Demo模式下可留空
    api_key: str = field(
        default_factory=lambda: os.environ.get("GRIDSYNERGY_API_KEY", "")
    )
    # API基础URL（支持任何OpenAI兼容的服务，如vLLM、DeepSeek等）
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GRIDSYNERGY_BASE_URL", "https://api.openai.com/v1"
        )
    )
    # 模型名称
    model: str = field(
        default_factory=lambda: os.environ.get("GRIDSYNERGY_MODEL", "gpt-4o")
    )
    # 生成温度（0-2），越低越确定
    temperature: float = 0.3
    # 最大输出Token数
    max_tokens: int = 4096
    # 请求超时时间（秒）
    timeout: int = 120
    # 最大重试次数
    max_retries: int = 3


# ============================================================================
# Agent 配置
# ============================================================================
@dataclass
class AgentConfig:
    """多智能体协作配置"""
    # Planner-Validator辩论最大轮次
    max_debate_rounds: int = 3
    # Planner每轮生成的候选方案数
    num_candidates: int = 3
    # 安全评分阈值（低于此值触发回退）
    safety_threshold: float = 75.0
    # Negotiator博弈收敛阈值
    nash_convergence_epsilon: float = 0.01
    # Negotiator最大迭代轮次
    max_negotiation_rounds: int = 50
    # MADRL折扣因子
    gamma: float = 0.95
    # MADRL软更新系数
    tau: float = 0.005
    # Actor网络学习率
    actor_lr: float = 1e-4
    # Critic网络学习率
    critic_lr: float = 1e-3


# ============================================================================
# 电网配置
# ============================================================================
@dataclass
class PowerGridConfig:
    """电网仿真环境配置"""
    # IEEE标准节点数（默认30节点系统）
    num_nodes: int = 30
    # 发电机数量
    num_generators: int = 6
    # 新能源渗透率（风电+光伏占比，0-1）
    renewable_penetration: float = 0.35
    # 线路数量（IEEE-30标准约41条）
    num_lines: int = 41
    # 基准电压（kV）
    base_voltage: float = 135.0
    # 基准功率（MVA）
    base_power: float = 100.0
    # 电压允许范围（标幺值）
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    # N-1安全校核开关（始终开启）
    n1_check_enabled: bool = True


# ============================================================================
# 记忆库配置
# ============================================================================
@dataclass
class MemoryConfig:
    """经验记忆库配置"""
    # 最大存储条目数
    max_entries: int = 10000
    # 检索返回的Top-K条相近场景
    top_k: int = 5
    # 场景特征向量维度（用于相似度检索）
    feature_dim: int = 128
    # 相似度阈值（低于此值不计入检索结果）
    similarity_threshold: float = 0.6
    # 成功经验的正向加权系数
    success_boost: float = 1.5
    # 失败经验的负向加权系数
    failure_penalty: float = 0.5
    # 经验衰减因子（老经验权重随时间衰减）
    decay_factor: float = 0.99


# ============================================================================
# 训练配置（MATD3 RL训练）
# ============================================================================
@dataclass
class TrainingConfig:
    """MATD3深度强化学习训练配置"""
    # 总训练 episode 数
    n_episodes: int = 5000
    # 每个 episode 的最大步数
    max_steps: int = 100
    # 训练 batch 大小
    batch_size: int = 256
    # 经验回放缓冲区容量
    replay_buffer_capacity: int = 1_000_000
    # 训练开始前的最小缓冲区填充量
    min_buffer_size: int = 1000
    # 探索噪声标准差
    exploration_noise: float = 0.1
    # 目标策略平滑噪声
    policy_noise: float = 0.2
    # 噪声截断范围
    noise_clip: float = 0.5
    # 策略更新延迟（每隔多少步更新）
    policy_delay: int = 2
    # 模型检查点保存间隔（每N个episode）
    checkpoint_interval: int = 500
    # 模型保存目录
    checkpoint_dir: str = "./checkpoints"
    # TensorBoard日志目录
    log_dir: str = "./logs"
    # 设备（cuda 或 cpu）
    device: str = "cuda" if os.environ.get("GRIDSYNERGY_USE_GPU", "1") == "1" else "cpu"


# ============================================================================
# 全局配置聚合
# ============================================================================
@dataclass
class SystemConfig:
    """GridSynergy系统全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    grid: PowerGridConfig = field(default_factory=PowerGridConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # 是否为Demo模式（无需API Key即可运行）
    demo_mode: bool = field(
        default_factory=lambda: os.environ.get("GRIDSYNERGY_API_KEY", "") == ""
    )
    # 是否为训练模式
    training_mode: bool = field(
        default_factory=lambda: os.environ.get("GRIDSYNERGY_TRAIN_MODE", "0") == "1"
    )
    # 输出目录
    output_dir: str = "./output"
    # 数据目录
    data_dir: str = "./data"


# 全局单例
_system_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """获取全局系统配置单例"""
    global _system_config
    if _system_config is None:
        _system_config = SystemConfig()
    return _system_config


def reset_config() -> None:
    """重置配置（主要用于测试）"""
    global _system_config
    _system_config = None
