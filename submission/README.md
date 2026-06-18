# GridSynergy — 基于多智能体协同决策的新能源电网自主调度系统

## 项目简介

GridSynergy 是一个**LLM驱动的多智能体系统**，用于高比例新能源接入下的电网自主调度。系统通过四大核心Agent协同工作，实现从自然语言调度指令到安全、经济调度方案的全自动闭环。

### 四大核心Agent

| Agent | 职责 | 核心技术 |
|-------|------|----------|
| **PlannerAgent** | 根据调度指令生成候选方案 | LLM推理 + 经济调度规则 |
| **ValidatorAgent** | 多维度安全验证与评分 | N-1安全分析、电压/频率校验 |
| **NegotiatorAgent** | 多VPP经济博弈与协调 | MADRL (MATD3)、纳什均衡 |
| **MemoryAgent** | 经验记忆存储与检索 | 向量相似度检索、经验进化 |

## 系统架构

```
用户自然语言指令
        │
        ▼
┌─────────────────┐    检索相似场景    ┌─────────────────┐
│  MemoryAgent    │◄──────────────────►│  PlannerAgent   │
│  (经验记忆)     │                    │  (方案生成)     │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │                              候选调度方案
         │                                      │
         │                                      ▼
         │                            ┌─────────────────┐
         │                    不通过  │  ValidatorAgent │
         │              ◄────────────│  (安全验证)     │
         │              │            └────────┬────────┘
         │              │    辩论迭代         │通过
         │              └────────────────────┘
         │                                      │
         │                                      ▼
         │                            ┌─────────────────┐
         └───────────────────────────►│ NegotiatorAgent │
                                      │  (经济博弈)     │
                                      └────────┬────────┘
                                               │
                                               ▼
                                         最终调度方案
```

## 运行方式

### Demo模式（无需API Key，开箱即用）

```bash
# 安装依赖
pip install -r requirements.txt

# 运行Demo
python main.py
```

Demo模式使用内置的经济调度规则生成方案，无需配置LLM API Key。

### LLM模式（接入大模型推理）

```bash
# 设置API Key
export GRIDSYNERGY_API_KEY="your-api-key"
export GRIDSYNERGY_BASE_URL="https://api.openai.com/v1"   # 或其他兼容地址
export GRIDSYNERGY_MODEL="gpt-4o"

# 运行
python main.py
```

## 文件结构

```
submission/
├── config.py                     # 系统配置（LLM、Agent、电网、记忆库）
├── main.py                       # 主入口，完整调度流水线
├── requirements.txt              # Python依赖
├── README.md                     # 本文件
│
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py          # 规划Agent — 生成调度方案
│   ├── validator_agent.py        # 验证Agent — 安全校验
│   ├── negotiator_agent.py       # 博弈Agent — VPP经济博弈
│   └── memory_agent.py           # 记忆Agent — 经验管理
│
├── env/
│   ├── __init__.py
│   ├── power_grid_env.py         # 电网仿真环境（IEEE-30）
│   └── scenario_generator.py     # 场景生成器
│
├── solver/
│   ├── __init__.py
│   ├── power_flow.py             # 潮流计算模拟器
│   └── safety_checker.py         # 安全校验器
│
├── llm/
│   ├── __init__.py
│   └── llm_client.py             # LLM API调用客户端
│
└── output/                       # 运行输出（自动创建）
    ├── dispatch_results.json     # 调度结果
    └── memory_store.json         # 记忆库导出
```

## 输出说明

运行后将在 `output/` 目录生成：
- `dispatch_results.json`: 包含每个场景的完整调度方案、安全评分、经济收益
- `memory_store.json`: 记忆库序列化文件，可在后续运行中加载复用

## 技术特点

- **辩论机制**: Planner与Validator之间多轮辩论，确保方案安全性
- **Demo模式**: 无需任何LLM API Key即可完整运行和测试
- **模块化设计**: 各Agent独立封装，接口清晰，易于扩展
- **中文注释**: 全部代码包含详细中文注释，便于理解和二次开发
- **类型提示**: 所有公共接口均使用Python类型提示
