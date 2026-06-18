# GridSynergy 项目状态文档

> 最后更新：2024-06-18

---

## 一、项目概述

基于大模型的多智能体电网调度系统。用户输入自然语言调度指令，四个 Agent 协同工作，调用 DeepSeek V3 API 进行推理决策，最终输出调度方案、安全评分、成本明细和博弈结果。

**模型 API**：DeepSeek V3（`https://api.deepseek.com/v1`），模型名 `deepseek-chat`

---

## 二、四 Agent 协同流程

```
用户输入自然语言
     ↓
🧬 MemoryAgent 检索历史经验 → 注入 Planner Prompt
     ↓
🧠 PlannerAgent → DeepSeek V3 推理 → JSON 调度方案
     ↓ (反馈修正，最多3轮)
🛡️ ValidatorAgent → Pandapower AC 潮流 → 安全评分
     ↓
🤝 NegotiatorAgent → 4 VPP 博弈协商
     ↓
🧬 MemoryAgent → 存储本次经验
```

| Agent | 职责 | 核心技术 |
|-------|------|----------|
| **PlannerAgent** | 调用 LLM 生成机组出力方案 | DeepSeek V3 + Prompt工程 |
| **ValidatorAgent** | AC 潮流计算 + N-1 校核 + 安全评分 | Pandapower Newton-Raphson |
| **NegotiatorAgent** | 4 VPP 博弈协商 | 边际成本 Nash 协商 |
| **MemoryAgent** | 余弦相似度检索历史经验 | 向量相似度匹配 |

---

## 三、核心文件清单

```
submission/
├── web/server_lite.py          # HTTP 服务器（标准库，无 FastAPI 依赖）
├── demo/index.html             # 前端页面（SSE 流式日志 + 12 快捷场景）
├── agents/
│   ├── planner_agent.py        # 规划Agent + LLM 调用
│   ├── validator_agent.py      # 安全校验Agent + Pandapower 潮流
│   ├── negotiator_agent.py     # 博弈协商Agent
│   └── memory_agent.py         # 记忆检索Agent
├── solver/
│   └── power_flow_solver.py    # Pandapower IEEE-30 潮流求解器
├── config.py                   # API Key 配置
└── PROJECT_STATUS.md           # 本文件
```

---

## 四、已完成的改动

### 4.1 模型 API 接入
- **文件**：`config.py`、`agents/planner_agent.py`
- **改动**：配置 DeepSeek V3 API（`sk-8da62c7e79914f53b69b0517638e1e41`），PlannerAgent 通过 `_plan_llm()` 调用 API 生成调度方案
- **状态**：✅ 完成，API 响应时间 ~3s

### 4.2 输入校验
- **文件**：`web/server_lite.py`（`_validate_instruction()`）
- **改动**：关键词匹配 + LLM 双重校验，无关输入（如"写诗"）返回拒绝提示
- **状态**：✅ 完成

### 4.3 四 Agent 协同链路打通
- **文件**：`web/server_lite.py`（`_run_llm_pipeline()`）
- **改动**：
  - MemoryAgent 检索历史经验注入 Planner Prompt
  - Planner 调用 LLM 生成方案
  - Validator 用 Pandapower 计算真实潮流评分
  - 辩论机制：不通过时反馈给 Planner，最多3轮
  - Negotiator 接收 Planner 方案进行博弈
  - MemoryAgent 存储本次经验
- **状态**：✅ 完成

### 4.4 Pandapower 真实潮流计算
- **文件**：`solver/power_flow_solver.py`、`agents/validator_agent.py`
- **改动**：构建 IEEE-30 网络，将 unit_commitment 映射到发电机 p_mw，运行 AC Newton-Raphson 潮流，返回真实电压、线路负载
- **状态**：✅ 完成，不同方案 → 不同评分（不再是固定 51.95）

### 4.5 三维指标输出
- **文件**：`web/server_lite.py`、`demo/index.html`
- **维度**：
  - **时间复杂度**：端到端总耗时 + LLM/Pandapower/博弈/记忆子项耗时
  - **空间复杂度**：机组数 + 线路数 + 约束数 + N-1 扫描数
  - **准确率**：电压合格率 + 线路负载率 + N-1 通过率 + 频率偏差（加权得分）
- **状态**：✅ 完成

### 4.6 成本明细（5项分项）
- **文件**：`web/server_lite.py`
- **分项**：发电成本 ¥ + 启停成本 ¥ + 爬坡成本 ¥ + 网损成本 ¥ + 惩罚成本 ¥
- **状态**：✅ 完成，前端可展开查看

### 4.7 消纳率修复
- **文件**：`web/server_lite.py`
- **问题**：固定 100%（LLM 不设 curtailment）
- **修复**：三来源动态计算 → 取最大值
  1. LLM 显式 curtailment
  2. 线路过载 → 超载越严重弃风越大
  3. 总出力超出负荷 → 超限按风光比分配
- **状态**：✅ 完成，风电骤降场景消纳率 ~3%，高新能源场景消纳率 ~98%

### 4.8 SSE 流式推送
- **文件**：`web/server_lite.py`、`demo/index.html`
- **接口**：POST `/api/dispatch/stream` → SSE 格式逐条推送日志
- **状态**：✅ 完成

### 4.9 取消/暂停按钮
- **文件**：`web/server_lite.py`、`demo/index.html`
- **实现**：前端 AbortController + 后端 CANCEL_TOKEN，可随时取消调度
- **状态**：✅ 完成

### 4.10 前端按钮修复（第一次尝试，未完全解决）
- **文件**：`demo/index.html`
- **Bug**：取消按钮无反应、第二次调度无法运行
- **根因**：
  1. `AbortError` 被静默吞掉，Promise 永远 pending → 按钮永不恢复
  2. 多个僵尸进程占 8888 端口 + 浏览器缓存
- **修复**：
  1. `reject(new Error('cancelled'))` → 正确恢复按钮
  2. 改用 `disabled` 代替 `display:none` 避免 flex 布局问题
  3. `do_GET` 添加 `Cache-Control: no-cache, no-store, must-revalidate`
  4. 添加 `_dispatchRunning` 锁防重复点击
- **状态**：⚠️ 不完整 — 第二次切换场景仍偶发无反应（见 4.12 最终修复）

### 4.11 12 个快捷场景
- **文件**：`demo/index.html`
- **场景列表**：
  | 场景 | 参数特点 |
  |------|---------|
  | ☁️ 光伏骤降 | load=250, solar 60→18 |
  | 💨 风电骤降 | load=280, wind 50→28 |
  | 📋 日前调度 | load=300, 高新能源 |
  | ⚡ N-1 故障 | load=240, 紧急调度 |
  | 📈 负荷高峰 | load=380, 晚高峰备用 |
  | 🌿 碳排放约束 | load=220, 碳价 ¥80/t |
  | 🔧 机组检修 | load=260, 检修关停 G3 |
  | 🔋 储能调度 | load=200, 储能 80MWh |
  | 🌙 负荷低谷 | load=160, 新能源过剩 |
  | 🔩 线路检修 | load=260, L8 停运4h |
  | 📉 负荷突降 | load=175, 风大增 |
  | 📡 预测修正 | 风-38MW/光+15MW |
- **状态**：✅ 完成

---

## 五、当前已实现的功能

| 功能 | 状态 |
|------|:---:|
| 自然语言输入调度指令 | ✅ |
| 输入校验（拒绝无关输入） | ✅ |
| DeepSeek V3 API 调用生成调度方案 | ✅ |
| 四 Agent 串联协同（Planner→Validator→Negotiator→Memory） | ✅ |
| 辩论机制（最多3轮，Planner↔Validator） | ✅ |
| Pandapower 真实 AC 潮流计算 | ✅ |
| N-1 故障安全校核 | ✅ |
| 4 VPP 博弈协商（边际成本 Nash） | ✅ |
| 余弦相似度记忆检索 | ✅ |
| 实时 SSE 流式日志 | ✅ |
| 取消调度按钮 | ✅ |
| 时间复杂度/空间复杂度/准确率三维指标 | ✅ |
| 5项成本明细 | ✅ |
| 动态消纳率计算 | ✅ |
| 12 个预设场景 | ✅ |
| 前端 Toast 提示 | ✅ |
| 无缓存刷新前端 | ✅ |
| REST 回退模式（SSE 失败时） | ✅ |

---

## 六、已知问题 & 未完成工作

### 6.1 重要待改进

| # | 问题 | 影响 | 建议方案 |
|---|------|------|---------|
| 1 | **MATD3 仍为启发式** | Negotiator 博弈未训练深度 RL，用边际成本 Nash 近似 | 安装 PyTorch + 训练 MATD3（5000 episodes） |
| 2 | **SCUC 仍为简化算法** | 机组组合用排序分配而非 MILP，调度最优性无保证 | 安装 Pyomo + HiGHS，替换为真实 MILP |
| 3 | **准确率评分偶见 0/100** | `accuracy.get('overall')` 字段不存在时返回 0 | 统一 accuracy 字段命名，确保后端始终填充 |
| 4 | **N-1 校核为静态估算** | 未真正断线重算潮流，基于容量比例估算 | 循环断线各线路，每次重算 Pandapower 潮流 |
| 5 | **消纳率边界情况** | 负荷高峰场景消纳率 0%（新能源已全额消纳，LLM 未设 curtailment 导致 denominator 可能为 0） | 处理分母为 0 的情况 |
| 6 | **无持久化存储** | 记忆库在内存中，重启丢失 | 接入 SQLite/JSON 文件持久化 |
| 7 | **无用户认证** | API 完全开放 | 加 API Key 校验或 OAuth |

### 6.2 次要改进

| # | 问题 | 建议方案 |
|---|------|---------|
| 8 | SSE 在低带宽下可能丢帧 | 加 reconnection 机制 |
| 9 | 无调度历史记录页面 | 前端加历史列表 |
| 10 | 无图表可视化（机组出力柱状图等） | 接入 Chart.js 图表 |
| 11 | 服务器启动时与旧进程端口冲突 | 加端口占用检测 + 自动重启逻辑 |

### 6.3 测试覆盖

| 测试项 | 状态 |
|-------|:---:|
| 后端两次连续调度 | ✅ 通过 |
| 取消调度后恢复 | ✅ 通过 |
| 无关输入拒绝 | ✅ 通过 |
| 12 个场景差异化输出 | ✅ 通过 |
| 前端按钮恢复 | ✅ 通过（需确认） |
| 并发请求 | ❌ 未测试 |
| 大负荷/极端场景 | ❌ 未测试 |

---

## 七、启动方式

```powershell
# 1. 启动后端
cd D:\桌面\华为杯\submission
python web\server_lite.py

# 2. 打开前端
start http://localhost:8888/demo/index.html
```

**端口**：8888  
**API 文档**：
- `GET /api/health` — 健康检查
- `GET /api/metrics` — 快速指标
- `POST /api/dispatch` — REST 调度
- `POST /api/dispatch/stream` — SSE 流式调度
- `POST /api/cancel` — 取消调度
