# GridSynergy — 基于多智能体协同决策的新能源电网自主调度系统

> 第八届中国研究生人工智能创新大赛 - v1

## 快速启动

### 1. 启动后端（Python）

```bash
cd submission
pip install -r requirements.txt
python web/server_lite.py
```

后端运行在 `http://localhost:8888`

### 2. 启动前端（React + Vite）

```bash
cd frontend
npm install
npm run dev
```

前端运行在 `http://localhost:5173`

### 3. 打开浏览器

访问 `http://localhost:5173` 即可使用系统。

## 技术栈

| 前端 | 后端 |
|------|------|
| React 19 + Vite | Python 3.10+ |
| Framer Motion | LLM (DeepSeek V3) |
| Chart.js | Pyomo + Gurobi |
| React Router | pandapower |
| react-chartjs-2 | MATD3 (MADRL) |

## 核心功能

- **智能调度**: 12个预设场景 + 自定义参数 + SSE实时Agent日志
- **系统架构**: Planner / Validator / Negotiator / Memory 四大Agent
- **实验结果**: 雷达图 + 6方法对比 + 消融实验 + 调度历史记录
- **多方法对比**: SCUC-MILP / 随机SCUC / Grid-Agent / LLM-SUC / MADDPG-VPP / GridSynergy

## 项目结构

```
├── submission/          # Python 后端
│   ├── web/
│   │   └── server_lite.py
│   ├── agents/
│   │   ├── planner_agent.py
│   │   ├── validator_agent.py
│   │   ├── negotiator_agent.py
│   │   └── memory_agent.py
│   ├── llm/
│   │   └── input_guard.py
│   └── config.py
│
├── frontend/            # React 前端
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Dispatch.jsx
│   │   │   ├── Architecture.jsx
│   │   │   └── Results.jsx
│   │   └── components/
│   │       ├── Navbar.jsx
│   │       ├── Hero.jsx
│   │       ├── Metrics.jsx
│   │       ├── AgentCards.jsx
│   │       ├── TechMarquee.jsx
│   │       └── Footer.jsx
│   └── vite.config.js
│
├── start_backend.bat
├── start_frontend.bat
└── .gitignore
```
