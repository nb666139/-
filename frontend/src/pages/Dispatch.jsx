import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import './Dispatch.css';

const API_BASE = '';

const PRESET_SCENARIOS = [
  { label: '☁️ 光伏骤降', desc: '负荷250MW · 风电80 · 光伏18', load: 250, wind: 80, solar: 18, mode: 'day_ahead', inst: '明日下午华北区域预计云层覆盖，光伏出力可能下降40%，请调整日前发电计划' },
  { label: '💨 风电骤降', desc: '负荷280MW · 风电28 · 光伏50', load: 280, wind: 28, solar: 50, mode: 'real_time', inst: '华东区域风速突降，风电出力预计从50MW降至28MW，请启动燃气备用并调整区域间联络线功率' },
  { label: '📋 日前调度', desc: '负荷300MW · 风电90 · 光伏60', load: 300, wind: 90, solar: 60, mode: 'day_ahead', inst: '请生成本周日前发电计划，最小化总成本，新能源消纳率不低于95%，N-1安全约束必须满足' },
  { label: '⚠️ N-1故障', desc: '负荷240MW · 风电60 · 光伏40', load: 240, wind: 60, solar: 40, mode: 'real_time', inst: '线路L12发生N-1故障跳闸，请立即生成拓扑重构和负荷再分配方案' },
  { label: '📈 负荷高峰', desc: '负荷380MW · 风电50 · 光伏20', load: 380, wind: 50, solar: 20, mode: 'real_time', inst: '明日晚高峰7-9点负荷预计增加30%，请提前调整机组出力并预留旋转备用' },
  { label: '🌿 碳排放约束', desc: '负荷260MW · 风电100 · 光伏70', load: 260, wind: 100, solar: 70, mode: 'day_ahead', inst: '日碳排放配额限制为500吨，碳排放因子较高机组降出力，优先调用清洁能源' },
  { label: '🔧 机组检修', desc: '负荷270MW · 风电55 · 光伏35', load: 270, wind: 55, solar: 35, mode: 'real_time', inst: 'G3机组因故障停机检修，请重新分配负荷，确保系统安全运行' },
  { label: '🔋 储能调度', desc: '负荷230MW · 风电85 · 光伏55', load: 230, wind: 85, solar: 55, mode: 'day_ahead', inst: '储能电站当前SOC为60%，请生成最优充放电策略配合新能源消纳' },
  { label: '🌙 负荷低谷', desc: '负荷160MW · 风电100 · 光伏80', load: 160, wind: 100, solar: 80, mode: 'real_time', inst: '凌晨负荷低谷，新能源出力过剩，电压偏高，请削减部分常规机组出力' },
  { label: '🔩 线路检修', desc: '负荷260MW · 风电55 · 光伏35', load: 260, wind: 55, solar: 35, mode: 'day_ahead', inst: '线路L8计划检修需停运4小时，请提前调整拓扑并重新分配潮流' },
  { label: '📉 负荷突降', desc: '负荷175MW · 风电110 · 光伏45', load: 175, wind: 110, solar: 45, mode: 'real_time', inst: '负荷突降30%至175MW，东风西风同时大增，请紧急降出力并保证频率稳定' },
  { label: '📡 预测修正', desc: '负荷290MW · 风电42 · 光伏75', load: 290, wind: 42, solar: 75, mode: 'real_time', inst: '日前风电预测80MW但实际只有42MW，光伏预测60MW实际有75MW，请修正发电计划' },
];

export default function Dispatch() {
  const [activeScenario, setActiveScenario] = useState(-1);
  const [totalLoad, setTotalLoad] = useState(250);
  const [windForecast, setWindForecast] = useState(60);
  const [solarForecast, setSolarForecast] = useState(30);
  const [instruction, setInstruction] = useState('');

  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);

  const [comparing, setComparing] = useState(false);
  const [compareResults, setCompareResults] = useState([]);
  const [comparingMethod, setComparingMethod] = useState('');

  const logRef = useRef(null);
  const sseRef = useRef(null);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [logs]);

  const selectScenario = (idx) => {
    if (running) return;
    setActiveScenario(idx);
    const s = PRESET_SCENARIOS[idx];
    setTotalLoad(s.load);
    setWindForecast(s.wind);
    setSolarForecast(s.solar);
    setInstruction(s.inst);
  };

  const clearScenario = () => {
    if (running) return;
    setActiveScenario(-1);
    setInstruction('');
  };

  const startDispatch = () => {
    if (running || !instruction.trim()) return;
    cancelSSE();
    setRunning(true);
    setCancelling(false);
    setLogs([]);
    setResult(null);
    setCompareResults([]);

    setLogs([]);
    dispatchViaFetch();
  };

  const dispatchViaFetch = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/dispatch/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction,
          total_load: totalLoad,
          wind_forecast: windForecast,
          solar_forecast: solarForecast,
        }),
      });

      if (!resp.ok) {
        setLogs(prev => [...prev, { time: '--', agent: 'Error', msg: `HTTP ${resp.status}`, isResult: true }]);
        setRunning(false);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (currentEvent === 'log') {
                setLogs(prev => [...prev, { time: data.time, agent: data.agent, msg: data.message, isResult: data.is_result }]);
              } else if (currentEvent === 'result') {
                if (data.status === 'success') setResult(data);
                else setLogs(prev => [...prev, { time: '--', agent: 'Error', msg: data.message || '调度异常', isResult: true }]);
                setRunning(false);
              } else if (currentEvent === 'cancelled') {
                setLogs(prev => [...prev, { time: '--', agent: 'System', msg: '调度已取消', isResult: true }]);
                setRunning(false);
                setCancelling(false);
              } else if (currentEvent === 'rejected') {
                setLogs(prev => [...prev, { time: '--', agent: 'Guard', msg: `输入被拒绝: ${data.reason}`, isResult: true }]);
                setRunning(false);
              }
            } catch (e) { /* skip */ }
          }
        }
      }
    } catch (err) {
      setLogs(prev => [...prev, { time: '--', agent: 'Error', msg: `连接异常: ${err.message} (请确认后端已启动)`, isResult: true }]);
    }
    setRunning(false);
  };

  const cancelDispatch = async () => {
    if (cancelling) return;
    setCancelling(true);
    try { await fetch(`${API_BASE}/api/cancel`, { method: 'POST' }); } catch (e) { /* ignore */ }
  };

  const cancelSSE = () => {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
  };

  const runComparison = async () => {
    if (comparing) return;
    setComparing(true);
    setCompareResults([]);
    setComparingMethod('');

    try {
      const resp = await fetch(`${API_BASE}/api/comparison/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction,
          total_load: totalLoad,
          wind_forecast: windForecast,
          solar_forecast: solarForecast,
        }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) currentEvent = line.slice(7).trim();
          else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (currentEvent === 'comparing') setComparingMethod(data.method);
              else if (currentEvent === 'result') setCompareResults(prev => [...prev, data]);
              else if (currentEvent === 'done') { setComparing(false); setComparingMethod(''); }
            } catch (e) { /* skip */ }
          }
        }
      }
    } catch (err) {
      setLogs(prev => [...prev, { time: '--', agent: 'Error', msg: `对比异常: ${err.message}`, isResult: true }]);
    }
    setComparing(false);
    setComparingMethod('');
  };

  const getAgentColor = (agent) => {
    const map = {
      'System': '#9ca3af', '系统守卫': '#a78bfa', '规划Agent': '#00d4ff', '验证Agent': '#10b981',
      '博弈Agent': '#f59e0b', '记忆Agent': '#a78bfa', 'GridSynergy': '#00d4ff',
      'Error': '#ef4444', 'Guard': '#a78bfa',
    };
    return map[agent] || '#9ca3af';
  };

  return (
    <motion.div className="dispatch-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
      <h1>智能<span className="text-gradient">调度</span></h1>
      <p className="page-desc">选择预设场景或自定义参数，运行多智能体调度流水线</p>

      <div className="dispatch-grid">
        {/* Left: Config */}
        <div className="panel">
          <h3>📋 场景配置</h3>

          <div className="scenario-list">
            {PRESET_SCENARIOS.map((s, i) => (
              <motion.button
                key={i}
                className={`scenario-chip ${activeScenario === i ? 'active' : ''}`}
                onClick={() => selectScenario(i)}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <span className="scenario-label">{s.label}</span>
                <span className="scenario-meta">{s.desc}</span>
              </motion.button>
            ))}
          </div>

          <div className="scenario-list" style={{ marginTop: 0 }}>
            <motion.button
              className={`scenario-chip ${activeScenario === -1 ? 'active' : ''}`}
              onClick={clearScenario}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <span className="scenario-label">✏️ 自定义输入</span>
              <span className="scenario-meta">手动设置所有参数</span>
            </motion.button>
          </div>

          <div className="form-group">
            <label>总负荷 (MW) <span className="val">{totalLoad}</span></label>
            <input type="range" min={100} max={400} value={totalLoad}
              onChange={e => { setTotalLoad(Number(e.target.value)); setActiveScenario(-1); }}
              disabled={running} />
          </div>

          <div className="form-group">
            <label>风电预测 (MW) <span className="val">{windForecast}</span></label>
            <input type="range" min={0} max={150} value={windForecast}
              onChange={e => { setWindForecast(Number(e.target.value)); setActiveScenario(-1); }}
              disabled={running} />
          </div>

          <div className="form-group">
            <label>光伏预测 (MW) <span className="val">{solarForecast}</span></label>
            <input type="range" min={0} max={120} value={solarForecast}
              onChange={e => { setSolarForecast(Number(e.target.value)); setActiveScenario(-1); }}
              disabled={running} />
          </div>

          <div className="form-group">
            <label>调度指令</label>
            <textarea value={instruction} onChange={e => { setInstruction(e.target.value); setActiveScenario(-1); }}
              disabled={running} placeholder="输入自然语言调度指令，例如：明日下午光伏出力预计下降30%，请调整日前计划..." />
          </div>

          {running ? (
            <motion.button className="btn-start cancelling" onClick={cancelDispatch}
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              {cancelling ? '⏳ 正在取消...' : '⏹ 取消调度'}
            </motion.button>
          ) : (
            <motion.button className="btn-start" onClick={startDispatch}
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              disabled={!instruction.trim()}>
              ⚡ 开始调度
            </motion.button>
          )}
        </div>

        {/* Right: Log + Results */}
        <div className="panel">
          <h3>📡 实时Agent日志</h3>

          <div className="log-panel" ref={logRef}>
            {logs.length === 0 && !running ? (
              <div className="log-placeholder">选择场景并点击"开始调度"查看多Agent协作全过程</div>
            ) : (
              logs.map((entry, i) => (
                <div key={i} className="log-entry">
                  <span className="log-time">{entry.time}</span>
                  <span className="log-agent" style={{ color: getAgentColor(entry.agent) }}>
                    [{entry.agent}]
                  </span>
                  <span className={`log-msg ${entry.isResult ? 'result' : ''}`}>
                    {entry.isResult ? '▸ ' : ''}{entry.msg}
                  </span>
                </div>
              ))
            )}
            {running && (
              <div className="log-entry">
                <span className="log-time">--:--:--</span>
                <span className="log-agent" style={{ color: '#f59e0b' }}>[...]</span>
                <span className="log-msg" style={{ color: '#f59e0b' }}>Agent协作进行中...</span>
              </div>
            )}
          </div>

          {result && result.status === 'success' && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
              <div className="result-grid">
                <div className="result-card">
                  <div className="result-val" style={{ color: '#10b981' }}>¥{(result.cost / 10000).toFixed(2)}万</div>
                  <div className="result-lbl">调度成本</div>
                </div>
                <div className="result-card">
                  <div className="result-val" style={{ color: '#00d4ff' }}>{result.res_rate}%</div>
                  <div className="result-lbl">新能源消纳率</div>
                </div>
                <div className="result-card">
                  <div className="result-val" style={{ color: result.n1_pass_rate >= 100 ? '#10b981' : '#f59e0b' }}>
                    {result.n1_pass_rate}%
                  </div>
                  <div className="result-lbl">N-1通过率</div>
                </div>
              </div>

              <motion.button className="btn-compare" onClick={runComparison} disabled={comparing}
                whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                {comparing ? `⏳ 对比中... ${comparingMethod}` : '🔄 多方法对比分析'}
              </motion.button>
            </motion.div>
          )}
        </div>
      </div>

      {compareResults.length > 0 && (
        <motion.div className="comparison-section" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <h2>🔄 多方法对比结果</h2>
          <div className="compare-grid">
            {compareResults.sort((a, b) => a.index - b.index).map((r) => (
              <motion.div key={r.method}
                className={`compare-card ${r.method === 'GridSynergy' ? 'active' : ''}`}
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.03, y: -4 }} transition={{ type: 'spring', stiffness: 300 }}>
                <div className="cm-name">{r.method}</div>
                <div className="cm-val" style={{ color: r.method === 'GridSynergy' ? '#00d4ff' : '#9ca3af' }}>
                  ¥{r.cost_wan?.toFixed(2) || '-'}万
                </div>
                <div className="cm-lbl">成本</div>
                <div className="cm-val" style={{ color: '#10b981', fontSize: 14 }}>{r.res_rate?.toFixed(1) || '-'}%</div>
                <div className="cm-lbl">消纳率</div>
                <div className="cm-val" style={{ color: '#f59e0b', fontSize: 14 }}>{r.shed_mwh?.toFixed(2) || '-'} MWh</div>
                <div className="cm-lbl">切负荷</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
