import { useRef } from 'react';
import { motion } from 'framer-motion';
import './AgentCards.css';

const agents = [
  {
    icon: '🧠',
    title: 'Planner Agent',
    subtitle: '规划智能体',
    description: '基于LLM语义推理，解析调度指令，生成多时间尺度发电计划与拓扑调整方案',
    tech: 'LLM推理 + 经济调度',
    color: '#00d4ff',
  },
  {
    icon: '🛡️',
    title: 'Validator Agent',
    subtitle: '验证智能体',
    description: '沙箱环境中执行N-1安全校验、潮流计算与回滚保护，确保方案安全可行',
    tech: 'N-1安全分析',
    color: '#10b981',
  },
  {
    icon: '⚔️',
    title: 'Negotiator Agent',
    subtitle: '博弈智能体',
    description: '基于MADRL实现多区域VPP非合作博弈经济调度，逼近纳什均衡最优解',
    tech: 'MADRL (MATD3)',
    color: '#f59e0b',
  },
  {
    icon: '💾',
    title: 'Memory Agent',
    subtitle: '记忆智能体',
    description: '维护进化记忆库，通过向量相似度检索实现跨场景调度策略智能复用',
    tech: '向量检索 + 经验进化',
    color: '#a78bfa',
  },
];

const container = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15 },
  },
};

const item = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

export default function AgentCards() {
  const ref = useRef(null);

  return (
    <section id="agents" className="agents-section" ref={ref}>
      <motion.div
        className="section-header"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.6 }}
      >
        <span className="section-tag">System Architecture</span>
        <h2 className="section-title">
          四大核心
          <span className="text-gradient">Agent</span>
        </h2>
        <p className="section-desc">
          LLM驱动的多智能体系统，实现从自然语言指令到安全调度方案的全自动闭环
        </p>
      </motion.div>

      <motion.div
        className="agents-grid"
        variants={container}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '-60px' }}
      >
        {agents.map((agent) => (
          <motion.div
            key={agent.title}
            className="agent-card"
            variants={item}
            whileHover={{
              scale: 1.03,
              y: -8,
              boxShadow: `0 20px 50px rgba(0,0,0,0.3), 0 0 30px ${agent.color}15`,
              borderColor: `${agent.color}40`,
            }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          >
            <div
              className="agent-icon-wrap"
              style={{ background: `${agent.color}15`, color: agent.color }}
            >
              <span className="agent-icon">{agent.icon}</span>
            </div>
            <div className="agent-info">
              <h3 className="agent-name">{agent.title}</h3>
              <p className="agent-subtitle">{agent.subtitle}</p>
              <p className="agent-desc">{agent.description}</p>
              <span
                className="agent-tech"
                style={{ color: agent.color, borderColor: `${agent.color}30` }}
              >
                {agent.tech}
              </span>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
