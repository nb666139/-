import { motion } from 'framer-motion';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer-section">
      <div className="footer-inner">
        <div className="footer-top">
          <motion.div
            className="footer-brand"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <span className="footer-logo">⚡ GridSynergy</span>
            <p className="footer-tagline">
              LLM驱动的多智能体新能源电网自主调度系统
            </p>
          </motion.div>

          <motion.div
            className="footer-links"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h4>核心Agent</h4>
            <a href="#agents">Planner Agent</a>
            <a href="#agents">Validator Agent</a>
            <a href="#agents">Negotiator Agent</a>
            <a href="#agents">Memory Agent</a>
          </motion.div>

          <motion.div
            className="footer-links"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h4>相关链接</h4>
            <a href="#hero">首页</a>
            <a href="#agents">系统架构</a>
            <a href="#metrics">核心指标</a>
            <a href="#tech">技术栈</a>
          </motion.div>
        </div>

        <div className="footer-bottom">
          <span className="footer-copy">
            第八届中国研究生人工智能创新大赛 · GridSynergy
          </span>
        </div>
      </div>
    </footer>
  );
}
