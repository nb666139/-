import { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import './Hero.css';

const floatingAnim = (delay = 0) => ({
  y: [0, -10, 0],
  transition: {
    duration: 4,
    repeat: Infinity,
    ease: 'easeInOut',
    delay,
  },
});

export default function Hero() {
  const ref = useRef(null);
  const navigate = useNavigate();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  });
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const y = useTransform(scrollYProgress, [0, 0.5], [0, 80]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95]);

  return (
    <section id="hero" className="hero-section" ref={ref}>
      <div className="energy-grid" />

      <motion.div className="hero-content" style={{ opacity, y, scale }}>
        <motion.p
          className="hero-tag"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          第八届中国研究生人工智能创新大赛
        </motion.p>

        <motion.h1
          className="hero-title"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
        >
          Grid
          <span className="text-gradient">Synergy</span>
        </motion.h1>

        <motion.p
          className="hero-subtitle"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
        >
          基于多智能体协同决策的
          <br />
          新能源电网自主调度系统
        </motion.p>

        <motion.p
          className="hero-desc"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          LLM驱动的语义推理 + 数值优化双引擎架构，
          <br />
          实现新能源电网日前-日内-实时全链条自主决策
        </motion.p>

        <motion.div
          className="hero-actions"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.55 }}
        >
          <motion.button
            className="btn-primary"
            whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(0,212,255,0.3)' }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/dispatch')}
          >
            探索系统
            <span className="btn-arrow">→</span>
          </motion.button>
          <motion.button
            className="btn-secondary"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/architecture')}
          >
            查看架构
          </motion.button>
        </motion.div>
      </motion.div>

      <motion.div className="floating-node node-1" animate={floatingAnim(0)} />
      <motion.div className="floating-node node-2" animate={floatingAnim(1.5)} />
      <motion.div className="floating-node node-3" animate={floatingAnim(3)} />
    </section>
  );
}
