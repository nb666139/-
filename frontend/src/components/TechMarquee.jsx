import { useRef } from 'react';
import { motion } from 'framer-motion';
import './TechMarquee.css';

const techStack = [
  { name: 'Python', icon: '🐍' },
  { name: 'PyTorch', icon: '🔥' },
  { name: 'LLM', icon: '🤖' },
  { name: 'MADRL', icon: '🎮' },
  { name: 'MILP', icon: '📐' },
  { name: 'IEEE 30', icon: '⚡' },
  { name: 'N-1 Analysis', icon: '🛡️' },
  { name: 'MATD3', icon: '🎯' },
  { name: 'VPP', icon: '🏭' },
  { name: 'SCUC', icon: '📊' },
  { name: 'FastAPI', icon: '🚀' },
  { name: 'React', icon: '⚛️' },
];

export default function TechMarquee() {
  const ref = useRef(null);

  return (
    <section id="tech" className="tech-section" ref={ref}>
      <motion.div
        className="section-header"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.6 }}
      >
        <span className="section-tag">Tech Stack</span>
        <h2 className="section-title">
          技术
          <span className="text-gradient">栈</span>
        </h2>
      </motion.div>

      <div className="marquee-wrap">
        <div className="marquee-track">
          {[...techStack, ...techStack].map((tech, i) => (
            <motion.div
              key={`${tech.name}-${i}`}
              className="tech-chip"
              whileHover={{ scale: 1.1, y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <span className="tech-icon">{tech.icon}</span>
              <span className="tech-name">{tech.name}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
