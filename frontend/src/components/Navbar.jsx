import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import './Navbar.css';

const navLinks = [
  { name: '首页', path: '/' },
  { name: '智能调度', path: '/dispatch' },
  { name: '系统架构', path: '/architecture' },
  { name: '实验结果', path: '/results' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const { scrollY } = useScroll();
  const location = useLocation();

  const navHeight = useTransform(scrollY, [0, 100], [80, 56]);
  const navBg = useTransform(
    scrollY,
    [0, 100],
    ['rgba(10, 10, 18, 0)', 'rgba(10, 10, 18, 0.85)']
  );

  useEffect(() => {
    const unsub = scrollY.on('change', (v) => setScrolled(v > 50));
    return () => unsub();
  }, [scrollY]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <motion.nav className="navbar" style={{ height: navHeight, backgroundColor: navBg }}>
      <div className="navbar-inner">
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Link to="/" className="nav-logo">
            <span className="logo-icon">⚡</span>
            <span className="logo-text">GridSynergy</span>
          </Link>
        </motion.div>

        <div className="nav-links">
          {navLinks.map((link) => (
            <motion.div key={link.path} whileHover={{ y: -2 }} transition={{ type: 'spring', stiffness: 300 }}>
              <Link
                to={link.path}
                className={`nav-link ${scrolled ? 'scrolled' : ''} ${
                  location.pathname === link.path ? 'active' : ''
                }`}
              >
                {link.name}
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.nav>
  );
}
