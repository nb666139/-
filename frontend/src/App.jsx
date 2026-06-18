import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Dispatch from './pages/Dispatch';
import Architecture from './pages/Architecture';
import Results from './pages/Results';
import Footer from './components/Footer';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dispatch" element={<Dispatch />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/results" element={<Results />} />
        </Routes>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
