import React, { useEffect } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import Session from './pages/Session';
import Report from './pages/Report';
import Analysis from './pages/Analysis';
import VideoReport from './pages/VideoReport';
import ComprehensiveAnalysis from './pages/ComprehensiveAnalysis';
import ComprehensiveReport from './pages/ComprehensiveReport';
import Navbar from './components/Navbar';
import { AppProvider, useAppContext } from './context/AppContext';
import OnboardingModal from './components/OnboardingModal';

const AppContent: React.FC = () => {
  const { focusArea } = useAppContext();

  return (
    <>
      <Navbar />
      <main id="main-content" className="p-4 sm:p-6 md:p-8 max-w-7xl mx-auto w-full">
        <div className="glass-effect rounded-2xl p-4 sm:p-6 md:p-8 shadow-2xl">
          <Routes>
            <Route path="/" element={<Session />} />
            <Route path="/report" element={<Report />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/video-report" element={<VideoReport />} />
            <Route path="/comprehensive-analysis" element={<ComprehensiveAnalysis />} />
            <Route path="/comprehensive-report" element={<ComprehensiveReport />} />
          </Routes>
        </div>
      </main>
      {!focusArea && <OnboardingModal />}
    </>
  );
};


function App(): React.JSX.Element {
  useEffect(() => {
    document.documentElement.lang = 'en';
  }, []);

  return (
    <div className="bg-slate-900/0 text-white min-h-screen font-sans">
      <AppProvider>
        <HashRouter>
          <a
            href="#main-content"
            className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-white text-slate-900 p-2 z-50 rounded-md"
          >
            Skip to main content
          </a>
          <AppContent />
        </HashRouter>
      </AppProvider>
    </div>
  );
}

export default App;