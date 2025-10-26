import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { SpeakerWaveIcon, ChartBarIcon, UserIcon, PlayIcon, StopIcon, EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';
import { useAppContext } from '../context/AppContext';
import VideoFeed from '../components/VideoFeed';
import { useSpeechAnalysis } from '../hooks/useSpeechAnalysis';
import { usePostureAnalysis } from '../hooks/usePostureAnalysis';
import { AnalysisReport } from '../types';
import Tooltip from '../components/Tooltip';

const FeedbackStat: React.FC<{ icon: React.ReactNode; label: string; value: string; unit: string; isPulsing?: boolean; isFocused?: boolean; status?: 'normal' | 'warning' }> = ({ icon, label, value, unit, isPulsing = false, isFocused = false, status = 'normal' }) => (
  <div className={`bg-slate-800 p-4 rounded-lg flex items-center space-x-4 border-2 transition-all duration-300 ${isPulsing ? (status === 'warning' ? 'animate-pulse-feedback-warning' : 'animate-pulse-feedback') : ''} ${isFocused ? 'border-indigo-500 shadow-lg shadow-indigo-500/20 scale-105' : 'border-transparent hover:bg-slate-700/50'}`}>
    <div className="bg-slate-850 p-3 rounded-full">
      {icon}
    </div>
    <div>
      <div className="text-sm text-slate-400">{label}</div>
      <div className="text-2xl font-bold text-white">
        {value} <span className="text-base font-normal text-slate-400">{unit}</span>
      </div>
    </div>
  </div>
);


const Session: React.FC = () => {
  useEffect(() => {
    document.title = "Aura: AI Public Speaking Coach - Practice Session";
  }, []);

  const navigate = useNavigate();
  const { focusArea, topicKeywords, setSessionReport } = useAppContext();
  const { isListening, transcript, liveFillerCount, liveWPM, startListening, stopListening } = useSpeechAnalysis();
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const { postureFeedback, startPostureAnalysis, stopPostureAnalysis, getPostureReport } = usePostureAnalysis(videoRef);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPerformanceMode, setIsPerformanceMode] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState('');

  const handleStartSession = () => {
    startListening(topicKeywords);
    startPostureAnalysis();
  };

  const handleStopSession = () => {
    const speechReport = stopListening(topicKeywords);
    stopPostureAnalysis();
    const postureReport = getPostureReport();
    
    const report: AnalysisReport = {
      ...speechReport,
      ...postureReport,
    };
    setSessionReport(report);
    
    setIsAnalyzing(true);
    
    const today = new Date().toDateString();
    const lastSessionDate = localStorage.getItem('auraLastSessionDate');
    const streak = parseInt(localStorage.getItem('auraStreak') || '0', 10);
    const sessionCount = parseInt(localStorage.getItem('auraSessionCount') || '0', 10);

    if (lastSessionDate !== today) {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        if (lastSessionDate === yesterday.toDateString()) {
            localStorage.setItem('auraStreak', (streak + 1).toString());
        } else {
            localStorage.setItem('auraStreak', '1');
        }
        localStorage.setItem('auraLastSessionDate', today);
    }
    localStorage.setItem('auraSessionCount', (sessionCount + 1).toString());

    setTimeout(() => {
        navigate('/report');
    }, 4000); 
  };

  if (isAnalyzing) {
    return (
      <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-50 animate-fade-in">
        <svg className="animate-spin h-12 w-12 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="text-xl text-slate-200 mt-4">Analyzing your performance...</p>
      </div>
    );
  }


  return (
    <div className="flex flex-col animate-fade-in">
      <div aria-live="polite" className="sr-only">
        {feedbackMessage}
      </div>
      <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 flex flex-col gap-4">
            <div className="bg-slate-850 rounded-xl flex items-center justify-center border border-slate-700 overflow-hidden aspect-video">
              <VideoFeed videoRef={videoRef} />
            </div>
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 min-h-[100px]">
                <h3 className="text-sm font-semibold text-slate-400 mb-2">Live Transcript</h3>
                <p className="text-slate-200">{transcript || "Your live transcript will appear here once you start the session..."}</p>
            </div>
        </div>

        <div className="flex flex-col h-full">
          <div className="flex justify-between items-center mb-4">
             <h2 className="text-xl font-semibold text-slate-200">Live Feedback</h2>
             <button 
                onClick={() => setIsPerformanceMode(!isPerformanceMode)} 
                className="flex items-center space-x-2 text-sm text-slate-400 hover:text-white transition-colors" 
                aria-label={isPerformanceMode ? "Exit performance mode and show feedback" : "Enter performance mode to hide feedback"}
             >
                {isPerformanceMode ? <EyeIcon className="h-5 w-5" aria-hidden="true" /> : <EyeSlashIcon className="h-5 w-5" aria-hidden="true" />}
                <span>{isPerformanceMode ? "Show" : "Hide"}</span>
             </button>
          </div>
          
          {isPerformanceMode ? (
            <div className="flex-grow bg-slate-800 rounded-lg flex flex-col items-center justify-center text-center p-4">
                <EyeSlashIcon className="h-10 w-10 text-slate-500" aria-hidden="true" />
                <p className="mt-2 font-semibold text-slate-300">Performance Mode</p>
                <p className="text-sm text-slate-400">Feedback is hidden. Focus on your delivery. Your full report is waiting at the end.</p>
            </div>
          ) : (
            <div className="flex-grow flex flex-col space-y-4 justify-between">
                <Tooltip content="Measures your speaking speed in Words Per Minute (WPM). Aim for a conversational pace, typically between 140-160 WPM.">
                  <FeedbackStat 
                      icon={<ChartBarIcon className="h-6 w-6 text-teal-400" aria-hidden="true" />}
                      label="Pace"
                      value={liveWPM.toString()}
                      unit="WPM"
                      isFocused={focusArea === 'pacing'}
                      status={liveWPM > 180 || (liveWPM > 0 && liveWPM < 100) ? 'warning' : 'normal'}
                  />
                </Tooltip>
                <Tooltip content="Counts words like 'um', 'like', and 'you know'. Pausing to think is often more effective than using a filler word.">
                  <FeedbackStat
                      icon={<SpeakerWaveIcon className="h-6 w-6 text-rose-400" aria-hidden="true" />}
                      label="Filler Words"
                      value={liveFillerCount.toString()}
                      unit="count"
                      isFocused={focusArea === 'fillers'}
                  />
                </Tooltip>
                <Tooltip content="Analyzes your body language for confidence. Try to stay centered, keep your shoulders back, and avoid swaying.">
                  <FeedbackStat 
                      icon={<UserIcon className="h-6 w-6 text-sky-400" aria-hidden="true" />}
                      label="Posture"
                      value={postureFeedback}
                      unit=""
                      isFocused={focusArea === 'posture'}
                  />
                </Tooltip>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-6 py-4 flex items-center justify-center space-x-4">
        <button
          onClick={handleStartSession}
          disabled={isListening}
          className="flex items-center space-x-2 bg-green-600 text-white px-8 py-3 rounded-lg font-semibold text-lg hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
          aria-label="Start session"
        >
          <PlayIcon className="h-6 w-6" aria-hidden="true" />
          <span>Start</span>
        </button>
        <button
          onClick={handleStopSession}
          disabled={!isListening}
          className="flex items-center space-x-2 bg-red-600 text-white px-8 py-3 rounded-lg font-semibold text-lg hover:bg-red-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
          aria-label="Stop session and view report"
        >
          <StopIcon className="h-6 w-6" aria-hidden="true" />
          <span>Stop</span>
        </button>
      </div>
    </div>
  );
};

export default Session;