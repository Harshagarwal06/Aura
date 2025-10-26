import React, { createContext, useState, useContext, ReactNode } from 'react';
import { AnalysisReport, VideoAnalysisReport, ComprehensiveReport } from '../types';

export type FocusArea = 'pacing' | 'fillers' | 'posture' | null;

interface AppContextType {
  focusArea: FocusArea;
  setFocusArea: (focus: FocusArea) => void;
  topicKeywords: string[];
  setTopicKeywords: (keywords: string[]) => void;
  sessionReport: AnalysisReport | null;
  setSessionReport: (report: AnalysisReport | null) => void;
  videoReport: VideoAnalysisReport | null;
  setVideoReport: (report: VideoAnalysisReport | null) => void;
  comprehensiveReport: ComprehensiveReport | null;
  setComprehensiveReport: (report: ComprehensiveReport | null) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [focusArea, setFocusAreaState] = useState<FocusArea>(() => {
    try {
      const item = window.localStorage.getItem('auraFocusArea');
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error(error);
      return null;
    }
  });

  const [topicKeywords, setTopicKeywords] = useState<string[]>([]);
  const [sessionReport, setSessionReport] = useState<AnalysisReport | null>(null);
  const [videoReport, setVideoReport] = useState<VideoAnalysisReport | null>(null);
  const [comprehensiveReport, setComprehensiveReport] = useState<ComprehensiveReport | null>(null);


  const setFocusArea = (focus: FocusArea) => {
    try {
      setFocusAreaState(focus);
      window.localStorage.setItem('auraFocusArea', JSON.stringify(focus));
    } catch (error) {
      console.error(error);
    }
  };
  
  return (
    <AppContext.Provider value={{ 
      focusArea, 
      setFocusArea,
      topicKeywords,
      setTopicKeywords,
      sessionReport,
      setSessionReport,
      videoReport,
      setVideoReport,
      comprehensiveReport,
      setComprehensiveReport
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};