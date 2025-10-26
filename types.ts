
export type { VideoAnalysisReport, SkillDetail, TimelineHighlight, VoiceMetrics } from './analysis/bodyLanguage';

export interface ChartDataPoint {
  time: string;
  wpm: number;
}

export interface AnalysisReport {
  transcript: { t: number; text: string }[];
  fullTranscript: string;
  fillerWordCount: number;
  fillerWords: { t: number; word: string }[];
  avgWPM: number;
  paceTimeline: ChartDataPoint[];
  repeatedPhrases: { phrase: string; count: number }[];
  topicKeywords: {
    expected: string[];
    found: string[];
    missing: string[];
    topicScore: number;
  };
  postureScore: number;
}

// Types for the new Comprehensive Analysis Feature
export interface AudioAnalysis {
  transcriptSummary: string;
  keyPoints: string[];
  vocalDelivery: string;
}

export interface VideoAnalysis {
  visualSummary: string;
  bodyLanguageAndSentiment: string;
  keyVisualMoments: { timestamp: string; description: string }[];
}

export interface ComprehensiveReport {
  executiveSummary: string;
  audioAnalysis: AudioAnalysis;
  videoAnalysis: VideoAnalysis;
  overallInsightsAndRecommendations: string[];
}