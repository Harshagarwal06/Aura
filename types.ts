
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

// Types for the Speaking Coach Video Analysis Feature
export interface SkillBreakdown {
  score: number;
  comments: string;
  wpm?: number;
  fillerWords?: number;
}

export interface ActionableHighlight {
  timestamp: string;
  type: 'Slouching' | 'Looking away' | 'Hand fidgeting' | 'Great presence';
  advice: string;
}

export interface VideoAnalysisReport {
  overallPresenceScore: number;
  bestTrait: string;
  needsImprovement: string;
  skillBreakdown: {
    posture: SkillBreakdown;
    eyeContact: SkillBreakdown;
    gestures: SkillBreakdown;
    calmHands: SkillBreakdown;
    smile: SkillBreakdown;
    openingPresence: SkillBreakdown;
    voice: SkillBreakdown;
  };
  actionableHighlights: ActionableHighlight[];
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