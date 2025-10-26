import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUpTrayIcon, VideoCameraIcon } from '@heroicons/react/24/solid';
import { useAppContext } from '../context/AppContext';
import { analyzeVideoFile } from '../analysis/videoAnalyzer';
import { VideoAnalysisReport, TimelineHighlight } from '../types';
import type { ComprehensiveReport } from '../types';

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

const formatHighlightTime = (highlight: TimelineHighlight) => {
  if (highlight.start === highlight.end) {
    return `${highlight.start.toFixed(1)}s`;
  }
  return `${highlight.start.toFixed(1)}s - ${highlight.end.toFixed(1)}s`;
};

const buildComprehensiveReport = (analysis: VideoAnalysisReport, goal: string): ComprehensiveReport => {
  const { presenceScore, bestTrait, needsImprovement, skills, highlights, voice } = analysis;
  const skillKeyMap: Record<string, keyof typeof skills> = {
    'Posture & Space': 'posture',
    'Eye Contact': 'eyeContact',
    'Gestures / Open Hands': 'gestures',
    'Calm Hands': 'calmHands',
    'Smile / Warmth': 'smile',
    'Opening Presence': 'openingPresence',
    Voice: 'voice',
  };
  const weakestKey = skillKeyMap[needsImprovement.name] ?? 'posture';

  const goalText = goal.trim() ? goal.trim() : 'video';

  const execSummary = `Your ${goalText} earned a ${presenceScore}/100 presence score. ${bestTrait.name} stood out as your strongest area, while ${needsImprovement.name.toLowerCase()} needs the most attention.`;

  const audioAnalysis = {
    transcriptSummary: `Delivery maintained an estimated pace of ${voice.paceWpm} WPM with ${voice.quietStart ? 'a quiet pause before the opening.' : 'an immediate start.'}`,
    keyPoints: [
      `Estimated vocal energy at ${(voice.energy ?? 0) > 0 ? Math.round((voice.energy ?? 0) * 100) : 0}% of the target range.`,
      voice.fillerPerMin > 0 ? `Approximate filler usage: ${voice.fillerPerMin}/min.` : 'No noticeable filler-word spikes detected in the audio energy.',
      `Voice active for ${voice.speakingRatio != null ? formatPercent(voice.speakingRatio) : 'a healthy share'} of the session timeline.`,
    ],
    vocalDelivery: `${voice.comment}${voice.quietStart ? ' Consider speaking right away to hook the audience.' : ''}`,
  };

  const videoAnalysis = {
    visualSummary: `Posture scored ${skills.posture.score}/100 with ${formatPercent(skills.posture.percentGood)} upright alignment; gestures landed at ${skills.gestures.score}/100 and were visible ${formatPercent(skills.gestures.percentGood)} of the time.`,
    bodyLanguageAndSentiment: `Eye contact rated ${skills.eyeContact.score}/100, calm hands ${skills.calmHands.score}/100, and smile warmth ${skills.smile.score}/100—together painting a ${skills.smile.score > 70 ? 'friendly' : 'neutral'} on-camera presence.`,
    keyVisualMoments: highlights.slice(0, 6).map((highlight) => ({
      timestamp: formatHighlightTime(highlight),
      description: `${highlight.tag}: ${highlight.advice}`,
    })),
  };

  const insights: string[] = [
    `Double down on ${bestTrait.name.toLowerCase()}—it is your signature strength right now.`,
    `Lift ${needsImprovement.name.toLowerCase()} by applying the coaching cues in the report (${skills[weakestKey]?.coachComment ?? 'focus on the fundamentals'}).`,
    skills.openingPresence.score < 80
      ? 'Rehearse the first five seconds until your posture, eye contact, and voice align from the start.'
      : 'Your opening presence is strong—keep leading with that confident stance.',
    voice.paceWpm > 180
      ? 'Dial back the pace slightly to give important ideas room to land.'
      : voice.paceWpm < 120
      ? 'Add a touch more pace to maintain energy.'
      : 'Maintain the conversational pace—it supports clarity.',
  ];

  return {
    executiveSummary: execSummary,
    audioAnalysis,
    videoAnalysis,
    overallInsightsAndRecommendations: insights,
  };
};

const ComprehensiveAnalysis: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoGoal, setVideoGoal] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const navigate = useNavigate();
  const { setComprehensiveReport, setVideoReport } = useAppContext();

  const handleFileChange = (files: FileList | null) => {
    if (files && files[0]) {
      if (files[0].type.startsWith('video/')) {
        setVideoFile(files[0]);
        setError(null);
      } else {
        setError('Please select a valid video file.');
        setVideoFile(null);
      }
    }
  };
  
  const handleDragEnter = (e: React.DragEvent<HTMLLabelElement>) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileChange(e.dataTransfer.files);
  };

  const handleAnalyze = async () => {
    if (!videoFile) {
      setError("Please select a video file first.");
      return;
    }
    if (!videoGoal.trim()) {
      setError("Please describe the goal of your video.");
      return;
    }
    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    try {
      const { report } = await analyzeVideoFile(videoFile, {
        onProgress: (value) => setProgress(value),
      });
      setVideoReport(report);
      const comprehensive = buildComprehensiveReport(report, videoGoal);
      setComprehensiveReport(comprehensive);
      navigate('/comprehensive-report');
    } catch (err) {
      console.error('Error analyzing video:', err);
      setError('Sorry, something went wrong during the analysis. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (isAnalyzing) {
    return (
      <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-50 animate-fade-in">
        <svg className="animate-spin h-12 w-12 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="text-xl text-slate-200 mt-4">Aura is generating your report...</p>
        <p className="text-sm text-slate-400 mb-4">This may take a moment.</p>
        <div className="w-64 h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 transition-all duration-300"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
        <p className="text-xs text-slate-500 mt-2">{Math.round(progress * 100)}% complete</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto text-center animate-fade-in-up">
      <h1 className="text-3xl font-bold text-slate-100 mb-2">Comprehensive Video Report</h1>
      <p className="text-slate-400 mb-8">Get AI-powered insights on any video. Just upload a file and describe its goal.</p>

      <div className="space-y-6">
        <div>
            <label 
                htmlFor="video-upload"
                className={`flex flex-col items-center justify-center w-full h-52 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer bg-slate-800 hover:bg-slate-700/50 transition-colors ${isDragging ? 'border-indigo-500' : ''}`}
                onDragEnter={handleDragEnter} onDragLeave={handleDragLeave} onDragOver={handleDragOver} onDrop={handleDrop}
            >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <ArrowUpTrayIcon className="w-10 h-10 mb-3 text-slate-400" />
                    <p className="mb-2 text-sm text-slate-400"><span className="font-semibold text-indigo-400">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-slate-500">MP4, MOV, or WEBM (max 500MB)</p>
                </div>
                <input id="video-upload" type="file" className="hidden" accept="video/mp4,video/quicktime,video/webm" onChange={(e) => handleFileChange(e.target.files)} />
            </label>
            {videoFile && (
                <div className="bg-slate-700/50 mt-4 p-3 rounded-lg text-left flex items-center space-x-3 animate-slide-in-right">
                    <VideoCameraIcon className="h-6 w-6 text-green-400 flex-shrink-0" />
                    <div>
                        <p className="text-sm font-medium text-slate-200">{videoFile.name}</p>
                    </div>
                </div>
            )}
        </div>
        
        <div className="text-left">
            <label htmlFor="video-goal" className="block text-sm font-medium text-slate-300 mb-2">
                What is the primary goal of this video?
            </label>
            <textarea
                id="video-goal"
                rows={3}
                value={videoGoal}
                onChange={(e) => setVideoGoal(e.target.value)}
                placeholder="e.g., A user testing session for a new app."
                className="w-full bg-slate-700 border-slate-600 rounded-md py-2 px-3 text-white placeholder-slate-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                required
            />
        </div>

        {error && <p className="text-rose-400 text-sm">{error}</p>}

        <button
          onClick={handleAnalyze}
          disabled={!videoFile || isAnalyzing || !videoGoal.trim()}
          className="w-full bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold text-lg hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
        >
          Generate Report
        </button>
      </div>
    </div>
  );
};

export default ComprehensiveAnalysis;