import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUpTrayIcon, VideoCameraIcon } from '@heroicons/react/24/solid';
import { GoogleGenAI, Type } from '@google/genai';
import { useAppContext } from '../context/AppContext';
import type { ComprehensiveReport } from '../types';

const comprehensiveResponseSchema = {
  type: Type.OBJECT,
  required: ['executiveSummary', 'audioAnalysis', 'videoAnalysis', 'overallInsightsAndRecommendations'],
  properties: {
    executiveSummary: { type: Type.STRING },
    audioAnalysis: {
      type: Type.OBJECT,
      required: ['transcriptSummary', 'keyPoints', 'vocalDelivery'],
      properties: {
        transcriptSummary: { type: Type.STRING },
        keyPoints: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
        },
        vocalDelivery: { type: Type.STRING },
      },
    },
    videoAnalysis: {
      type: Type.OBJECT,
      required: ['visualSummary', 'bodyLanguageAndSentiment', 'keyVisualMoments'],
      properties: {
        visualSummary: { type: Type.STRING },
        bodyLanguageAndSentiment: { type: Type.STRING },
        keyVisualMoments: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            required: ['timestamp', 'description'],
            properties: {
              timestamp: { type: Type.STRING },
              description: { type: Type.STRING },
            },
          },
        },
      },
    },
    overallInsightsAndRecommendations: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
    },
  },
} as const;

async function fileToGenerativePart(file: File) {
  const data = await file.arrayBuffer();
  const base64Data = btoa(String.fromCharCode(...new Uint8Array(data)));
  return {
    inlineData: {
      data: base64Data,
      mimeType: file.type,
    },
  } as const;
}

const buildComprehensivePrompt = (goal: string) => `You are Aura, an AI public speaking coach. Analyze the uploaded video as if you ran our posture and voice pipeline:

- Posture is good when torso tilt is under 20°. Flag slouching moments when it exceeds that threshold.
- Eye contact is tracked via shoulder depth difference; looking away is when the absolute difference exceeds 0.20.
- Hands count as gesturing when at least one wrist rises above 60% of torso height AND the hands stay calm (jitter < 0.02). Mark fidgeting when jitter > 0.06 or wrists stay clasped.
- Smiling is detected when the mouth aspect ratio exceeds 0.35.
- Opening presence evaluates the first five seconds: good posture, facing forward, and voice launching promptly.
- Voice scoring considers pace (words per minute), filler words per minute, and projection/energy.

The speaker's goal for this video: "${goal}". Evaluate how well the delivery supports that goal, including whether the topic appears on track.

Provide:
1. An executive summary referencing the overall presence score implied by those metrics.
2. Audio analysis: summarize transcript patterns, list 3 concise bullet key points about pace/clarity/energy, and comment on vocal delivery.
3. Video analysis: describe posture/gestures/smile, overall sentiment, and list notable visual moments. Each moment should include a timestamp like "45.2s" and mention events such as slouching, looking away, fidgeting, strong opening presence, or genuine smiles.
4. Overall insights: at least three tailored recommendations tying back to the measured skills and the stated goal.

Return valid JSON that matches the provided schema exactly. Keep the tone constructive and specific.`;

const ComprehensiveAnalysis: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoGoal, setVideoGoal] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const navigate = useNavigate();
  const { setComprehensiveReport } = useAppContext();

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

  const handleDragEnter = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };
  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };
  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileChange(e.dataTransfer.files);
  };

  const handleAnalyze = async () => {
    if (!videoFile) {
      setError('Please select a video file first.');
      return;
    }
    if (!videoGoal.trim()) {
      setError('Please describe the goal of your video.');
      return;
    }

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      setError('Missing Gemini API key. Please set VITE_GEMINI_API_KEY.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const ai = new GoogleGenAI({ apiKey });
      const videoPart = await fileToGenerativePart(videoFile);
      const prompt = buildComprehensivePrompt(videoGoal.trim());
      const result = await ai.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: [
          {
            role: 'user',
            parts: [{ text: prompt }, videoPart],
          },
        ],
        config: {
          responseMimeType: 'application/json',
          responseSchema: comprehensiveResponseSchema,
        },
      });

      const rawText = result.response?.text() ?? result.text();
      let parsed: ComprehensiveReport;
      try {
        parsed = JSON.parse(rawText) as ComprehensiveReport;
      } catch (parseError) {
        console.error('Failed to parse Gemini comprehensive response', parseError, rawText);
        setError('AI returned invalid JSON. Please try again.');
        return;
      }

      setComprehensiveReport(parsed);
      navigate('/comprehensive-report');
    } catch (err) {
      console.error('Error generating comprehensive report:', err);
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
        <p className="text-xl text-slate-200 mt-4">Analyzing your video with Aura…</p>
        <p className="text-sm text-slate-400">Do not close this tab.</p>
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
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <ArrowUpTrayIcon className="w-10 h-10 mb-3 text-slate-400" />
              <p className="mb-2 text-sm text-slate-400">
                <span className="font-semibold text-indigo-400">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-slate-500">MP4, MOV, or WEBM (max 500MB)</p>
            </div>
            <input
              id="video-upload"
              type="file"
              className="hidden"
              accept="video/mp4,video/quicktime,video/webm"
              onChange={(e) => handleFileChange(e.target.files)}
            />
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
