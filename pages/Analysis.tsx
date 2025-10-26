import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleGenAI, Type } from "@google/genai";
import { useAppContext } from '../context/AppContext';
import { ArrowUpTrayIcon, VideoCameraIcon } from '@heroicons/react/24/solid';

const analysisPrompt = `
You are an expert public speaking coach AI named Aura. You must analyze a hypothetical user's uploaded video presentation based on a very specific set of rules and thresholds. Your analysis must be strict and follow the logic provided.

Your Analysis Rules:
1.  **Posture:**
    *   Threshold: POSTURE_TILT_BAD_DEG = 20.0
    *   Logic: Analyze torso uprightness. "Slouching" (bad) is >20 degrees from vertical. Score is % of time with good posture.
2.  **Eye Contact (Facing Forward):**
    *   Threshold: FACING_DEPTH_DIFF_BAD = 0.20
    *   Logic: Analyze if the user is facing the camera via shoulder depth difference. If >0.20, it's "Looking away" (bad). Score is % of time facing forward.
3.  **Calm Hands:**
    *   Thresholds: FIDGET_MOTION_CALM = 0.02, FIDGET_MOTION_FIDGET = 0.06
    *   Logic: Analyze hand movements for "jitter." If motion >0.06, it's "Hand fidgeting" (bad). If <0.02, it's "Calm" (good). Score is % of time hands are calm.
4.  **Gestures:**
    *   Threshold: GESTURE_MIN_HEIGHT_FRAC = 0.6
    *   Logic: "Good" gestures are when hands are visible, raised to at least 60% of torso height (chest level), and are calm (not fidgeting). Score based on this.
5.  **Smile / Warmth:**
    *   Threshold: SMILE_MOUTH_RATIO_SMILE = 0.35
    *   Logic: A smile is detected when mouth's "openness-to-width" ratio is >0.35. Score is % of time appearing warm and friendly.
6.  **Opening Presence:**
    *   Threshold: OPENING_WINDOW_SEC = 5.0
    *   Logic: Special score for the first 5 seconds, combining Posture, Eye Contact, and Voice (no long silences).
7.  **Voice (Audio Analysis):**
    *   Logic: Score based on projection, pace (aim for 150-170 WPM), and low filler word count.

Required Output Format:
You MUST return a single JSON object. Do not include any markdown formatting. The JSON must conform to the provided schema. The "comments" fields should be insightful and provide specific, actionable coaching advice. Generate a realistic and varied report for a hypothetical speaker who has some strengths but also clear areas for improvement.

The overall presence score is calculated with this weighted formula:
0.18 * Posture_Score + 0.25 * Eye_Contact_Score + 0.25 * Voice_Score + 0.12 * Gestures_Score + 0.05 * Calm_Hands_Score + 0.05 * Smile_Score + 0.10 * Opening_Presence_Score
`;

const responseSchema = {
    type: Type.OBJECT,
    properties: {
        overallPresenceScore: { type: Type.NUMBER },
        bestTrait: { type: Type.STRING },
        needsImprovement: { type: Type.STRING },
        skillBreakdown: {
            type: Type.OBJECT,
            properties: {
                posture: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                eyeContact: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                gestures: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                calmHands: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                smile: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                openingPresence: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING } } },
                voice: { type: Type.OBJECT, properties: { score: { type: Type.NUMBER }, comments: { type: Type.STRING }, wpm: { type: Type.NUMBER }, fillerWords: { type: Type.NUMBER } } },
            }
        },
        actionableHighlights: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    timestamp: { type: Type.STRING },
                    type: { type: Type.STRING },
                    advice: { type: Type.STRING },
                }
            }
        },
    }
};


const Analysis: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const navigate = useNavigate();
  const { setVideoReport } = useAppContext();

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
        setError("Please select a video file first.");
        return;
    }
    setIsAnalyzing(true);
    setError(null);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: analysisPrompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: responseSchema,
        },
      });

      const report = JSON.parse(response.text);
      setVideoReport(report);
      navigate('/video-report');

    } catch (err) {
      console.error("Error analyzing video:", err);
      setError("Sorry, something went wrong during the analysis. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  if (isAnalyzing) {
    return (
      <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-50 animate-fade-in">
        <svg className="animate-spin h-12 w-12 text-indigo-400" xmlns="http://www.w.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="text-xl text-slate-200 mt-4">Aura is analyzing your performance...</p>
        <p className="text-sm text-slate-400">This can take a moment.</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto text-center animate-fade-in-up">
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Video Performance Analysis</h1>
        <p className="text-slate-400 mb-8">Upload a recorded speech and get a detailed breakdown of your public speaking skills.</p>
        
        <div className="space-y-6">
            <label 
                htmlFor="video-upload"
                className={`flex flex-col items-center justify-center w-full h-64 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer bg-slate-800 hover:bg-slate-700/50 transition-colors ${isDragging ? 'border-indigo-500' : ''}`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <ArrowUpTrayIcon className="w-10 h-10 mb-3 text-slate-400" />
                    <p className="mb-2 text-sm text-slate-400"><span className="font-semibold text-indigo-400">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-slate-500">MP4, MOV, or WEBM (max 500MB)</p>
                </div>
                <input id="video-upload" type="file" className="hidden" accept="video/mp4,video/quicktime,video/webm" onChange={(e) => handleFileChange(e.target.files)} />
            </label>

            {videoFile && (
                <div className="bg-slate-700/50 p-3 rounded-lg text-left flex items-center space-x-3 animate-slide-in-right">
                    <VideoCameraIcon className="h-6 w-6 text-green-400 flex-shrink-0" />
                    <div>
                        <p className="text-sm font-medium text-slate-200">{videoFile.name}</p>
                        <p className="text-xs text-slate-400">{(videoFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                </div>
            )}
            
            {error && <p className="text-rose-400 text-sm">{error}</p>}
            
            <button
                onClick={handleAnalyze}
                disabled={!videoFile || isAnalyzing}
                className="w-full bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold text-lg hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
            >
                Analyze My Performance
            </button>
        </div>
    </div>
  );
};

export default Analysis;
