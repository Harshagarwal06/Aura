import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleGenAI, Type } from "@google/genai";
import { useAppContext } from '../context/AppContext';
import { ArrowUpTrayIcon, VideoCameraIcon, ChatBubbleBottomCenterTextIcon } from '@heroicons/react/24/solid';

const createAnalysisPrompt = (videoGoal: string) => `
You are an expert analyst AI named Aura. Your task is to conduct a comprehensive analysis of a user's uploaded video, considering its stated goal. The user has provided the following goal for their video: "${videoGoal}".

Please analyze both the video and audio tracks of this hypothetical video and generate a detailed report. Your report MUST be a single JSON object conforming to the schema provided. Do not include any markdown formatting or explanatory text outside the JSON.

Your analysis should be insightful, objective, and provide actionable recommendations.

The required JSON output structure is:
- executiveSummary: A one-paragraph overview of the video's content and your most important findings.
- audioAnalysis:
  - transcriptSummary: A summary of the main topics discussed. Do not provide a full transcript.
  - keyPoints: A bulleted list (array of strings) of the 3-5 most important ideas, decisions, or conclusions from the audio.
  - vocalDelivery: A brief analysis of the speaker's tone (e.g., confident, hesitant), pace, and clarity.
- videoAnalysis:
  - visualSummary: A description of the main subjects, setting, and key visual events.
  - bodyLanguageAndSentiment: An analysis of the primary subject's body language and expressions.
  - keyVisualMoments: An array of objects, each with a 'timestamp' (e.g., "[~0:30]") and 'description' of a significant visual event.
- overallInsightsAndRecommendations: An array of 3-5 actionable insights or recommendations for improvement, based on the video's stated goal.
`;

const responseSchema = {
    type: Type.OBJECT,
    properties: {
        executiveSummary: { type: Type.STRING },
        audioAnalysis: {
            type: Type.OBJECT,
            properties: {
                transcriptSummary: { type: Type.STRING },
                keyPoints: { type: Type.ARRAY, items: { type: Type.STRING } },
                vocalDelivery: { type: Type.STRING },
            }
        },
        videoAnalysis: {
            type: Type.OBJECT,
            properties: {
                visualSummary: { type: Type.STRING },
                bodyLanguageAndSentiment: { type: Type.STRING },
                keyVisualMoments: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            timestamp: { type: Type.STRING },
                            description: { type: Type.STRING },
                        }
                    }
                },
            }
        },
        overallInsightsAndRecommendations: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
        },
    }
};

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
    setError(null);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const prompt = createAnalysisPrompt(videoGoal);
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: responseSchema,
        },
      });
      const report = JSON.parse(response.text);
      setComprehensiveReport(report);
      navigate('/comprehensive-report');
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
        <svg className="animate-spin h-12 w-12 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="text-xl text-slate-200 mt-4">Aura is generating your report...</p>
        <p className="text-sm text-slate-400">This may take a moment.</p>
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