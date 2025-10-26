import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUpTrayIcon, VideoCameraIcon } from '@heroicons/react/24/solid';
import { GoogleGenAI, Type } from '@google/genai';
import { useAppContext } from '../context/AppContext';
import type { VideoAnalysisReport, TimelineHighlight } from '../types';

const analysisPrompt = `You are Aura, an AI speaking coach. Pretend you just analyzed a public speaking video frame by frame using the following rules that mirror our Python reference pipeline:

1. Posture: calculate torso_upright_angle_deg from shoulders and hips. Posture is good if tilt < 20° from vertical.
2. Eye contact: calculate facing_depth_diff from shoulder Z values. Eye contact is good when |left-right shoulder depth| < 0.20.
3. Hands and gestures: detect hands above 60% of torso height. Gestures are credited only when at least one hand is that high AND the hands are calm.
4. Calm hands / fidget: track per-frame wrist motion with camera-motion compensation. calm if jitter < 0.02. fidget if jitter > 0.06. Intermediate values lean calm when closer to 0.02.
5. Smile: use mouth aspect ratio (upper/lower lip distance divided by mouth width). Smile is on when ratio > 0.35.
6. Opening presence: score the first five seconds using posture, facing, and whether the voice started promptly (quiet openings score 0 for voice).
7. Voice: consider pace_wpm, filler_per_min, projection/energy. Translate those metrics into a 0-100 voice score.

Convert the frame metrics into percentages (percent of frames marked good) for posture, eye contact, gestures, calm hands, smile, and opening presence. Voice score stays on its 0-100 scale.

Then compute presence_score as:
0.18 * posture_score +
0.25 * eye_contact_score +
0.25 * voice_score +
0.12 * gesture_score +
0.05 * calm_hands_score +
0.05 * smile_score +
0.10 * opening_presence_score.

Identify the highest-scoring skill as the best trait and the lowest-scoring skill as the needs-improvement trait. Use these trait labels exactly: "Posture & Space", "Eye Contact", "Gestures / Open Hands", "Calm Hands", "Smile / Warmth", "Opening Presence", "Voice".

Create actionable highlights whenever a sequence of frames shows: slouching (posture bad), looking away (eye contact bad), hand fidgeting (hands jittering while close together), or great presence (posture + facing + calm + gesture all good). Each highlight needs a timestamp in seconds, one of those four tags, and concise coaching advice.

Return a JSON object that strictly follows the provided schema. Keep comments short, direct, and coaching-oriented.`;

const responseSchema = {
  type: Type.OBJECT,
  required: ['overallPresenceScore', 'bestTrait', 'needsImprovement', 'skillBreakdown', 'actionableHighlights'],
  properties: {
    overallPresenceScore: { type: Type.NUMBER },
    bestTrait: { type: Type.STRING },
    needsImprovement: { type: Type.STRING },
    skillBreakdown: {
      type: Type.OBJECT,
      required: ['posture', 'eyeContact', 'gestures', 'calmHands', 'smile', 'openingPresence', 'voice'],
      properties: {
        posture: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        eyeContact: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        gestures: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        calmHands: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        smile: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        openingPresence: {
          type: Type.OBJECT,
          required: ['score', 'comments'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
          },
        },
        voice: {
          type: Type.OBJECT,
          required: ['score', 'comments', 'wpm', 'fillerWords'],
          properties: {
            score: { type: Type.NUMBER },
            comments: { type: Type.STRING },
            wpm: { type: Type.NUMBER },
            fillerWords: { type: Type.NUMBER },
          },
        },
      },
    },
    actionableHighlights: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        required: ['timestamp', 'type', 'advice'],
        properties: {
          timestamp: { type: Type.NUMBER },
          type: { type: Type.STRING },
          advice: { type: Type.STRING },
        },
      },
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

type SkillBreakdownResponse = {
  score: number;
  comments: string;
  wpm?: number;
  fillerWords?: number;
};

type GeminiAnalysisResponse = {
  overallPresenceScore: number;
  bestTrait: string;
  needsImprovement: string;
  skillBreakdown: Record<string, SkillBreakdownResponse> & {
    posture: SkillBreakdownResponse;
    eyeContact: SkillBreakdownResponse;
    gestures: SkillBreakdownResponse;
    calmHands: SkillBreakdownResponse;
    smile: SkillBreakdownResponse;
    openingPresence: SkillBreakdownResponse;
    voice: Required<SkillBreakdownResponse>;
  };
  actionableHighlights: Array<{ timestamp: number; type: string; advice: string }>;
};

const traitLabelToSkillKey: Record<string, keyof GeminiAnalysisResponse['skillBreakdown']> = {
  'Posture & Space': 'posture',
  'Eye Contact': 'eyeContact',
  'Gestures / Open Hands': 'gestures',
  'Calm Hands': 'calmHands',
  'Smile / Warmth': 'smile',
  'Opening Presence': 'openingPresence',
  Voice: 'voice',
};

const highlightTagMap: Record<string, TimelineHighlight['tag']> = {
  'Slouching': 'Slouching',
  'Looking away': 'Looking away',
  'Hand fidgeting': 'Hand fidgeting',
  'Great presence': 'Great presence',
};

const skillMeta = {
  posture: {
    label: 'Stand tall and take up space',
    why: 'Strong posture = confidence signal.',
  },
  eyeContact: {
    label: 'Maintain eye contact',
    why: 'Facing the audience makes you sound certain.',
  },
  gestures: {
    label: 'Talk with your hands / Use open gestures',
    why: 'Open hands make you look trustworthy and engaged.',
  },
  calmHands: {
    label: 'Eliminate nervous gestures',
    why: 'Fidgeting reads as stress.',
  },
  smile: {
    label: 'Smile',
    why: 'Warmth builds trust.',
  },
  openingPresence: {
    label: 'Prime the pump',
    why: 'Your first five seconds decide if people lean in.',
  },
  voice: {
    label: 'Get that voice down',
    why: 'Clear confident delivery makes people trust you.',
  },
} as const;

function clampScore(value: number) {
  if (Number.isFinite(value)) {
    return Math.min(100, Math.max(0, Math.round(value)));
  }
  return 0;
}

function percentFromScore(score: number) {
  if (!Number.isFinite(score)) {
    return 0;
  }
  return Math.min(1, Math.max(0, score / 100));
}

function toVideoReport(data: GeminiAnalysisResponse): VideoAnalysisReport {
  const highlights: TimelineHighlight[] = data.actionableHighlights
    .map((highlight) => {
      const tag = highlightTagMap[highlight.type];
      if (!tag) {
        return null;
      }
      const timestamp = Number(highlight.timestamp) || 0;
      return {
        start: Number(timestamp.toFixed(2)),
        end: Number(timestamp.toFixed(2)),
        tag,
        advice: highlight.advice,
      } satisfies TimelineHighlight;
    })
    .filter((item): item is TimelineHighlight => Boolean(item));

  const pickMoments = (tags: TimelineHighlight['tag'][]): TimelineHighlight[] =>
    highlights.filter((h) => tags.includes(h.tag));

  const skills: VideoAnalysisReport['skills'] = {
    posture: {
      score: clampScore(data.skillBreakdown.posture.score),
      percentGood: percentFromScore(data.skillBreakdown.posture.score),
      label: skillMeta.posture.label,
      whyItMatters: skillMeta.posture.why,
      coachComment: data.skillBreakdown.posture.comments,
      moments: pickMoments(['Slouching', 'Great presence']),
    },
    eyeContact: {
      score: clampScore(data.skillBreakdown.eyeContact.score),
      percentGood: percentFromScore(data.skillBreakdown.eyeContact.score),
      label: skillMeta.eyeContact.label,
      whyItMatters: skillMeta.eyeContact.why,
      coachComment: data.skillBreakdown.eyeContact.comments,
      moments: pickMoments(['Looking away', 'Great presence']),
    },
    gestures: {
      score: clampScore(data.skillBreakdown.gestures.score),
      percentGood: percentFromScore(data.skillBreakdown.gestures.score),
      label: skillMeta.gestures.label,
      whyItMatters: skillMeta.gestures.why,
      coachComment: data.skillBreakdown.gestures.comments,
      moments: pickMoments(['Great presence']),
    },
    calmHands: {
      score: clampScore(data.skillBreakdown.calmHands.score),
      percentGood: percentFromScore(data.skillBreakdown.calmHands.score),
      label: skillMeta.calmHands.label,
      whyItMatters: skillMeta.calmHands.why,
      coachComment: data.skillBreakdown.calmHands.comments,
      moments: pickMoments(['Hand fidgeting']),
    },
    smile: {
      score: clampScore(data.skillBreakdown.smile.score),
      percentGood: percentFromScore(data.skillBreakdown.smile.score),
      label: skillMeta.smile.label,
      whyItMatters: skillMeta.smile.why,
      coachComment: data.skillBreakdown.smile.comments,
      moments: [],
    },
    openingPresence: {
      score: clampScore(data.skillBreakdown.openingPresence.score),
      percentGood: percentFromScore(data.skillBreakdown.openingPresence.score),
      label: skillMeta.openingPresence.label,
      whyItMatters: skillMeta.openingPresence.why,
      coachComment: data.skillBreakdown.openingPresence.comments,
      moments: [],
      windowSeconds: [0, 5],
    },
    voice: {
      score: clampScore(data.skillBreakdown.voice.score),
      percentGood: percentFromScore(data.skillBreakdown.voice.score),
      label: skillMeta.voice.label,
      whyItMatters: skillMeta.voice.why,
      coachComment: data.skillBreakdown.voice.comments,
      moments: [],
      paceWpm: Math.round(data.skillBreakdown.voice.wpm ?? 0),
      fillerPerMin: data.skillBreakdown.voice.fillerWords ?? 0,
    },
  };

  const bestSkillKey = traitLabelToSkillKey[data.bestTrait] ?? 'posture';
  const needsSkillKey = traitLabelToSkillKey[data.needsImprovement] ?? 'posture';

  const emptySeries: VideoAnalysisReport['timeSeries'] = {
    frameCount: 0,
    durationSec: 0,
    fps: 0,
    t: [],
    posture: [],
    facing: [],
    gesture: [],
    calm: [],
    smile: [],
    torsoAngle: [],
    depthDiff: [],
    jitter: [],
    mouthRatio: [],
    handsRaised: [],
    visible: [],
    fidget: [],
    format: {
      description: 'Time-series data not returned by the Gemini summary.',
      note: 'This report focuses on skill scores and highlights only.',
      arrays: [
        't',
        'posture',
        'facing',
        'gesture',
        'calm',
        'smile',
        'torsoAngle',
        'depthDiff',
        'jitter',
        'mouthRatio',
        'handsRaised',
        'visible',
        'fidget',
      ],
    },
  };

  return {
    presenceScore: clampScore(data.overallPresenceScore),
    bestTrait: {
      name: data.bestTrait,
      score: clampScore(data.skillBreakdown[bestSkillKey]?.score ?? data.overallPresenceScore),
      summary: '',
    },
    needsImprovement: {
      name: data.needsImprovement,
      score: clampScore(data.skillBreakdown[needsSkillKey]?.score ?? data.overallPresenceScore),
      summary: '',
    },
    skills,
    highlights,
    timeSeries: emptySeries,
    voice: {
      voiceScore: clampScore(data.skillBreakdown.voice.score),
      paceWpm: Math.round(data.skillBreakdown.voice.wpm ?? 0),
      fillerPerMin: data.skillBreakdown.voice.fillerWords ?? 0,
      comment: data.skillBreakdown.voice.comments,
      quietStart: false,
    },
  } satisfies VideoAnalysisReport;
}

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
      setError('Please select a video file first.');
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
      const result = await ai.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: [
          {
            role: 'user',
            parts: [{ text: analysisPrompt }, videoPart],
          },
        ],
        config: {
          responseMimeType: 'application/json',
          responseSchema,
        },
      });

      const rawText = result.response?.text() ?? result.text();
      let parsed: GeminiAnalysisResponse;
      try {
        parsed = JSON.parse(rawText) as GeminiAnalysisResponse;
      } catch (parseError) {
        console.error('Failed to parse Gemini response', parseError, rawText);
        setError('AI returned invalid JSON. Please try again.');
        return;
      }

      const report = toVideoReport(parsed);
      setVideoReport(report);
      navigate('/video-report');
    } catch (err) {
      console.error('Error analyzing video with Gemini:', err);
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
