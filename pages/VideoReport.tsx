import React, { useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import { VideoAnalysisReport, SkillDetail, TimelineHighlight } from '../types';
import {
  UserIcon, EyeIcon, HandThumbUpIcon, SparklesIcon, FaceSmileIcon, RocketLaunchIcon, SpeakerWaveIcon, StarIcon, LightBulbIcon, ClipboardDocumentCheckIcon, CheckCircleIcon, ExclamationTriangleIcon, ChatBubbleBottomCenterTextIcon
} from '@heroicons/react/24/solid';

const skillIcons: { [key in keyof VideoAnalysisReport['skills']]: React.ReactNode } = {
  posture: <UserIcon className="h-6 w-6 text-sky-400" />,
  eyeContact: <EyeIcon className="h-6 w-6 text-violet-400" />,
  gestures: <HandThumbUpIcon className="h-6 w-6 text-amber-400" />,
  calmHands: <SparklesIcon className="h-6 w-6 text-rose-400" />,
  smile: <FaceSmileIcon className="h-6 w-6 text-yellow-400" />,
  openingPresence: <RocketLaunchIcon className="h-6 w-6 text-green-400" />,
  voice: <SpeakerWaveIcon className="h-6 w-6 text-teal-400" />,
};

const ScoreGauge: React.FC<{ score: number }> = ({ score }) => {
  const circumference = 2 * Math.PI * 45; // 2 * pi * r
  const offset = circumference - (score / 100) * circumference;

  const getColor = (s: number) => {
    if (s > 80) return 'text-green-400';
    if (s > 60) return 'text-yellow-400';
    return 'text-rose-400';
  }

  return (
    <div className="relative flex items-center justify-center w-48 h-48">
      <svg className="absolute w-full h-full" viewBox="0 0 100 100">
        <circle
          className="text-slate-700"
          stroke="currentColor"
          strokeWidth="8"
          cx="50"
          cy="50"
          r="45"
          fill="transparent"
        />
        <circle
          className={`${getColor(score)} transition-all duration-1000 ease-out`}
          stroke="currentColor"
          strokeWidth="8"
          cx="50"
          cy="50"
          r="45"
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
        />
      </svg>
      <div className="text-center">
        <span className={`text-5xl font-bold ${getColor(score)}`}>{score}</span>
        <p className="text-slate-400 text-sm">Overall</p>
      </div>
    </div>
  );
}

const SkillCard: React.FC<{ name: keyof VideoAnalysisReport['skills']; data: SkillDetail }> = ({ name, data }) => {
  const icon = skillIcons[name];
  const percentLabel =
    data.percentGood !== undefined && data.percentGood !== null
      ? `${Math.round(data.percentGood * 100)}% positive`
      : null;
  return (
    <div className="bg-slate-800 p-5 rounded-xl shadow-lg border border-slate-700 hover-lift">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          {icon}
          <h3 className="text-lg font-semibold text-slate-200">{data.label}</h3>
        </div>
        <span className="font-bold text-xl text-slate-300">{data.score}/100</span>
      </div>
      {percentLabel && <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">{percentLabel}</p>}
      <p className="text-sm text-slate-400 leading-relaxed">{data.coachComment}</p>
      {name === 'voice' && (data.paceWpm || data.fillerPerMin !== undefined) && (
        <div className="mt-3 text-xs flex flex-wrap gap-4 text-slate-300">
          {data.paceWpm ? (
            <span>
              Avg Pace: <strong className="text-teal-400">{data.paceWpm} WPM</strong>
            </span>
          ) : null}
          {data.fillerPerMin !== undefined ? (
            <span>
              Filler Words: <strong className="text-rose-400">{Math.round(data.fillerPerMin)}</strong>/min
            </span>
          ) : null}
        </div>
      )}
      {data.windowSeconds && (
        <p className="mt-3 text-xs text-slate-500">
          Focus on {data.windowSeconds[0]}s to {data.windowSeconds[1]}s
        </p>
      )}
      {data.moments && data.moments.length > 0 && (
        <div className="mt-3 space-y-2">
          {data.moments.slice(0, 2).map((moment, index) => (
            <div key={index} className="bg-slate-900/40 px-3 py-2 rounded-md">
              <p className="text-xs font-mono text-indigo-400">
                {moment.start === moment.end
                  ? `${moment.start.toFixed(1)}s`
                  : `${moment.start.toFixed(1)}s - ${moment.end.toFixed(1)}s`}
              </p>
              <p className="text-xs text-slate-300">{moment.tag}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const traitSkillKeyMap: Record<string, keyof VideoAnalysisReport['skills']> = {
  'Posture & Space': 'posture',
  'Eye Contact': 'eyeContact',
  'Gestures / Open Hands': 'gestures',
  'Calm Hands': 'calmHands',
  'Smile / Warmth': 'smile',
  'Opening Presence': 'openingPresence',
  Voice: 'voice',
};

const formatHighlightRange = (highlight: TimelineHighlight) =>
  highlight.start === highlight.end
    ? `${highlight.start.toFixed(1)}s`
    : `${highlight.start.toFixed(1)}s - ${highlight.end.toFixed(1)}s`;


const VideoReport: React.FC = () => {
  const { videoReport } = useAppContext();

  useEffect(() => {
    document.title = "Aura Coach - Video Analysis Report";
  }, []);

  if (!videoReport) {
    return (
      <div className="text-center animate-fade-in-up">
        <ClipboardDocumentCheckIcon className="h-16 w-16 mx-auto text-slate-600" />
        <h1 className="text-2xl font-bold text-slate-200 mt-4">No video report found.</h1>
        <p className="text-slate-400 mt-2">Upload a video to get your performance analysis.</p>
        <NavLink to="/analysis" className="mt-6 inline-block bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors">
          Analyze a Video
        </NavLink>
      </div>
    );
  }

  const { presenceScore, bestTrait, needsImprovement, skills, highlights } = videoReport;
  const bestSkillKey = traitSkillKeyMap[bestTrait.name];
  const bestSkill = bestSkillKey ? skills[bestSkillKey] : undefined;
  const needsSkillKey = traitSkillKeyMap[needsImprovement.name];
  const needsSkill = needsSkillKey ? skills[needsSkillKey] : undefined;

  return (
    <div className="space-y-8 animate-fade-in-up">
        <h1 className="text-3xl font-bold text-slate-100 text-center">Video Analysis Report</h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="flex justify-center md:col-span-1">
                <ScoreGauge score={presenceScore} />
            </div>
            <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">
                 <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 hover-lift">
                    <div className="flex items-center space-x-3 mb-2">
                        <StarIcon className="h-8 w-8 text-green-400" />
                        <h2 className="text-lg font-semibold text-slate-300">Best Trait</h2>
                    </div>
                    <p className="text-2xl font-bold text-white">{bestTrait.name} ({bestTrait.score})</p>
                    {bestSkill && (
                      <p className="text-sm text-slate-400 mt-2">{bestSkill.coachComment}</p>
                    )}
                </div>
                 <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 hover-lift">
                    <div className="flex items-center space-x-3 mb-2">
                        <LightBulbIcon className="h-8 w-8 text-amber-400" />
                        <h2 className="text-lg font-semibold text-slate-300">Needs Improvement</h2>
                    </div>
                    <p className="text-2xl font-bold text-white">{needsImprovement.name} ({needsImprovement.score})</p>
                    {needsSkill && (
                      <p className="text-sm text-slate-400 mt-2">{needsSkill.coachComment}</p>
                    )}
                </div>
            </div>
        </div>

        <div>
            <h2 className="text-2xl font-bold text-slate-200 mb-4 text-center md:text-left">Detailed Skill Breakdown</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {Object.entries(skills).map(([key, value]) => (
                    <SkillCard key={key} name={key as keyof VideoAnalysisReport['skills']} data={value as SkillDetail} />
                ))}
            </div>
        </div>

        <div>
            <h2 className="text-2xl font-bold text-slate-200 mb-4 text-center md:text-left">Actionable Highlights</h2>
            <div className="space-y-4">
                {highlights.map((item, index) => (
                    <div key={index} className="bg-slate-800/70 p-4 rounded-lg flex items-start space-x-4 border border-slate-700/50">
                        <div className="flex-shrink-0 mt-1">
                            {item.tag === 'Great presence' ? <CheckCircleIcon className="h-6 w-6 text-green-400" /> : <ExclamationTriangleIcon className="h-6 w-6 text-yellow-400" />}
                        </div>
                        <div>
                            <p className="font-mono text-sm text-indigo-400">{formatHighlightRange(item)}</p>
                            <p className="font-semibold text-slate-200">{item.tag}</p>
                            <p className="text-sm text-slate-400">{item.advice}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>

    </div>
  );
};

export default VideoReport;
