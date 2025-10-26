import React, { useEffect, useMemo } from 'react';
import ReportChart from '../components/ReportChart';
import { ChartDataPoint } from '../types';
import { StarIcon, AdjustmentsHorizontalIcon, QuestionMarkCircleIcon, DocumentTextIcon, CheckCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';
import { useAppContext } from '../context/AppContext';
import Tooltip from '../components/Tooltip';
import ProgressBar from '../components/ProgressBar';
import { NavLink } from 'react-router-dom';

const TranscriptViewer: React.FC<{ report: any }> = ({ report }) => {
    const highlightedTranscript = useMemo(() => {
        if (!report?.fullTranscript) return '';

        const fillerWords = report.fillerWords.map((fw: { word: string }) => fw.word.toLowerCase());
        const fillerSet = new Set(fillerWords);
        
        let processedText = report.fullTranscript;

        // Highlight repeated phrases
        report.repeatedPhrases.forEach((phrase: { phrase: string }) => {
            const regex = new RegExp(`\\b${phrase.phrase}\\b`, 'gi');
            processedText = processedText.replace(regex, (match: string) => `<u class="decoration-sky-400 decoration-2">${match}</u>`);
        });

        const words = processedText.split(' ');

        const html = words.map((word) => {
            const cleanWord = word.replace(/[.,!?]/g, '').toLowerCase();
            if (fillerSet.has(cleanWord)) {
                return `<strong class="text-rose-400">${word}</strong>`;
            }
            return word;
        }).join(' ');
        
        return html;
    }, [report]);

    return (
        <div className="text-slate-300 leading-relaxed max-h-60 overflow-y-auto p-4 bg-slate-850 rounded-lg">
            <p dangerouslySetInnerHTML={{ __html: highlightedTranscript }} />
        </div>
    );
};


const Report: React.FC = () => {
  const { sessionReport } = useAppContext();

  useEffect(() => {
    document.title = "Aura: AI Public Speaking Coach - Performance Report";
  }, []);

  if (!sessionReport) {
      return (
        <div className="text-center animate-fade-in-up">
            <h1 className="text-2xl font-bold text-slate-200">No report data found.</h1>
            <p className="text-slate-400 mt-2">Complete a session to see your performance analysis.</p>
            <NavLink to="/" className="mt-6 inline-block bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors">
                Start New Session
            </NavLink>
        </div>
      );
  }

  const { avgWPM, fillerWordCount, paceTimeline, repeatedPhrases, topicKeywords, postureScore } = sessionReport;

  const growthSuggestions = useMemo(() => {
    const tips: string[] = [];

    if (avgWPM > 180) {
      tips.push("You're speaking fast. Add half-second pauses at the ends of sentences to let ideas land.");
    } else if (avgWPM < 140) {
      tips.push(`Your pace averaged ${avgWPM} WPM. Add a little lift and tempo to avoid sounding subdued.`);
    } else {
      tips.push(`Hold your ${avgWPM} WPM pace but mark transitions with a deliberate pause or emphasis.`);
    }

    if (fillerWordCount > 5) {
      tips.push("You’re relying on filler words. Try pausing silently instead of saying 'um' or 'like'.");
    } else {
      tips.push(`Keep fillers low—you only used ${fillerWordCount}. Take a breath before key points to stay crisp.`);
    }

    if (postureScore < 70) {
      tips.push("Your posture score suggests you're leaning or collapsing forward. Keep shoulders open and level.");
    } else {
      tips.push(`Lock in that ${postureScore}% posture score by resetting your stance every minute.`);
    }

    return tips;
  }, [avgWPM, fillerWordCount, postureScore]);

  const hasTopicGuidance = topicKeywords.expected.length > 0;
  let topicMessage = '';
  let missingTopics: string[] = [];

  if (!hasTopicGuidance) {
    topicMessage = 'No topic guidance was set for this session.';
  } else if (topicKeywords.missing.length > 0) {
    topicMessage = 'You did not cover:';
    missingTopics = topicKeywords.missing;
  } else {
    topicMessage = 'Nice. You touched all the key talking points.';
  }

  return (
    <div className="animate-fade-in-up">
      <h1 className="text-3xl font-bold mb-6 text-slate-100">Your Session Report</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 hover-lift md:col-span-1">
            <h2 className="text-lg font-medium text-slate-300 mb-4">Key Metrics</h2>
            <div className="space-y-6">
                <div className="flex justify-between items-center">
                    <span className="text-slate-200 text-lg">Avg. Pace</span>
                    <span className="font-bold text-2xl text-teal-400">{avgWPM} <span className="text-base font-normal">WPM</span></span>
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-slate-200 text-lg">Filler Words</span>
                    <span className="font-bold text-2xl text-rose-400">{fillerWordCount}</span>
                </div>
                 <div className="flex justify-between items-center">
                    <span className="text-slate-200 text-lg">Posture Score</span>
                    <span className="font-bold text-2xl text-sky-400">{postureScore}%</span>
                </div>
            </div>
        </div>
        
        <div className="bg-slate-800 p-6 rounded-xl shadow-lg md:col-span-2 hover-lift">
          <h2 className="text-lg font-medium text-slate-300 mb-4">Pace (WPM) Timeline</h2>
          <div className="h-64">
            <ReportChart data={paceTimeline} />
          </div>
        </div>

        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 hover-lift md:col-span-3">
            <div className="flex items-center space-x-3 mb-3">
                <DocumentTextIcon className="h-8 w-8 text-indigo-400" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-200">Transcript Analysis</h2>
            </div>
            <TranscriptViewer report={sessionReport} />
        </div>
        
        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 hover-lift md:col-span-2">
            <div className="flex items-center space-x-3 mb-3">
                <ExclamationTriangleIcon className="h-8 w-8 text-amber-400" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-200">Growth Opportunities</h2>
            </div>
            {repeatedPhrases.length > 0 ? (
                <div>
                    <p className="text-lg font-semibold text-white">Repetitive Phrases</p>
                    <p className="text-slate-400 mb-3">Try to vary your language to keep your speech dynamic. You leaned on:</p>
                    <ul className="space-y-1">
                        {repeatedPhrases.slice(0, 3).map((p) => (
                            <li key={p.phrase} className="flex justify-between bg-slate-700/50 p-2 rounded-md">
                                <span className="italic text-slate-300">"{p.phrase}"</span>
                                <span className="font-mono text-amber-300">{p.count}×</span>
                            </li>
                        ))}
                    </ul>
                </div>
            ) : (
                <div className="space-y-3">
                    <p className="text-slate-300">Based on this run, focus on:</p>
                    <ul className="space-y-2">
                        {growthSuggestions
                            .filter(Boolean)
                            .slice(0, 3)
                            .map((tip, index) => (
                                <li key={index} className="bg-slate-700/40 px-3 py-2 rounded-md text-left text-slate-200">
                                    {tip}
                                </li>
                            ))}
                    </ul>
                </div>
            )}
        </div>

        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 hover-lift md:col-span-1">
             <div className="flex items-center space-x-3 mb-3">
                <CheckCircleIcon className="h-8 w-8 text-green-400" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-200">Topic Adherence</h2>
            </div>
            <div className="space-y-2">
                <div className="flex justify-between items-center text-slate-200 text-lg">
                    <span>Focus Score</span>
                </div>
                <div className="flex items-center space-x-3">
                    <ProgressBar progress={topicKeywords.topicScore} color="from-green-500 to-emerald-400" />
                    <span className="font-bold text-xl text-green-400">{topicKeywords.topicScore}%</span>
                </div>
                <p className="text-sm text-slate-300 pt-2">{topicMessage}</p>
                {missingTopics.length > 0 && (
                    <ul className="mt-2 text-xs text-rose-300 list-disc list-inside space-y-1">
                        {missingTopics.map((item) => (
                            <li key={item}>{item}</li>
                        ))}
                    </ul>
                )}
            </div>
        </div>

      </div>
    </div>
  );
};

export default Report;
