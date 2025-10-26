import React, { useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import {
  ClipboardDocumentCheckIcon, SpeakerWaveIcon, VideoCameraIcon, LightBulbIcon
} from '@heroicons/react/24/solid';

const ReportCard: React.FC<{ icon: React.ReactNode; title: string; children: React.ReactNode }> = ({ icon, title, children }) => (
    <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 hover-lift">
        <div className="flex items-center space-x-3 mb-4">
            {icon}
            <h2 className="text-xl font-bold text-slate-200">{title}</h2>
        </div>
        <div className="space-y-4 text-slate-300 leading-relaxed">
            {children}
        </div>
    </div>
);

const ComprehensiveReport: React.FC = () => {
  const { comprehensiveReport } = useAppContext();

  useEffect(() => {
    document.title = "Aura Coach - Comprehensive Report";
  }, []);

  if (!comprehensiveReport) {
    return (
      <div className="text-center animate-fade-in-up">
        <ClipboardDocumentCheckIcon className="h-16 w-16 mx-auto text-slate-600" />
        <h1 className="text-2xl font-bold text-slate-200 mt-4">No report data found.</h1>
        <p className="text-slate-400 mt-2">Analyze a video to generate a comprehensive report.</p>
        <NavLink to="/comprehensive-analysis" className="mt-6 inline-block bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors">
          Analyze a New Video
        </NavLink>
      </div>
    );
  }

  const { executiveSummary, audioAnalysis, videoAnalysis, overallInsightsAndRecommendations } = comprehensiveReport;

  return (
    <div className="space-y-8 animate-fade-in-up">
      <h1 className="text-3xl font-bold text-slate-100 text-center">Comprehensive Analysis Report</h1>

      <ReportCard icon={<ClipboardDocumentCheckIcon className="h-8 w-8 text-indigo-400" />} title="Executive Summary">
        <p>{executiveSummary}</p>
      </ReportCard>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ReportCard icon={<SpeakerWaveIcon className="h-8 w-8 text-teal-400" />} title="Audio Analysis">
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Transcript Summary</h3>
                <p>{audioAnalysis.transcriptSummary}</p>
            </div>
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Key Points</h3>
                <ul className="list-disc list-inside space-y-1">
                    {audioAnalysis.keyPoints.map((point, i) => <li key={i}>{point}</li>)}
                </ul>
            </div>
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Vocal Delivery</h3>
                <p>{audioAnalysis.vocalDelivery}</p>
            </div>
        </ReportCard>

        <ReportCard icon={<VideoCameraIcon className="h-8 w-8 text-rose-400" />} title="Video Analysis">
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Visual Summary</h3>
                <p>{videoAnalysis.visualSummary}</p>
            </div>
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Body Language & Sentiment</h3>
                <p>{videoAnalysis.bodyLanguageAndSentiment}</p>
            </div>
            <div>
                <h3 className="font-semibold text-slate-100 mb-2">Key Visual Moments</h3>
                <ul className="space-y-2">
                    {videoAnalysis.keyVisualMoments.map((moment, i) => (
                        <li key={i} className="flex items-start">
                            <span className="font-mono text-xs text-indigo-400 mr-2 mt-1 flex-shrink-0">{moment.timestamp}</span>
                            <span>{moment.description}</span>
                        </li>
                    ))}
                </ul>
            </div>
        </ReportCard>
      </div>

      <ReportCard icon={<LightBulbIcon className="h-8 w-8 text-amber-400" />} title="Overall Insights & Recommendations">
        <ul className="list-disc list-inside space-y-2">
            {overallInsightsAndRecommendations.map((insight, i) => <li key={i}>{insight}</li>)}
        </ul>
      </ReportCard>
    </div>
  );
};

export default ComprehensiveReport;