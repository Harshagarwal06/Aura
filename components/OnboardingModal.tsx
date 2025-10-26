import React, { useState } from 'react';
import { useAppContext, FocusArea } from '../context/AppContext';
import { ChartBarIcon, SpeakerWaveIcon, UserIcon } from '@heroicons/react/24/outline';

const FocusButton: React.FC<{ icon: React.ReactNode; label: string; description: string; onClick: () => void }> = ({ icon, label, description, onClick }) => (
    <button 
        onClick={onClick}
        className="w-full text-left p-4 bg-slate-800 rounded-lg border-2 border-slate-700 hover:bg-slate-700/50 hover:border-indigo-500 transition-all duration-200 flex items-center space-x-4 focus:outline-none focus:ring-2 focus:ring-indigo-500"
    >
        <div className="bg-slate-850 p-3 rounded-full">
            {icon}
        </div>
        <div>
            <p className="font-semibold text-white">{label}</p>
            <p className="text-sm text-slate-400">{description}</p>
        </div>
    </button>
);

const OnboardingModal: React.FC = () => {
    const { setFocusArea, setTopicKeywords } = useAppContext();
    const [keywords, setKeywords] = useState('');

    const handleSelectFocus = (focus: FocusArea) => {
        const keywordArray = keywords.split(',').map(k => k.trim()).filter(Boolean);
        setTopicKeywords(keywordArray);
        setFocusArea(focus);
    };

    return (
        <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in p-4">
            <div className="bg-slate-850 border border-slate-700 rounded-xl shadow-2xl p-6 md:p-8 max-w-lg w-full">
                <h1 className="text-2xl font-bold text-white mb-2">Welcome to Aura!</h1>
                <p className="text-slate-400 mb-6">Let's set up your session. First, what's your main goal today?</p>
                
                <div className="space-y-4">
                    <FocusButton 
                        icon={<ChartBarIcon className="h-6 w-6 text-teal-400" aria-hidden="true" />}
                        label="My Pacing"
                        description="I tend to talk too fast or too slow."
                        onClick={() => handleSelectFocus('pacing')}
                    />
                    <FocusButton 
                        icon={<SpeakerWaveIcon className="h-6 w-6 text-rose-400" aria-hidden="true" />}
                        label="My Filler Words"
                        description="I use words like 'um' or 'like' too often."
                        onClick={() => handleSelectFocus('fillers')}
                    />
                    <FocusButton 
                        icon={<UserIcon className="h-6 w-6 text-sky-400" aria-hidden="true" />}
                        label="My Body Language"
                        description="I'm not sure if I look confident."
                        onClick={() => handleSelectFocus('posture')}
                    />
                </div>
                
                <div className="mt-6">
                    <label htmlFor="keywords" className="block text-sm font-medium text-slate-300 mb-2">What's your topic? (Optional)</label>
                    <input
                        type="text"
                        id="keywords"
                        value={keywords}
                        onChange={(e) => setKeywords(e.target.value)}
                        placeholder="e.g., AI, education, healthcare"
                        className="w-full bg-slate-700 border-slate-600 rounded-md py-2 px-3 text-white placeholder-slate-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                    <p className="text-xs text-slate-500 mt-1">Separate keywords with commas.</p>
                </div>

                 <p className="text-xs text-slate-500 mt-6 text-center">Your choice helps us tailor the feedback just for you.</p>
            </div>
        </div>
    );
};

export default OnboardingModal;
