import React from 'react';

interface ProgressBarProps {
    progress: number;
    color?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, color = 'from-rose-500 to-pink-500' }) => {
  return (
    <div className="w-full bg-slate-700 rounded-full h-2.5">
      <div 
        className={`h-2.5 rounded-full bg-gradient-to-r ${color} transition-all duration-500 ease-out`}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
};

export default ProgressBar;