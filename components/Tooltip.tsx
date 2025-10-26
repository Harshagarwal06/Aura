import React, { useState, ReactNode } from 'react';

interface TooltipProps {
  children: ReactNode;
  content: ReactNode;
}

const Tooltip: React.FC<TooltipProps> = ({ children, content }) => {
  const [visible, setVisible] = useState(false);
  
  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
      >
        {children}
      </div>
      {visible && (
        <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 animate-fade-in-up">
          <div className="bg-slate-800 text-white text-xs rounded-lg py-2 px-3 whitespace-nowrap glass-effect border border-slate-700 shadow-xl">
            {content}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-slate-700" />
          </div>
        </div>
      )}
    </div>
  );
};

export default Tooltip;