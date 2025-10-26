import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-slate-850/80 backdrop-blur-md sticky top-0 z-50 border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-gradient-start to-gradient-end rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">A</span>
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-rose-400 to-pink-400 bg-clip-text text-transparent">
              Aura Coach
            </h1>
          </div>
          
          <div className="flex space-x-1 bg-slate-750 p-1 rounded-lg">
            <NavLink 
              to="/"
              className={({ isActive }) => 
                `px-3 py-2 rounded-md transition-all duration-200 text-sm font-medium ${
                  isActive 
                    ? 'bg-slate-700 text-white shadow-sm' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`
              }
            >
              Live Session
            </NavLink>
             <NavLink 
              to="/analysis"
              className={({ isActive }) => 
                `px-3 py-2 rounded-md transition-all duration-200 text-sm font-medium ${
                  isActive 
                    ? 'bg-slate-700 text-white shadow-sm' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`
              }
            >
              Speaking Coach
            </NavLink>
            <NavLink 
              to="/comprehensive-analysis"
              className={({ isActive }) => 
                `px-3 py-2 rounded-md transition-all duration-200 text-sm font-medium ${
                  isActive 
                    ? 'bg-slate-700 text-white shadow-sm' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`
              }
            >
              Comprehensive Analysis
            </NavLink>
            <NavLink 
              to="/report"
              className={({ isActive }) => 
                `px-3 py-2 rounded-md transition-all duration-200 text-sm font-medium ${
                  isActive 
                    ? 'bg-slate-700 text-white shadow-sm' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`
              }
            >
              Session Report
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;