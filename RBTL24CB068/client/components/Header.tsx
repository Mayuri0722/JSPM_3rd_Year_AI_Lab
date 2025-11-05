
import React from 'react';
import { SparklesIcon } from '../constants';

const Header: React.FC = () => {
  return (
    <header className="flex items-center justify-between p-4 bg-gray-900/50 border-b border-gray-700/50 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <SparklesIcon className="w-8 h-8 text-blue-400" />
        <div>
          <h1 className="text-xl font-bold text-white">TabMind</h1>
          <p className="text-xs text-gray-400">Your Private Thinking Companion</p>
        </div>
      </div>
    </header>
  );
};

export default Header;
