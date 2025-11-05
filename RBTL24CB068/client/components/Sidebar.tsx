
import React, { useState } from 'react';
import type { Summary, View } from '../types';
import { ChatBubbleIcon, DocumentTextIcon, RectangleStackIcon } from '../constants';
import Loader from './Loader';

interface SidebarProps {
  summaries: Summary[];
  onSummarize: (text: string) => void;
  onSelectSummary: (summary: Summary) => void;
  isLoading: boolean;
  onNavigate: (view: View) => void;
  activeView: View;
  currentUser?: { id: string; name: string; email: string; role: 'user' | 'admin' } | null;
  onLogout?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ summaries, onSummarize, onSelectSummary, isLoading, onNavigate, activeView, currentUser, onLogout }) => {
  const [content, setContent] = useState('');

  const handleSubmit = () => {
    if (content.trim()) {
      onSummarize(content);
      setContent('');
    }
  };
  
  const NavItem: React.FC<{ view: View; icon: React.ReactNode; label: string }> = ({ view, icon, label }) => (
    <button
      onClick={() => onNavigate(view)}
      className={`flex items-center gap-3 w-full text-left px-3 py-2 rounded-md transition-colors ${
        activeView === view ? 'bg-blue-600/20 text-blue-300' : 'text-gray-400 hover:bg-gray-700/50 hover:text-gray-200'
      }`}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );


  return (
    <aside className="w-1/3 max-w-sm bg-gray-800/50 p-4 border-r border-gray-700/50 flex flex-col">
      <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-700/50">
        <h2 className="text-lg font-semibold mb-2 text-white">Summarize New Tab</h2>
        <p className="text-sm text-gray-400 mb-4">Paste content here to create a new memory.</p>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Paste article content..."
          className="w-full h-32 p-2 bg-gray-800 border border-gray-600 rounded-md resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
          disabled={isLoading}
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading || !content.trim()}
          className="w-full mt-2 px-4 py-2 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {isLoading ? <Loader /> : 'Create Memory'}
        </button>
      </div>

      <nav className="my-6 space-y-2">
         <NavItem view="chat" icon={<ChatBubbleIcon className="w-5 h-5" />} label="Chat with Tabs" />
         <NavItem view="flashcards" icon={<RectangleStackIcon className="w-5 h-5" />} label="Flashcards" />
         {currentUser?.role === 'admin' && (
           <NavItem view="admin" icon={<DocumentTextIcon className="w-5 h-5" />} label="Admin" />
         )}
         <div className="h-px bg-gray-700/50 my-2" />
         {!currentUser ? (
           <>
             <NavItem view="login" icon={<DocumentTextIcon className="w-5 h-5" />} label="Login" />
             <NavItem view="signup" icon={<DocumentTextIcon className="w-5 h-5" />} label="Sign Up" />
           </>
         ) : (
           <div className="flex items-center justify-between px-3 py-2 rounded-md bg-gray-800/60 border border-gray-700/50">
             <div>
               <div className="text-sm text-gray-200 font-medium">{currentUser.name}</div>
               <div className="text-xs text-gray-400">{currentUser.role}</div>
             </div>
             <button onClick={onLogout} className="text-sm text-red-300 hover:text-red-200">Logout</button>
           </div>
         )}
      </nav>

      <div className="flex-1 overflow-y-auto mt-4 pt-4 border-t border-gray-700/50">
        <h3 className="text-base font-semibold text-gray-300 px-2 mb-2">Memories</h3>
        {summaries.length > 0 ? (
          <ul className="space-y-1">
            {summaries.map((summary) => (
              <li key={summary.id}>
                <button
                  onClick={() => onSelectSummary(summary)}
                  className="w-full text-left p-3 rounded-md hover:bg-gray-700/50 transition-colors"
                >
                  <p className="font-medium text-gray-100 truncate">{summary.title}</p>
                  <p className="text-xs text-gray-400">{new Date(summary.timestamp).toLocaleString()}</p>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-center py-8 px-4 text-gray-500">
            <DocumentTextIcon className="w-12 h-12 mx-auto mb-2" />
            <p className="text-sm">Your summarized tabs will appear here.</p>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;

