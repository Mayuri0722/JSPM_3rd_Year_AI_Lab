
import React from 'react';
import type { Summary } from '../types';
import { DocumentTextIcon, RectangleStackIcon, LanguageIcon, PencilSquareIcon } from '../constants';
import Loader from './Loader';

interface SummaryDetailViewProps {
  summary: Summary | null;
  onGenerateFlashcards: (text: string) => void;
  onTranslate: (text: string, language: string) => Promise<void>;
  onRephrase: (text: string, tone: string) => Promise<void>;
  isLoading: boolean;
}

const ActionButton: React.FC<{ icon: React.ReactNode; label: string; onClick: () => void; disabled: boolean; }> = 
({ icon, label, onClick, disabled }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-300 bg-gray-700/50 rounded-md hover:bg-gray-600/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
        {icon}
        <span>{label}</span>
    </button>
);


const SummaryDetailView: React.FC<SummaryDetailViewProps> = ({ summary, onGenerateFlashcards, onTranslate, onRephrase, isLoading }) => {
  if (!summary) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <DocumentTextIcon className="w-24 h-24 mb-4" />
        <h2 className="text-xl font-medium">No Summary Selected</h2>
        <p>Select a memory from the sidebar to view its details.</p>
      </div>
    );
  }

  const handleTranslate = () => {
    const language = prompt("Enter the language to translate to (e.g., Spanish, French):");
    if (language) {
      onTranslate(summary.summary, language);
    }
  };

  const handleRephrase = () => {
    const tone = prompt("Enter the desired tone (e.g., formal, casual, professional):");
    if (tone) {
      onRephrase(summary.summary, tone);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white">{summary.title}</h1>
        <p className="text-sm text-gray-400">Summarized on {new Date(summary.timestamp).toLocaleString()}</p>
      </div>

       <div className="flex items-center gap-4 mb-6 pb-4 border-b border-gray-700/50">
          <ActionButton 
            icon={<RectangleStackIcon className="w-5 h-5"/>} 
            label="Create Flashcards" 
            onClick={() => onGenerateFlashcards(summary.summary)}
            disabled={isLoading}
          />
          <ActionButton 
            icon={<LanguageIcon className="w-5 h-5"/>} 
            label="Translate Summary" 
            onClick={handleTranslate}
            disabled={isLoading}
          />
          <ActionButton 
            icon={<PencilSquareIcon className="w-5 h-5"/>} 
            label="Rephrase Summary"
            onClick={handleRephrase}
            disabled={isLoading}
          />
          {isLoading && <div className="flex items-center gap-2 text-blue-400"><Loader /><span>Processing...</span></div>}
       </div>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 overflow-y-auto">
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2 text-gray-300">Generated Summary</h2>
          <p className="text-gray-200 whitespace-pre-wrap leading-relaxed">{summary.summary}</p>
        </div>
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2 text-gray-400">Original Content</h2>
          <p className="text-gray-400 whitespace-pre-wrap leading-relaxed text-sm">{summary.originalContent}</p>
        </div>
      </div>
    </div>
  );
};

export default SummaryDetailView;
