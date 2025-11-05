
import React, { useState, useRef, useEffect } from 'react';
import type { ChatMessage } from '../types';
import { SparklesIcon } from '../constants';
import Loader from './Loader';

interface ChatViewProps {
  chatHistory: ChatMessage[];
  onChatSubmit: (message: string) => void;
  isLoading: boolean;
}

const ChatView: React.FC<ChatViewProps> = ({ chatHistory, onChatSubmit, isLoading }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [chatHistory]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onChatSubmit(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full">
      <h1 className="text-2xl font-bold mb-4 text-white">Chat With Your Past Tabs</h1>
      <div className="flex-1 overflow-y-auto pr-4 -mr-4 space-y-4">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'model' && <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0"><SparklesIcon className="w-5 h-5 text-white" /></div>}
            <div
              className={`max-w-xl p-3 rounded-lg ${
                msg.role === 'user' ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-300'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && chatHistory[chatHistory.length - 1]?.role === 'user' && (
           <div className="flex gap-3 justify-start">
             <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0"><SparklesIcon className="w-5 h-5 text-white" /></div>
             <div className="max-w-xl p-3 rounded-lg bg-gray-800 text-gray-300">
               <Loader />
             </div>
           </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your summarized tabs..."
            className="w-full p-3 pr-12 bg-gray-800 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="absolute inset-y-0 right-0 flex items-center justify-center w-10 text-gray-400 hover:text-white disabled:text-gray-600"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M3.105 2.289a.75.75 0 0 0-.826.95l1.414 4.925A1.5 1.5 0 0 0 5.135 9.25h6.115a.75.75 0 0 1 0 1.5H5.135a1.5 1.5 0 0 0-1.442 1.086L2.279 16.76a.75.75 0 0 0 .95.826l16-5.333a.75.75 0 0 0 0-1.418l-16-5.333Z" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatView;
