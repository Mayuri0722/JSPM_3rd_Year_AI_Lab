
import React from 'react';
import type { View, ChatMessage, Summary, Flashcard } from '../types';
import ChatView from './ChatView';
import SummaryDetailView from './SummaryDetailView';
import FlashcardView from './FlashcardView';
import Login from './Login';
import Signup from './Signup';
import { Routes, Route, Navigate } from 'react-router-dom';
import AdminView from './AdminView';
import { RequireAuth, RequireRole } from './RouteGuards';

interface MainContentProps {
  activeView: View;
  chatHistory: ChatMessage[];
  onChatSubmit: (message: string) => void;
  selectedSummary: Summary | null;
  flashcards: Flashcard[];
  onGenerateFlashcards: (text: string) => void;
  onTranslate: (text: string, language: string) => Promise<void>;
  onRephrase: (text: string, tone: string) => Promise<void>;
  isLoading: boolean;
  onAuthSuccess?: (payload: { token: string; user: { id: string; name: string; email: string; role: 'user' | 'admin' } }) => void;
  currentUser?: { id: string; name: string; email: string; role: 'user' | 'admin' } | null;
  authToken?: string | null;
}

const MainContent: React.FC<MainContentProps> = ({
  activeView,
  chatHistory,
  onChatSubmit,
  selectedSummary,
  flashcards,
  onGenerateFlashcards,
  onTranslate,
  onRephrase,
  isLoading,
  onAuthSuccess,
  currentUser,
  authToken,
}) => {
  return (
    <main className="flex-1 p-6 bg-gray-900 overflow-y-auto">
      <Routes>
        <Route
          path="/"
          element={
            <RequireAuth currentUser={currentUser} redirectTo="/signup">
              <ChatView chatHistory={chatHistory} onChatSubmit={onChatSubmit} isLoading={isLoading} />
            </RequireAuth>
          }
        />
        <Route
          path="/flashcards"
          element={
            <RequireAuth currentUser={currentUser} redirectTo="/signup">
              <FlashcardView flashcards={flashcards} />
            </RequireAuth>
          }
        />
        {/* Keeping summary route for potential deep-linking */}
        <Route
          path="/summary"
          element={
            <RequireAuth currentUser={currentUser} redirectTo="/signup">
              <SummaryDetailView
                summary={selectedSummary}
                onGenerateFlashcards={onGenerateFlashcards}
                onTranslate={onTranslate}
                onRephrase={onRephrase}
                isLoading={isLoading}
              />
            </RequireAuth>
          }
        />
        <Route
          path="/admin"
          element={
            <RequireRole
              currentUser={currentUser}
              roles={['admin']}
              redirectToIfUnauthed="/signup"
              redirectToIfForbidden="/"
            >
              <AdminView token={authToken || ''} />
            </RequireRole>
          }
        />
        <Route path="/login" element={currentUser ? <Navigate to="/" replace /> : <Login onSuccess={onAuthSuccess!} />} />
        <Route path="/signup" element={currentUser ? <Navigate to="/" replace /> : <Signup />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </main>
  );
};

export default MainContent;

