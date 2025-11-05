
export interface Summary {
  id: string;
  title: string;
  originalContent: string;
  summary: string;
  timestamp: string;
}

export interface Flashcard {
  question: string;
  answer: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

export type View = 'chat' | 'summary' | 'flashcards' | 'login' | 'signup' | 'admin';
