
import React, { useState } from 'react';
import type { Flashcard } from '../types';
import { RectangleStackIcon } from '../constants';

interface FlashcardViewProps {
  flashcards: Flashcard[];
}

const SingleFlashcard: React.FC<{ card: Flashcard }> = ({ card }) => {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div className="w-full h-64 perspective-1000" onClick={() => setIsFlipped(!isFlipped)}>
      <div
        className={`relative w-full h-full transform-style-preserve-3d transition-transform duration-700 ${isFlipped ? 'rotate-y-180' : ''}`}
      >
        {/* Front of card */}
        <div className="absolute w-full h-full backface-hidden bg-gray-700 rounded-lg p-6 flex flex-col items-center justify-center text-center">
            <p className="text-gray-400 text-sm mb-2">Question</p>
            <p className="text-xl font-semibold text-white">{card.question}</p>
        </div>
        {/* Back of card */}
        <div className="absolute w-full h-full backface-hidden bg-blue-800 rounded-lg p-6 flex flex-col items-center justify-center text-center rotate-y-180">
            <p className="text-blue-200 text-sm mb-2">Answer</p>
            <p className="text-lg text-white">{card.answer}</p>
        </div>
      </div>
    </div>
  );
};

const FlashcardView: React.FC<FlashcardViewProps> = ({ flashcards }) => {
  if (flashcards.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <RectangleStackIcon className="w-24 h-24 mb-4" />
        <h2 className="text-xl font-medium">No Flashcards Generated</h2>
        <p>Create flashcards from a summary to start studying.</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4 text-white">Flashcards</h1>
      <p className="text-gray-400 mb-6">Click on any card to flip it and reveal the answer.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {flashcards.map((card, index) => (
          <SingleFlashcard key={index} card={card} />
        ))}
      </div>
    </div>
  );
};

export default FlashcardView;
