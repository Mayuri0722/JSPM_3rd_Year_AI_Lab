import { GoogleGenAI, Type } from "@google/genai";
import type { ChatMessage, Flashcard, Summary } from '../types';

const getAiClient = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function summarizeText(text: string): Promise<{ summary: string, title: string }> {
  const ai = getAiClient();
  const model = 'gemini-2.5-flash';
  const prompt = `Summarize the following content from a web page. Also, provide a short, concise title (5 words or less) for it. Return the response as a JSON object with two keys: "title" and "summary".\n\nCONTENT:\n${text}`;
  
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          title: { type: Type.STRING },
          summary: { type: Type.STRING }
        },
        required: ['title', 'summary']
      }
    }
  });

  const jsonResponse = JSON.parse(response.text);
  return { summary: jsonResponse.summary, title: jsonResponse.title };
}

export async function chatWithContext(history: ChatMessage[], summaries: Summary[]): Promise<string> {
    const ai = getAiClient();
    const model = 'gemini-2.5-pro';

    let systemInstruction = "You are TabMind, an AI assistant. Answer the user's question based on the provided CONTEXT (a list of web page summaries) and the conversation HISTORY. If the answer is not in the context, say so.\n\n--- CONTEXT ---\n";
    if (summaries.length > 0) {
        systemInstruction += summaries.map(s => `Title: ${s.title}\nSummary: ${s.summary}`).join('\n\n');
    } else {
        systemInstruction += "No summaries available.";
    }
    systemInstruction += "\n--- END CONTEXT ---";

    const contents = history.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.content }]
    }));

    const response = await ai.models.generateContent({
        model,
        contents: contents,
        config: {
            systemInstruction: systemInstruction,
        },
    });

    return response.text;
}

export async function generateFlashcards(text: string): Promise<Flashcard[]> {
  const ai = getAiClient();
  const model = 'gemini-2.5-flash';
  const prompt = `Based on the following text, generate a set of 3 to 5 flashcards. Each flashcard should have a clear question and a concise answer. The questions should test the key concepts from the text. Format the output as a JSON array of objects, where each object has "question" and "answer" keys.\n\nTEXT:\n${text}`;
  
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
        responseMimeType: 'application/json',
        responseSchema: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    question: { type: Type.STRING },
                    answer: { type: Type.STRING }
                },
                required: ['question', 'answer']
            }
        }
    }
  });
  
  return JSON.parse(response.text);
}

export async function translateText(text: string, language: string): Promise<string> {
    const ai = getAiClient();
    const model = 'gemini-2.5-flash';
    const prompt = `Translate the following text into ${language}:\n\n${text}`;
    
    const response = await ai.models.generateContent({
        model,
        contents: prompt
    });
    
    return response.text;
}

export async function rephraseText(text: string, tone: string): Promise<string> {
    const ai = getAiClient();
    const model = 'gemini-2.5-flash';
    const prompt = `Rephrase the following text to have a more ${tone} tone:\n\n${text}`;

    const response = await ai.models.generateContent({
        model,
        contents: prompt
    });

    return response.text;
}