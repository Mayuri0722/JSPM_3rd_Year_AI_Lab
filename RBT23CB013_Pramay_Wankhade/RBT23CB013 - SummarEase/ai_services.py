"""
AI services using Google Gemini API
"""
import google.generativeai as genai
import json
from config import GEMINI_MODEL, MAX_TEXT_LENGTH

def generate_summary(text, summary_type="comprehensive"):
    """
    Generate summary using Gemini API
    
    Args:
        text (str): Document text
        summary_type (str): Type of summary ('comprehensive', 'brief', 'reference-linked')
        
    Returns:
        str: Generated summary or None if error
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        if summary_type == "comprehensive":
            prompt = f"""Provide a comprehensive summary of the following document. 
Include key points, main arguments, and important details. 
Format the summary with clear sections and bullet points where appropriate.

Document:
{text[:MAX_TEXT_LENGTH]}
"""
        elif summary_type == "brief":
            prompt = f"""Provide a brief, concise summary of the following document in 3-5 sentences.
Focus on the most important points only.

Document:
{text[:MAX_TEXT_LENGTH]}
"""
        else:  # reference-linked
            prompt = f"""Provide a detailed summary of the following document with references to specific pages.
For each key point, indicate which page(s) it comes from using the format [Page X].

Document:
{text[:MAX_TEXT_LENGTH]}
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")

def answer_question(text, question):
    """
    Answer questions about the PDF using Gemini API
    
    Args:
        text (str): Document text
        question (str): User's question
        
    Returns:
        str: Answer or None if error
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Based on the following document, answer this question: {question}

Provide a clear, detailed answer and reference specific parts of the document if possible.

Document:
{text[:MAX_TEXT_LENGTH]}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error answering question: {str(e)}")

def generate_quiz(text, num_questions=5):
    """
    Generate quiz questions from the PDF
    
    Args:
        text (str): Document text
        num_questions (int): Number of questions to generate
        
    Returns:
        list: Quiz questions or None if error
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Based on the following document, create {num_questions} multiple-choice questions to test understanding.

For each question, provide:
1. The question
2. Four options (A, B, C, D)
3. The correct answer
4. A brief explanation

Format your response as JSON with this structure:
[
  {{
    "question": "Question text",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Explanation text"
  }}
]

Document:
{text[:MAX_TEXT_LENGTH]}
"""
        response = model.generate_content(prompt)
        
        # Try to parse JSON from response
        response_text = response.text
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        quiz_data = json.loads(response_text.strip())
        return quiz_data
    except Exception as e:
        raise Exception(f"Error generating quiz: {str(e)}")