"""
Export utilities for saving summaries and Q&A history
"""
from datetime import datetime

def export_summary(summary, filename="summary.txt"):
    """
    Export summary to text file format
    
    Args:
        summary (str): Summary text
        filename (str): Output filename
        
    Returns:
        str: Formatted content for export
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""PDF Summary
Generated: {timestamp}
{'=' * 60}

{summary}

{'=' * 60}
End of Summary
"""
    return content

def export_qa_history(qa_history):
    """
    Export Q&A history to text file format
    
    Args:
        qa_history (list): List of Q&A dictionaries
        
    Returns:
        str: Formatted Q&A content for export
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""Q&A History
Generated: {timestamp}
{'=' * 60}

"""
    
    for i, qa in enumerate(qa_history, 1):
        content += f"""Question {i}:
{qa['question']}

Answer:
{qa['answer']}

Asked at: {qa['timestamp']}

{'-' * 60}

"""
    
    content += f"""{'=' * 60}
Total Questions: {len(qa_history)}
End of Q&A History
"""
    return content

def export_quiz_results(quiz, user_answers, score):
    """
    Export quiz results to text file format
    
    Args:
        quiz (list): Quiz questions
        user_answers (dict): User's answers
        score (int): Total score
        
    Returns:
        str: Formatted quiz results for export
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(quiz)
    percentage = (score / total * 100) if total > 0 else 0
    
    content = f"""Quiz Results
Generated: {timestamp}
{'=' * 60}

Score: {score}/{total} ({percentage:.1f}%)

"""
    
    for i, q in enumerate(quiz, 1):
        user_answer = user_answers.get(i-1, [""])[0] if i-1 in user_answers else ""
        is_correct = user_answer == q['correct_answer']
        
        content += f"""Question {i}:
{q['question']}

Your Answer: {user_answer} {'✓ Correct' if is_correct else '✗ Incorrect'}
Correct Answer: {q['correct_answer']}

Explanation:
{q['explanation']}

{'-' * 60}

"""
    
    content += f"""{'=' * 60}
Final Score: {score}/{total} ({percentage:.1f}%)
End of Quiz Results
"""
    return content