import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SummarEase - AI PDF Summarizer",
    page_icon="📄",
    layout="wide"
)


# Configure API Key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'quiz' not in st.session_state:
    st.session_state.quiz = []

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center;
            font-family:'Segoe UI', Helvetica, sans-serif;
            background: linear-gradient(90deg, #ff6a00, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight:900;
            font-size:36px;
            margin-top:10px;
            margin-bottom:0;
        ">
            SummarEase
        </div>
        <p style="text-align:center; color:#adb5bd; font-size:14px; margin-top:2px;">
            AI-Powered PDF Summarizer
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    # st.caption("© 2025 SummarEase | Built with ❤️ using Streamlit")

    st.title("📚 About")
    st.info(
        "**AI PDF Summarizer**\n\n"
        "Features:\n"
        "- 📝 Summarization\n"
        "- ❓ Q&A System\n"
        "- 🔗 Reference Links\n"
        "- 🎯 Quiz Generation\n"
        "- 📄 PDF Viewer\n"
        "- 💾 Export Summary"
    )
    
    st.markdown("---")
    st.markdown("### 🎓 College Project")
    st.markdown("**Subject:** Artificial Intelligence")
    st.markdown("**Topic:** PDF Summarizer with AI")

# Main title
st.title("📄 AI-Powered PDF Summarizer")
st.markdown("Upload a PDF document to get started with AI-powered analysis")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
    return text

def generate_summary(text, summary_type="comprehensive"):
    """Generate summary using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if summary_type == "comprehensive":
            prompt = f"""Provide a comprehensive summary of the following document. 
Include key points, main arguments, and important details. 
Format the summary with clear sections and bullet points where appropriate.

Document:
{text[:30000]}  # Limiting text length for API
"""
        elif summary_type == "brief":
            prompt = f"""Provide a brief, concise summary of the following document in 3-5 sentences.
Focus on the most important points only.

Document:
{text[:30000]}
"""
        else:  # reference-linked
            prompt = f"""Provide a detailed summary of the following document with references to specific pages.
For each key point, indicate which page(s) it comes from using the format [Page X].

Document:
{text[:30000]}
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def answer_question(text, question):
    """Answer questions about the PDF using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Based on the following document, answer this question: {question}

Provide a clear, detailed answer and reference specific parts of the document if possible.

Document:
{text[:30000]}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return None

def generate_quiz(text, num_questions=5):
    """Generate quiz questions from the PDF"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
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
{text[:30000]}
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
        st.error(f"Error generating quiz: {str(e)}")
        return None

def export_summary(summary, filename="summary.txt"):
    """Export summary to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""PDF Summary
Generated: {timestamp}

{summary}
"""
    return content

# Main content
if uploaded_file is not None:
    # Extract text from PDF
    if not st.session_state.pdf_text:
        with st.spinner("Extracting text from PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summary", 
        "❓ Q&A", 
        "🎯 Quiz", 
        "📄 PDF Viewer",
        "💾 Export"
    ])
    
    # Tab 1: Summary
    with tab1:
        st.header("Document Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            summary_type = st.selectbox(
                "Summary Type",
                ["comprehensive", "brief", "reference-linked"]
            )
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary = generate_summary(
                    st.session_state.pdf_text, 
                    summary_type
                )
        
        if st.session_state.summary:
            st.markdown("### Summary:")
            st.markdown(st.session_state.summary)
    
    # Tab 2: Q&A
    with tab2:
        st.header("Question & Answer")
        
        question = st.text_input("Ask a question about the document:")
        
        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Finding answer..."):
                    answer = answer_question(st.session_state.pdf_text, question)
                    if answer:
                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
        
        # Display Q&A history
        if st.session_state.qa_history:
            st.markdown("### Q&A History:")
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(f"Q: {qa['question']}", expanded=(i==0)):
                    st.markdown(f"**Answer:** {qa['answer']}")
                    st.caption(f"Asked at: {qa['timestamp']}")
    
    # Tab 3: Quiz
    with tab3:
        st.header("Quiz Generation")
        
        num_questions = st.slider("Number of questions", 3, 10, 5)
        
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = generate_quiz(
                    st.session_state.pdf_text, 
                    num_questions
                )
        
        if st.session_state.quiz:
            st.markdown("### Quiz Questions:")
            user_answers = {}
            
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                user_answers[i] = st.radio(
                    f"Select your answer for Q{i+1}:",
                    q['options'],
                    key=f"q_{i}"
                )
                st.markdown("---")
            
            if st.button("Submit Quiz"):
                score = 0
                for i, q in enumerate(st.session_state.quiz):
                    user_answer = user_answers[i][0]  # Get letter (A, B, C, D)
                    if user_answer == q['correct_answer']:
                        score += 1
                        st.success(f"Q{i+1}: Correct! ✓")
                    else:
                        st.error(f"Q{i+1}: Incorrect. Correct answer: {q['correct_answer']}")
                    st.info(f"Explanation: {q['explanation']}")
                
                st.markdown(f"### Final Score: {score}/{len(st.session_state.quiz)}")
                percentage = (score / len(st.session_state.quiz)) * 100
                st.progress(percentage / 100)
    
    # Tab 4: PDF Viewer
    with tab4:
        st.header("PDF Document Viewer")
        
        # Display PDF text with page markers
        st.markdown("### Document Content:")
        with st.expander("View extracted text", expanded=False):
            st.text_area(
                "Document Text",
                st.session_state.pdf_text,
                height=400,
                label_visibility="collapsed"
            )
        
        # Display PDF file info
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        st.info(f"Total Pages: {len(pdf_reader.pages)}")
    
    # Tab 5: Export
    with tab5:
        st.header("Export Summary")
        
        if st.session_state.summary:
            export_content = export_summary(st.session_state.summary)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=export_content,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Export Q&A history
                if st.session_state.qa_history:
                    qa_content = "Q&A History\n" + "="*50 + "\n\n"
                    for qa in st.session_state.qa_history:
                        qa_content += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
                    
                    st.download_button(
                        label="📥 Download Q&A History",
                        data=qa_content,
                        file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            st.markdown("### Preview:")
            st.text_area("Export Preview", export_content, height=300)
        else:
            st.warning("Please generate a summary first before exporting.")

else:
    # Landing page
    st.info("👆 Upload a PDF file to get started!")
    
    st.markdown("""
    ### How to use:
    1. **Upload PDF**: Upload your PDF document above
    2. **Explore Features**:
       - Generate summaries (comprehensive, brief, or with references)
       - Ask questions about the document
       - Generate and take quizzes
       - View the PDF content
       - Export summaries and Q&A history
    
    ### Features:
    - **Smart Summarization**: Get comprehensive or brief summaries with AI
    - **Interactive Q&A**: Ask any question about your document
    - **Reference-Linked**: Summaries with page references
    - **Quiz Generation**: Auto-generate quizzes to test understanding
    - **PDF Viewer**: View extracted text from your PDF
    - **Export Options**: Download summaries and Q&A history
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Made with ❤️ using Streamlit and Google Gemini AI</div>",
    unsafe_allow_html=True
)