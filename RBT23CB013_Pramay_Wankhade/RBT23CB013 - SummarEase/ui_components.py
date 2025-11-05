"""
UI Components for the Streamlit application
"""
import streamlit as st

def render_sidebar():
    """Render the sidebar with project information"""
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

def render_landing_page():
    """Render the landing page when no PDF is uploaded"""
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

def render_summary_tab(pdf_text):
    """Render the Summary tab content"""
    from ai_services import generate_summary
    
    st.header("Document Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        summary_type = st.selectbox(
            "Summary Type",
            ["comprehensive", "brief", "reference-linked"]
        )
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            try:
                st.session_state.summary = generate_summary(pdf_text, summary_type)
            except Exception as e:
                st.error(str(e))
    
    if st.session_state.summary:
        st.markdown("### Summary:")
        st.markdown(st.session_state.summary)

def render_qa_tab(pdf_text):
    """Render the Q&A tab content"""
    from ai_services import answer_question
    from datetime import datetime
    
    st.header("Question & Answer")
    
    question = st.text_input("Ask a question about the document:")
    
    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Finding answer..."):
                try:
                    answer = answer_question(pdf_text, question)
                    if answer:
                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                except Exception as e:
                    st.error(str(e))
    
    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("### Q&A History:")
        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            with st.expander(f"Q: {qa['question']}", expanded=(i==0)):
                st.markdown(f"**Answer:** {qa['answer']}")
                st.caption(f"Asked at: {qa['timestamp']}")

def render_quiz_tab(pdf_text):
    """Render the Quiz tab content"""
    from ai_services import generate_quiz
    
    st.header("Quiz Generation")
    
    num_questions = st.slider("Number of questions", 3, 10, 5)
    
    if st.button("Generate Quiz", type="primary"):
        with st.spinner("Generating quiz..."):
            try:
                st.session_state.quiz = generate_quiz(pdf_text, num_questions)
            except Exception as e:
                st.error(str(e))
    
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

def render_pdf_viewer_tab(pdf_text, pdf_file):
    """Render the PDF Viewer tab content"""
    from pdf_processor import get_pdf_info
    
    st.header("PDF Document Viewer")
    
    # Display PDF text with page markers
    st.markdown("### Document Content:")
    with st.expander("View extracted text", expanded=False):
        st.text_area(
            "Document Text",
            pdf_text,
            height=400,
            label_visibility="collapsed"
        )
    
    # Display PDF file info
    pdf_info = get_pdf_info(pdf_file)
    st.info(f"Total Pages: {pdf_info['page_count']}")

def render_export_tab():
    """Render the Export tab content"""
    from export_utils import export_summary, export_qa_history
    from datetime import datetime
    
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
                qa_content = export_qa_history(st.session_state.qa_history)
                
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