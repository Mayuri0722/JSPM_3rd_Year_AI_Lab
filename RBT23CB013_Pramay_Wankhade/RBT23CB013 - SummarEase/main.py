"""
AI-Powered PDF Summarizer
Main application file - Streamlit interface
"""
import streamlit as st

# Import custom modules
from config import APP_TITLE, APP_ICON, LAYOUT, configure_gemini
from pdf_processor import extract_text_from_pdf
from ui_components import (
    render_sidebar,
    render_landing_page,
    render_summary_tab,
    render_qa_tab,
    render_quiz_tab,
    render_pdf_viewer_tab,
    render_export_tab
)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT
)

# Configure Gemini API
if not configure_gemini():
    st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it in the .env file.")
    st.stop()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'quiz' not in st.session_state:
        st.session_state.quiz = []

init_session_state()

# Render sidebar
render_sidebar()

# Main title
st.title("📄 AI-Powered PDF Summarizer")
st.markdown("Upload a PDF document to get started with AI-powered analysis")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Main content
if uploaded_file is not None:
    # Extract text from PDF
    if not st.session_state.pdf_text:
        with st.spinner("Extracting text from PDF..."):
            try:
                st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(str(e))
                st.stop()
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summary", 
        "❓ Q&A", 
        "🎯 Quiz", 
        "📄 PDF Viewer",
        "💾 Export"
    ])
    
    # Render each tab
    with tab1:
        render_summary_tab(st.session_state.pdf_text)
    
    with tab2:
        render_qa_tab(st.session_state.pdf_text)
    
    with tab3:
        render_quiz_tab(st.session_state.pdf_text)
    
    with tab4:
        render_pdf_viewer_tab(st.session_state.pdf_text, uploaded_file)
    
    with tab5:
        render_export_tab()

else:
    # Landing page
    render_landing_page()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Made with ❤️ using Streamlit and Google Gemini AI</div>",
    unsafe_allow_html=True
)