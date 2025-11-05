"""
Configuration file for the PDF Summarizer application
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Text Processing Limits
MAX_TEXT_LENGTH = 30000

# App Configuration
APP_TITLE = "SummarEase - AI PDF Summarizer"
APP_ICON = "📄"
LAYOUT = "wide"

def configure_gemini():
    """Configure Gemini API with the API key"""
    if not GEMINI_API_KEY:
        return False
    genai.configure(api_key=GEMINI_API_KEY)
    return True