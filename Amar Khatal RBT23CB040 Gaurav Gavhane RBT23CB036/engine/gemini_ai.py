import os
import google.generativeai as genai

#  Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyDS-bUY51uB2Vfn2RWLte46BGiQ6Yw1oak"

#  Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

#  Create a function to get AI response
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"
