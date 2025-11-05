import json
import os
from typing import Tuple

import joblib
import numpy as np
import streamlit as st
import pandas as pd
import preprocessing  # ensure importable module for unpickling

MODEL_PATH = "spam_model.pkl"
METRICS_PATH = "metrics.json"

st.set_page_config(page_title="AI Email Classifier", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Please run: python model.py"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_with_confidence(model, text: str) -> Tuple[str, float, np.ndarray]:
    probs = None
    input_series = pd.Series([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_series)[0]
    pred = model.predict(input_series)[0]
    label = "Spam" if int(pred) == 1 else "Not Spam"
    if probs is not None:
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs = np.array([1.0 - float(pred), float(pred)])
    return label, confidence, probs


def main():
    st.markdown(
        """
        <style>
            /* Background gradient and base font */
            .stApp {
                background: linear-gradient(180deg, #c7deff 0%, #a5c7ff 100%);
                font-family: 'Times New Roman', serif;
                min-height: 100vh;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }

            /* Container */
            .container {
                background: transparent;
                border-radius: 16px;
                padding: 0 40px 40px 40px;
                width: 420px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            /* Title with larger font size and Times New Roman font, centered */
            .title {
                font-family: 'Times New Roman', serif;
                font-weight: 700;
                font-size: 48px;
                color: #1e293b;
                margin-bottom: 24px;
                display: flex;
                justify-content: center;
                text-align: center;
                gap: 10px;
                width: 100%;
            }

            /* Textarea with previous size */
            textarea {
                width: 100% !important;
                height: 120px !important;
                border-radius: 12px !important;
                border: 1px solid #cbd5e1 !important;
                padding: 12px !important;
                font-size: 15px !important;
                font-family: 'Times New Roman', serif !important;
                box-sizing: border-box;
                resize: vertical !important;
                outline: none !important;
                transition: border-color 0.3s ease;
            }

            textarea::placeholder {
                color: #94a3b8;
            }

            textarea:focus {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 8px #3b82f6aa !important;
            }

            /* Button */
            div.stButton > button:first-child {
                background: linear-gradient(90deg, #3b82f6, #7c3aed);
                border: none;
                color: white;
                font-weight: 700;
                padding: 12px 0;
                border-radius: 12px;
                width: 100%;
                cursor: pointer;
                font-size: 17px;
                box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
                transition: transform 0.1s ease-in-out, box-shadow 0.15s ease-in-out;
                margin-top: 20px;
            }
            div.stButton > button:first-child:hover {
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 12px 28px rgba(124, 58, 237, 0.4);
            }

            /* Result box */
            .result-box {
                margin-top: 25px;
                border-radius: 12px;
                padding: 18px 20px;
                width: 100%;
                box-sizing: border-box;
                font-weight: 600;
                font-size: 16px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .spam {
                background-color: #ffedd5;
                border: 1.75px solid #f97316;
                color: #b45309;
            }
            .not-spam {
                background-color: #ecfdf5;
                border: 1.75px solid #10b981;
                color: #047857;
            }

            /* Icon in results */
            .icon {
                font-size: 20px;
                margin-right: 8px;
            }

            /* Progress bar container */
            .progress-container {
                width: 100%;
                background-color: #e5e7eb;
                height: 12px;
                border-radius: 8px;
                overflow: hidden;
                margin-top: 4px;
            }

            /* Progress bar fill */
            .progress-fill {
                height: 100%;
                border-radius: 8px;
                background: linear-gradient(90deg, #f97316, #fb923c);
                transition: width 0.4s ease;
            }

            /* Footer */
            .footer {
                margin-top: 40px;
                color: #64748b;
                font-size: 13px;
                text-align: center;
                font-weight: 400;
                user-select: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="container">', unsafe_allow_html=True)

    # Centered and large title with emoji
    st.markdown('<div class="title">📩 AI Email Classifier – Spam or Not Spam</div>', unsafe_allow_html=True)

    model = load_model()

    user_text = st.text_area("Paste your email content here…", placeholder="Paste your email content here...")

    if st.button("Check for Spam"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing with AI..."):
                label, confidence, probs = predict_with_confidence(model, user_text)

            confidence_pct = int(confidence * 100)

            if label == "Spam":
                st.markdown(
                    f'''
                    <div class="result-box spam">
                        <div><span class="icon">⚠️</span>This email is likely Spam</div>
                        <div>{confidence_pct}% Spam Probability</div>
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {confidence_pct}%;"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''
                    <div class="result-box not-spam">
                        <div><span class="icon">✅</span>This email appears Safe</div>
                        <div>{confidence_pct}% Safe Probability</div>
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {confidence_pct}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '''
        <div class="footer">
            Developed by Kalyani Dupare | Powered by Streamlit & Machine Learning
        </div>
        ''',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
