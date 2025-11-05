# 📧 AI Email Classifier – Spam or Not Spam

A modern Streamlit web app that classifies email text as Spam or Not Spam using a TF‑IDF + Logistic Regression model. The UI features a premium, minimal design with a built‑in light/dark theme toggle, animated CTA, and styled result cards.

## ✨ Features
- Elegant, centered interface with soft gradient backgrounds
- Theme toggle (🌗 light/dark) with smooth transitions
- Large textarea with modern card design and hover lift
- Gradient primary button (blue → purple) with subtle glow
- Clear result states:
  - ✅ This email appears Safe
  - ⚠️ This email is likely Spam
- Confidence indicator bar
- Metrics expander with accuracy and macro‑avg report

## 🧠 Model
- Pipeline: `TfidfVectorizer` → `LogisticRegression`
- Dataset: SMS Spam Collection (downloaded automatically)
- Training script: `model.py` writes `spam_model.pkl` and `metrics.json`

## 📦 Tech Stack
- Python, Streamlit
- scikit‑learn, pandas, numpy, nltk
- joblib for model serialization

## 🚀 Quickstart
> Commands are shown for Windows PowerShell. Replace with your OS equivalents if needed.

1) Clone and enter the project
```powershell
git clone <your-repo-url>.git
cd AI_project
```

2) Create and activate a virtual environment
```powershell
# If Python is on PATH
python -m venv .venv

# Activate (temporary policy bypass if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

3) Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4) Train the model (first run only or when retraining)
```powershell
python model.py
```
This produces `spam_model.pkl` and `metrics.json` in the project root.

5) Run the app
```powershell
python -m streamlit run app.py
```
Open the printed URL, typically `http://localhost:8501`.

## 🧭 Using the App
- Toggle theme via the top‑left “🌗 Theme” button
- Paste email content in the large textbox
- Click “Check for Spam”
- View the styled result card and confidence bar
- Expand “Model Performance” at the bottom for metrics

## 🗂️ Project Structure
```
AI_project/
├─ app.py                # Streamlit UI + inference
├─ model.py              # Training script (downloads data, trains, saves artifacts)
├─ preprocessing.py      # NLTK setup + text preprocessing helpers
├─ requirements.txt      # Python dependencies
├─ spam_model.pkl        # Trained pipeline (generated)
├─ metrics.json          # Evaluation metrics (generated)
└─ README.md
```

## 🛠️ Troubleshooting
- “streamlit is not recognized”
  - Use `python -m streamlit run app.py` to avoid PATH issues
- PowerShell activation blocked
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - Then `. .\.venv\Scripts\Activate.ps1`
- Pickle AttributeError (preprocess function)
  - Already fixed by moving preprocessing into `preprocessing.py`. Retrain via `python model.py` if needed.
- FileNotFoundError: `spam_model.pkl` not found
  - Run `python model.py` once to generate artifacts

## 🧪 Re‑training With Different Random Seeds
```powershell
python - << 'PY'
from model import train_and_evaluate
print(train_and_evaluate(random_state=1337)["accuracy"])
PY
```

## 📄 License
This project is provided as‑is for educational purposes. Add your preferred license before publishing.

## 🙌 Credits
- Developed by Kalyani Dupare
- Powered by Streamlit & Machine Learning

