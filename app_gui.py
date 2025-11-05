import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import joblib, librosa, sounddevice as sd, wavio, os

# --- Config ---
MODEL_PATH = "models/ser_svm.joblib"
EMOJI = {
    "neutral":"😐", "calm":"😌", "happy":"😄", "sad":"😢",
    "angry":"😡", "fearful":"😨", "disgust":"🤢", "surprised":"😮"
}

bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
SR = bundle["sr"]; DUR = bundle["dur"]; N_MFCC = bundle["n_mfcc"]

# --- Functions ---
def extract_features(y, sr):
    target_len = int(SR * DUR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)

def record_to_wav(path="live.wav", duration=DUR, fs=SR):
    try:
        msg.set("Recording... Speak now 🎤")
        root.update_idletasks()
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavio.write(path, rec, fs, sampwidth=2)
        msg.set("Recording done! Predicting...")
        root.update_idletasks()
        return path
    except Exception as e:    
        msg.set("Recording done! Predicting...")
        root.update_idlestacks()
        messagebox.showerror("Mic Error", str(e))
        return None

def predict_from_wav(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    feats = extract_features(y, sr).reshape(1, -1)
    pred = pipe.predict(feats)[0]
    prob = pipe.predict_proba(feats)[0].max()
    return pred, float(prob)

def on_record_and_predict():
    wav_path = record_to_wav()
    if not wav_path: return
    label, p = predict_from_wav(wav_path)
    result.set(f"{label} {EMOJI.get(label,'')}")
    msg.set(f"Confidence: {p:.2f}")

def on_open_file():
    f = filedialog.askopenfilename(title="Choose a WAV file", filetypes=[("WAV files","*.wav")])
    if not f: return
    try:
        label, p = predict_from_wav(f)
        result.set(f"{label} {EMOJI.get(label,'')}")
        msg.set(f"Confidence: {p:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        result.set(f"Confidence: {p:.3f}")


# --- Tkinter UI ---
root = tk.Tk()
root.title("🎤 Live Emotion Detector")
root.geometry("450x380")
root.config(bg="#1e1e2f")

# Title
title = tk.Label(root, text="Emotion Detection from Speech", font=("Segoe UI", 18, "bold"), bg="#1e1e2f", fg="#ffffff")
title.pack(pady=10)

# Result frame
frame_res = tk.Frame(root, bg="#2e2e50", bd=2, relief="ridge")
frame_res.pack(pady=15, padx=20, fill="both")
result = tk.StringVar(value="—")
lbl_res = tk.Label(frame_res, textvariable=result, font=("Segoe UI", 28, "bold"), bg="#2e2e50", fg="#00ff00")
lbl_res.pack(pady=20)

# Buttons frame
frame_btn = tk.Frame(root, bg="#1e1e2f")
frame_btn.pack(pady=10)
btn_rec = tk.Button(frame_btn, text="🎤 Record 5s & Predict", command=on_record_and_predict, font=("Segoe UI", 12), bg="#ff6f61", fg="white", width=20)
btn_rec.grid(row=0, column=0, padx=10, pady=5)
btn_file = tk.Button(frame_btn, text="📁 Pick WAV & Predict", command=on_open_file, font=("Segoe UI", 12), bg="#4caf50", fg="white", width=20)
btn_file.grid(row=0, column=1, padx=10, pady=5)

# Status / message
msg = tk.StringVar(value="Ready.")
lbl_msg = tk.Label(root, textvariable=msg, font=("Segoe UI", 10), bg="#1e1e2f", fg="#ffffff")
lbl_msg.pack(pady=5)

# Progress bar for fun (not actual recording time)
progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
progress.pack(pady=10)

root.mainloop()
