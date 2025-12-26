# =========================================================
# DYSARTHRIA DETECTION BACKEND (FINAL – XAI ENABLED + PDF + PATIENT INFO)
# =========================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from datetime import datetime
import uuid
import os
import uvicorn
from utils import extract_features  # SAME extract_features as training

# =========================================================
# APP INITIALIZATION
# =========================================================
app = FastAPI(title="Dysarthria Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "dysarthria_model.keras"),
    compile=False
)

# =========================================================
# LOAD NORMAL BASELINE
# =========================================================
baseline_path = os.path.join(BASE_DIR, "normal_baseline.npz")
if not os.path.exists(baseline_path):
    raise RuntimeError("❌ normal_baseline.npz not found")

baseline = np.load(baseline_path)
NORMAL_BASELINE = {"mean": baseline["mean"], "std": baseline["std"]}

# =========================================================
# FEATURE GROUPS
# =========================================================
FEATURE_GROUPS = {
    "mfcc": slice(0, 20),
    "delta": slice(20, 40),
    "delta2": slice(40, 60),
    "zcr": slice(60, 61),
    "rms": slice(61, 62)
}

# =========================================================
# LABEL LOGIC
# =========================================================
def get_prediction_label(prob):
    if prob > 0.6:
        return "DYSARTHRIC"
    elif prob < 0.4:
        return "NORMAL"
    else:
        return "UNCERTAIN"

# =========================================================
# XAI
# =========================================================
def explain_prediction(features, prob):
    explanation, reasons = {}, []

    for group, idx in FEATURE_GROUPS.items():
        dev = np.mean(abs((features[:, idx] - NORMAL_BASELINE["mean"][idx]) / NORMAL_BASELINE["std"][idx]))
        explanation[group] = "High deviation" if dev > 1.2 else "Moderate deviation" if dev > 0.7 else "Normal"

    label = get_prediction_label(prob)

    if label == "DYSARTHRIC":
        reasons += ["Articulation clarity affected", "Irregular speech pauses", "Abnormal loudness control"]
    elif label == "UNCERTAIN":
        reasons.append("Borderline speech patterns detected — clinical review advised")
    else:
        reasons.append("Speech patterns within healthy range")

    return explanation, reasons

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
def root():
    return {"status": "API running"}

# =========================================================
# PREDICTION
# =========================================================
@app.post("/predict")
async def predict_dysarthria(file: UploadFile = File(...)):

    raw_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    wav_path = raw_path.rsplit(".", 1)[0] + ".wav"

    try:
        with open(raw_path, "wb") as f:
            f.write(await file.read())

        y, sr = librosa.load(raw_path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) < sr * 2:
            raise HTTPException(status_code=400, detail="Audio too short")

        sf.write(wav_path, y, sr, subtype="PCM_16")

        features = extract_features(wav_path)
        prob = float(model.predict(np.expand_dims(features, 0))[0][0])

        explanation, reasons = explain_prediction(features, prob)

        return {
            "prediction": get_prediction_label(prob),
            "confidence_percent": round(prob * 100, 2),
            "feature_analysis": explanation,
            "reasons": reasons
        }

    finally:
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)

# =========================================================
# PDF GENERATION — MEDICAL PRESCRIPTION STYLE
# =========================================================
@app.post("/generate-pdf")
def generate_prescription_pdf(data: dict):

    file_id = uuid.uuid4().hex
    pdf_path = os.path.join(UPLOAD_DIR, f"Prescription_{file_id}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # ================= HEADER =================
    c.setFillColorRGB(0.05, 0.3, 0.55)
    c.rect(0, height - 110, width, 110, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(50, height - 60, "AI SPEECH CLINIC")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 85, "Neurological Speech Screening Report")

    # ================= PATIENT INFO =================
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 140, "Patient:")
    c.drawString(300, height - 140, "Report ID:")

    c.setFont("Helvetica", 12)
    # ✅ Display actual patient info
    patient_name = data.get("name", "Voice Sample User")
    patient_age = data.get("age", "N/A")
    patient_gender = data.get("gender", "N/A")
    c.drawString(120, height - 140, f"{patient_name}, {patient_age} yrs, {patient_gender}")
    c.drawString(380, height - 140, file_id[:8].upper())
    c.drawString(300, height - 165, "Date:")
    c.drawString(380, height - 165, datetime.now().strftime("%d %b %Y"))

    # ================= DIAGNOSIS BOX =================
    is_normal = data["prediction"] == "NORMAL"
    box_color = (0.0, 0.6, 0.2) if is_normal else (0.75, 0.1, 0.1)

    c.setStrokeColorRGB(*box_color)
    c.setLineWidth(2)
    c.roundRect(50, height - 270, width - 100, 100, 12)

    c.setFillColorRGB(*box_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(60, height - 215, "SCREENING RESULT:")

    c.setFont("Helvetica-Bold", 22)
    c.drawString(260, height - 215, data["prediction"])

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    c.drawString(60, height - 245, f"Confidence Level: {data['confidence_percent']}%")

    # ================= CLINICAL ANALYSIS =================
    y = height - 320
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, y, "Clinical Interpretation")
    y -= 25

    c.setFont("Helvetica", 12)

    if is_normal:
        c.drawString(70, y, "• Speech parameters are within normal clinical limits")
        y -= 18
        c.drawString(70, y, "• No dysarthric motor speech patterns detected")
        y -= 18
        c.drawString(70, y, "• No immediate medical or therapeutic intervention required")
    else:
        c.drawString(70, y, "• Deviations suggest impaired motor speech coordination")
        y -= 18
        c.drawString(70, y, "• Findings consistent with dysarthric speech characteristics")
        y -= 18
        c.drawString(70, y, "• Early clinical evaluation recommended")

    # ================= TREATMENT =================
    y -= 40
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, y, "Treatment & Recommendations")
    y -= 25
    c.setFont("Helvetica", 12)

    if is_normal:
        c.drawString(70, y, "• No treatment required at this stage")
        y -= 18
        c.drawString(70, y, "• Maintain healthy vocal habits and hydration")
        y -= 18
        c.drawString(70, y, "• Seek evaluation if new speech symptoms appear")
    else:
        c.drawString(70, y, "• Comprehensive speech-language pathology evaluation advised")
        y -= 18
        c.drawString(70, y, "• Speech therapy may be initiated based on clinical findings")
        y -= 18
        c.drawString(70, y, "• Neurological consultation recommended for cause assessment")

    # ================= SIGNATURE =================
    c.line(50, 140, 250, 140)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 120, "Dr. AI NeuroSpeech")
    c.setFont("Helvetica", 11)
    c.drawString(50, 105, "Consultant – Speech & Neurology (AI-Assisted)")
    c.drawString(50, 90, "Registration ID: AI-SLP-2025")

    # ================= DISCLAIMER =================
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 60, "Disclaimer: This report is generated using AI-based screening technology.")
    c.drawString(50, 45, "It is not a definitive medical diagnosis and should be reviewed by a qualified clinician.")

    c.showPage()
    c.save()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="AI_Dysarthria_Report.pdf"
    )
    if __name__ == "__main__":

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000))
        )