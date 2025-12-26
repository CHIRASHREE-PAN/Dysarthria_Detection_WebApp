import React, { useState, useRef } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate, Link } from "react-router-dom";
import axios from "axios";
import "./App.css"; // CSS for gradients, glass cards, nav, transitions

const API_URL = "http://127.0.0.1:8000";

// ======================== Navbar ========================
function Navbar() {
  return (
    <nav className="navbar">
      <div className="nav-logo">🧠 AI Speech Clinic</div>
      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/check">Check</Link>
        <Link to="/result">Result</Link>
      </div>
    </nav>
  );
}

// ======================== Patient Info Page ========================
function PatientInfo({ setPatient }) {
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [language, setLanguage] = useState("");
  const navigate = useNavigate();

  const handleGo = () => {
    if (!name || !age || !gender || !language) return;
    setPatient({ name, age, gender, language });
    navigate("/check");
  };

  return (
    <div className="page-container">
      <div className="section">
        <h1 className="page-title">Welcome to AI Speech Clinic</h1>
        <p className="page-subtitle">Enter your details to start the assessment</p>
        <div className="card glass-card">
          <input type="text" placeholder="Name" value={name} onChange={e => setName(e.target.value)} className="input-field" />
          <input type="number" placeholder="Age" value={age} onChange={e => setAge(e.target.value)} className="input-field" />
          <select value={gender} onChange={e => setGender(e.target.value)} className="input-field">
            <option value="">Gender</option>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
          <select value={language} onChange={e => setLanguage(e.target.value)} className="input-field">
            <option value="">Language</option>
            <option>English</option>
            <option>Hindi</option>
            <option>Other</option>
          </select>
          <button className="btn-gradient" onClick={handleGo}>🚀 Start Assessment</button>
        </div>
      </div>
    </div>
  );
}

// ======================== Audio Check Page ========================
function AudioCheck({ patient, setResult }) {
  const [audioFile, setAudioFile] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);

  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setAudioFile(file);
    setRecordedBlob(null);
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!audioFile) {
      setError("Please upload or record audio");
      return;
    }
    const formData = new FormData();
    formData.append("file", audioFile);

    try {
      setLoading(true);
      setError("");
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      navigate("/result");
    } catch (err) {
      if (err.response) setError(err.response.data?.detail || "Backend error");
      else setError("Backend not reachable. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new AudioContext({ sampleRate: 16000 });
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        const length = audioBuffer.length * 2 + 44;
        const buffer = new ArrayBuffer(length);
        const view = new DataView(buffer);
        let offset = 0;
        const writeString = (s) => { for (let i=0;i<s.length;i++) view.setUint8(offset++, s.charCodeAt(i)); }
        writeString("RIFF");
        view.setUint32(offset, 36 + audioBuffer.length*2, true); offset+=4;
        writeString("WAVEfmt "); view.setUint32(offset,16,true); offset+=4;
        view.setUint16(offset,1,true); offset+=2;
        view.setUint16(offset,1,true); offset+=2;
        view.setUint32(offset,16000,true); offset+=4;
        view.setUint32(offset,16000*2,true); offset+=4;
        view.setUint16(offset,2,true); offset+=2;
        view.setUint16(offset,16,true); offset+=2;
        writeString("data"); view.setUint32(offset,audioBuffer.length*2,true); offset+=4;
        const channelData = audioBuffer.getChannelData(0);
        for(let i=0;i<channelData.length;i++){let sample=Math.max(-1,Math.min(1,channelData[i]));view.setInt16(offset,sample*0x7fff,true);offset+=2;}
        const wavBlob = new Blob([view], { type: "audio/wav" });
        setRecordedBlob(wavBlob);
        setAudioFile(new File([wavBlob], "recorded.wav", { type: "audio/wav" }));
      };

      mediaRecorder.start();
      setRecording(true);
    } catch { setError("Microphone permission denied"); }
  };
  const stopRecording = () => { mediaRecorderRef.current.stop(); setRecording(false); };

  return (
    <div className="page-container">
      <div className="section">
        <h1 className="page-title">🗣️Dysarthria Detection- 🎤 Audio Check</h1>
        <p className="page-subtitle">
          Patient: <b>{patient.name}</b> | Age: {patient.age} | Gender: {patient.gender} | Language: {patient.language}
        </p>
        <div className="card glass-card">
          <label className="file-upload">
            🎵 {audioFile ? audioFile.name : "Choose WAV File"}
            <input type="file" accept=".wav" onChange={handleFileChange} ref={fileInputRef} />
          </label>

          <button className={`btn-gradient ${recording?"btn-recording":""}`} onClick={recording?stopRecording:startRecording}>
            {recording ? "⏹ Stop Recording" : "🎤 Tap to Record"}
          </button>

          {recordedBlob && <audio controls className="mt-4 w-full" src={URL.createObjectURL(recordedBlob)} />}

          <button className="btn-gradient mt-4" onClick={handlePredict} disabled={loading}>
            {loading ? "Analyzing..." : "🚀 Predict"}
          </button>

          {error && <p className="error-text">❌ {error}</p>}
        </div>
      </div>
    </div>
  );
}

// ======================== Result Page ========================
function ResultPage({ patient, result }) {
  const handleDownloadPDF = async () => {
    if (!result) return;
    try {
      // ✅ Include patient info in the PDF request
      const pdfPayload = {
        ...result,
        name: patient.name,
        age: patient.age,
        gender: patient.gender
      };

      const response = await axios.post(`${API_URL}/generate-pdf`, pdfPayload, { responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "AI_Dysarthria_Prescription.pdf");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {}
  };

  if (!result) return <p className="page-container">No result available</p>;

  const isNormal = result.prediction.toLowerCase().includes("normal");

  return (
    <div className="page-container">
      <div className="section">
        <h1 className="page-title">🧠 Diagnosis Result</h1>
        <p className="page-subtitle">
          Patient: <b>{patient.name}</b> | Age: {patient.age} | Gender: {patient.gender} | Language: {patient.language}
        </p>
        <div className={`card glass-card ${isNormal?"normal":"abnormal"}`}>
          <p className="result-label">{isNormal ? "✔ Speech appears NORMAL" : "⚠ Possible DYSARTHRIA detected"}</p>
          <p className="result-confidence">Confidence: <b>{result.confidence_percent}%</b></p>

          <div className="explain-panel">
            <h3>🧠 Explainable AI Analysis</h3>
            <ul>
              {Object.entries(result.feature_analysis).map(([key, value]) => (
                <li key={key}><b>{key.toUpperCase()}</b>: {value}</li>
              ))}
            </ul>
            <h3>📌 Reasons</h3>
            <ul>{result.reasons.map((r,idx)=><li key={idx}>{r}</li>)}</ul>
          </div>

          <button className="btn-gradient mt-4" onClick={handleDownloadPDF}>
            📄 Download Prescription (PDF)
          </button>
        </div>
      </div>
    </div>
  );
}

// ======================== Main App ========================
function App() {
  const [patient, setPatient] = useState({});
  const [result, setResult] = useState(null);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<PatientInfo setPatient={setPatient} />} />
        <Route path="/check" element={<AudioCheck patient={patient} setResult={setResult} />} />
        <Route path="/result" element={<ResultPage patient={patient} result={result} />} />
      </Routes>
    </Router>
  );
}

export default App;