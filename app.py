from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    groq_api_key: str
    groq_model_id: str
    system_prompt: str
    tts_engine: str
    gtts_lang: str
    piper_model_path: str
    piper_config_path: str


def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip(),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant for students. Keep replies short and clear."
        ).strip(),
        tts_engine=os.getenv("TTS_ENGINE", "gtts").strip().lower(),
        gtts_lang=os.getenv("GTTS_LANG", "en").strip(),
        piper_model_path=os.getenv("PIPER_MODEL_PATH", "").strip(),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", "").strip(),
    )

CFG = load_config()

# ── ASR ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size, device, compute_type, cpu_threads):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads)


def transcribe_wav_bytes(wav_bytes: bytes) -> Tuple[str, float]:
    model = get_whisper_model(CFG.whisper_model, CFG.whisper_device, CFG.whisper_compute_type, CFG.whisper_cpu_threads)
    tmp_path = None
    start_time = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        segments, _ = model.transcribe(tmp_path, language=CFG.whisper_language, beam_size=1, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        return text, time.time() - start_time
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# ── LLM ───────────────────────────────────────────────────────────────────────
def offline_demo_reply(user_text: str) -> Tuple[str, float]:
    if not user_text:
        return "I did not catch that. Please try again.", 0.0
    return (
        f"Offline demo mode. You said: {user_text}\n\n"
        "Add GROQ_API_KEY in .env to enable real AI replies."
    ), 0.0


def groq_chat_completion(messages: List[Dict[str, str]]) -> Tuple[str, float]:
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CFG.groq_api_key}", "Content-Type": "application/json"}
    payload = {"model": CFG.groq_model_id, "messages": messages, "temperature": 0.4, "max_tokens": 250}
    start_time = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip(), time.time() - start_time


def generate_reply(user_text: str, history: List[Dict[str, str]]) -> Tuple[str, float]:
    if not CFG.groq_api_key:
        return offline_demo_reply(user_text)
    trimmed = history[-6:] if len(history) > 6 else history
    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_text})
    try:
        return groq_chat_completion(messages)
    except Exception as e:
        return f"Could not reach Groq. Error: {e}", 0.0

# ── TTS ───────────────────────────────────────────────────────────────────────
def tts_to_audio_file(text: str) -> Tuple[bytes, str, str, float]:
    text = (text or "").strip() or "I do not have a response to speak."
    if CFG.tts_engine == "piper":
        return piper_tts(text)
    return gtts_tts(text)


def gtts_tts(text: str) -> Tuple[bytes, str, str, float]:
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out_path = f.name
    start_time = time.time()
    try:
        gTTS(text=text, lang=CFG.gtts_lang).save(out_path)
        with open(out_path, "rb") as rf:
            audio_bytes = rf.read()
        return audio_bytes, "audio/mpeg", "reply.mp3", time.time() - start_time
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


def piper_tts(text: str) -> Tuple[bytes, str, str, float]:
    import wave
    from piper import PiperVoice
    if not CFG.piper_model_path or not CFG.piper_config_path:
        return gtts_tts("Piper is not configured. Please set PIPER_MODEL_PATH and PIPER_CONFIG_PATH in .env.")
    voice = PiperVoice.load(CFG.piper_model_path, CFG.piper_config_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    start_time = time.time()
    try:
        with wave.open(out_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        with open(out_path, "rb") as rf:
            audio_bytes = rf.read()
        return audio_bytes, "audio/wav", "reply.wav", time.time() - start_time
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceBridge · AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════ GLOBALS ═══════════════════════ */
:root {
  --bg:        #0a0b0e;
  --surface:   #111318;
  --card:      #161a22;
  --border:    #242835;
  --amber:     #f5a623;
  --amber-dim: #7a5010;
  --teal:      #00d4b4;
  --teal-dim:  #004d42;
  --red:       #ff4b6e;
  --text:      #e8e9f0;
  --muted:     #5a6075;
  --mono:      'IBM Plex Mono', monospace;
  --display:   'Syne', sans-serif;
}



/* Root background */
.stApp {
  background: var(--bg) !important;
  font-family: var(--mono);
  color: var(--text);
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--amber-dim); border-radius: 2px; }

/* ═══════════════════════ SIDEBAR ═══════════════════════ */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

.exvv1vr0 {
    display: none;
}

.sidebar-logo {
  font-family: var(--display);
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--amber);
  letter-spacing: -0.02em;
  margin-bottom: 0.2rem;
}
.sidebar-tagline {
  color: var(--muted);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
}

/* Status indicator */
.status-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  background: var(--card);
  border: 1px solid var(--border);
  margin-bottom: 0.4rem;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot-online  { background: var(--teal);  box-shadow: 0 0 6px var(--teal); }
.dot-offline { background: var(--amber); box-shadow: 0 0 6px var(--amber);}
.dot-off     { background: var(--muted); }

/* Config code block */
.cfg-block {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  font-size: 0.68rem;
  color: var(--muted);
  line-height: 1.8;
  margin: 0.5rem 0 1rem;
}
.cfg-val { color: var(--amber); }

/* ═══════════════════════ HEADER ═══════════════════════ */
.hero {
  padding: 2.5rem 0 1.5rem;
  text-align: center;
  position: relative;
}
.hero-eyebrow {
  font-size: 0.65rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: var(--display);
  font-size: clamp(2.5rem, 5vw, 4rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--text);
  line-height: 1;
  margin-bottom: 0.4rem;
}
.hero-title span { color: var(--amber); }
.hero-subtitle {
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.08em;
}

/* Animated waveform decoration */
.waveform {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  height: 32px;
  margin: 1rem auto;
}
.waveform-bar {
  width: 3px;
  border-radius: 2px;
  background: var(--amber);
  animation: wave 1.2s ease-in-out infinite;
}
.waveform-bar:nth-child(1)  { height: 8px;  animation-delay: 0.0s; }
.waveform-bar:nth-child(2)  { height: 18px; animation-delay: 0.1s; }
.waveform-bar:nth-child(3)  { height: 28px; animation-delay: 0.2s; }
.waveform-bar:nth-child(4)  { height: 14px; animation-delay: 0.3s; }
.waveform-bar:nth-child(5)  { height: 22px; animation-delay: 0.4s; }
.waveform-bar:nth-child(6)  { height: 30px; animation-delay: 0.5s; }
.waveform-bar:nth-child(7)  { height: 18px; animation-delay: 0.4s; }
.waveform-bar:nth-child(8)  { height: 26px; animation-delay: 0.3s; }
.waveform-bar:nth-child(9)  { height: 12px; animation-delay: 0.2s; }
.waveform-bar:nth-child(10) { height: 20px; animation-delay: 0.1s; }
.waveform-bar:nth-child(11) { height: 8px;  animation-delay: 0.0s; }
@keyframes wave {
  0%, 100% { opacity: 0.3; transform: scaleY(0.5); }
  50%       { opacity: 1.0; transform: scaleY(1.0); }
}

/* ═══════════════════════ RECORDER PANEL ═══════════════════════ */
.recorder-panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 1rem 0;
  position: relative;
  overflow: hidden;
}
.recorder-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--amber), transparent);
}
.recorder-label {
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.pulse-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--red);
  box-shadow: 0 0 0 0 rgba(255,75,110,0.4);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0   rgba(255,75,110,0.4); }
  70%  { box-shadow: 0 0 0 8px rgba(255,75,110,0.0); }
  100% { box-shadow: 0 0 0 0   rgba(255,75,110,0.0); }
}

/* ═══════════════════════ METRIC CARDS ═══════════════════════ */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 1.25rem 0;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--amber-dim); }
.metric-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--amber);
  opacity: 0.4;
}
.metric-icon { font-size: 1rem; margin-bottom: 0.25rem; }
.metric-label {
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.25rem;
}
.metric-value {
  font-size: 1.4rem;
  font-weight: 500;
  color: var(--amber);
  letter-spacing: -0.02em;
}

/* ═══════════════════════ CHAT BUBBLES ═══════════════════════ */
.chat-wrap {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem 0;
}
.bubble {
  max-width: 78%;
  padding: 0.75rem 1.1rem;
  border-radius: 12px;
  font-size: 0.82rem;
  line-height: 1.55;
  position: relative;
}
.bubble-user {
  align-self: flex-end;
  background: linear-gradient(135deg, #1a1f2e, #1e243a);
  border: 1px solid #2a3050;
  border-bottom-right-radius: 3px;
  color: #c8d0f0;
}
.bubble-user::before {
  content: 'YOU';
  font-size: 0.55rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  display: block;
  margin-bottom: 0.3rem;
}
.bubble-assistant {
  align-self: flex-start;
  background: linear-gradient(135deg, #101a18, #14201e);
  border: 1px solid #1a3530;
  border-bottom-left-radius: 3px;
  color: #b8e8e0;
}
.bubble-assistant::before {
  content: 'VOICEBRIDGE';
  font-size: 0.55rem;
  letter-spacing: 0.18em;
  color: var(--teal);
  display: block;
  margin-bottom: 0.3rem;
}

/* ═══════════════════════ SECTION HEADERS ═══════════════════════ */
.section-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 1.5rem 0 0.75rem;
}
.section-icon {
  font-size: 0.9rem;
  background: var(--card);
  border: 1px solid var(--border);
  width: 32px; height: 32px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
}
.section-title {
  font-family: var(--display);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text);
}
.section-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ═══════════════════════ AUDIO PLAYER ═══════════════════════ */
.audio-panel {
  background: var(--card);
  border: 1px solid var(--teal-dim);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin: 0.75rem 0;
  position: relative;
  overflow: hidden;
}
.audio-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
}
.audio-panel-label {
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 0.75rem;
}

/* ═══════════════════════ DIVIDER ═══════════════════════ */
.fancy-divider {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 1.5rem 0;
}

/* ═══════════════════════ OVERRIDE STREAMLIT ELEMENTS ═══════════════════════ */
/* Audio widget */
audio {
  width: 100%;
  height: 36px;
  border-radius: 6px;
  filter: hue-rotate(190deg) saturate(0.6);
}

/* Buttons */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  padding: 0.55rem 1rem !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  color: var(--text) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber) !important;
  background: rgba(245,166,35,0.05) !important;
}
[data-testid="baseButton-primary"] > button,
.stButton > button[kind="primary"] {
  background: var(--amber) !important;
  color: #0a0b0e !important;
  border-color: var(--amber) !important;
  font-weight: 500 !important;
}
[data-testid="baseButton-primary"] > button:hover,
.stButton > button[kind="primary"]:hover {
  background: #e09510 !important;
  color: #0a0b0e !important;
}

/* Download button */
.stDownloadButton > button {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  border-radius: 8px !important;
  background: rgba(0,212,180,0.08) !important;
  border: 1px solid var(--teal-dim) !important;
  color: var(--teal) !important;
}

/* Spinner */
.stSpinner > div { border-color: var(--amber) transparent transparent transparent !important; }

/* Alert / warning / error */
.stAlert {
  background: var(--card) !important;
  border-radius: 8px !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}
.stSuccess { border-left: 3px solid var(--teal) !important; }
.stError   { border-left: 3px solid var(--red)  !important; }
.stWarning { border-left: 3px solid var(--amber) !important; }

/* JSON / code blocks */
.stJson, .stCode {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
}

/* Checkbox */
.stCheckbox label span { font-family: var(--mono) !important; font-size: 0.75rem !important; }

/* Audio input widget */
[data-testid="stAudioInput"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Metrics (override default streamlit metric) */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.3rem !important;
  color: var(--amber) !important;
}

/* hr override */
hr { border-color: var(--border) !important; }

/* sidebar heading */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: var(--display) !important;
  color: var(--text) !important;
}

/* Expander */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("chat_history", []),
    ("latencies", {}),
    ("last_audio", None),
    ("recorder_key", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">VoiceBridge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Speech · AI · Speech System</div>', unsafe_allow_html=True)

    # Live status indicators
    groq_status = ("dot-online", "GROQ API · ONLINE") if CFG.groq_api_key else ("dot-offline", "GROQ API · OFFLINE")
    whisper_status = ("dot-online", f"WHISPER · {CFG.whisper_model.upper()}")
    tts_status = ("dot-online", f"TTS ENGINE · {CFG.tts_engine.upper()}")

    for cls, label in [groq_status, whisper_status, tts_status]:
        st.markdown(f"""
        <div class="status-row">
          <div class="status-dot {cls}"></div>
          <span style="color:var(--muted);letter-spacing:0.08em">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.25rem;font-size:0.6rem;letter-spacing:0.2em;color:var(--amber);text-transform:uppercase;margin-bottom:0.5rem;">System Config</div>', unsafe_allow_html=True)

    cfg_lines = [
        ("MODEL", CFG.whisper_model),
        ("LANG", CFG.whisper_language),
        ("DEVICE", CFG.whisper_device),
        ("COMPUTE", CFG.whisper_compute_type),
        ("THREADS", str(CFG.whisper_cpu_threads)),
        ("GROQ_ID", CFG.groq_model_id),
        ("TTS", CFG.tts_engine),
    ]
    cfg_html = '<div class="cfg-block">'
    for k2, v2 in cfg_lines:
        cfg_html += f'<div>{k2} <span class="cfg-val">→ {v2}</span></div>'
    cfg_html += '</div>'
    st.markdown(cfg_html, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.62rem;color:var(--muted);margin-bottom:0.75rem;">💡 Low RAM? Use <span style="color:var(--amber)">tiny.en</span> Whisper model.</div>', unsafe_allow_html=True)

    if st.button("⬛  Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.latencies = {}
        st.success("Conversation cleared.")

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">● Live · Real-Time Voice AI</div>
  <div class="hero-title">Voice<span>Bridge</span></div>
  <div class="hero-subtitle">Speech Recognition &nbsp;·&nbsp; Language Model &nbsp;·&nbsp; Text-to-Speech</div>
  <div class="waveform">
    <div class="waveform-bar"></div><div class="waveform-bar"></div>
    <div class="waveform-bar"></div><div class="waveform-bar"></div>
    <div class="waveform-bar"></div><div class="waveform-bar"></div>
    <div class="waveform-bar"></div><div class="waveform-bar"></div>
    <div class="waveform-bar"></div><div class="waveform-bar"></div>
    <div class="waveform-bar"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Recorder Panel ────────────────────────────────────────────────────────────
st.markdown("""
<div class="recorder-panel">
  <div class="recorder-label">
    <div class="pulse-dot"></div>
    Input · Voice Capture
  </div>
</div>
""", unsafe_allow_html=True)

audio_value = st.audio_input(
    "🎤  Hold to record — release to process",
    key=f"recorder_{st.session_state.recorder_key}",
    label_visibility="visible",
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if audio_value and audio_value != st.session_state.last_audio:
    st.session_state.last_audio = audio_value

    # Playback of captured audio
    st.markdown('<div class="section-header"><div class="section-icon">▶</div><div class="section-title">Your Recording</div><div class="section-line"></div></div>', unsafe_allow_html=True)
    st.audio(audio_value, format="audio/wav")

    wav_bytes = audio_value.getvalue()

    if len(wav_bytes) > 4000:
        with st.spinner("Processing pipeline…"):

            with st.spinner("① Transcribing speech…"):
                transcript, asr_latency = transcribe_wav_bytes(wav_bytes)

            if not transcript or not transcript.strip():
                st.markdown('<div style="background:var(--card);border:1px solid var(--red);border-radius:10px;padding:0.9rem 1.2rem;font-size:0.78rem;color:var(--red);margin:0.75rem 0;">⚠ No speech detected — please try in a quieter environment.</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄  Record Again", use_container_width=True, key="ra_err"):
                        st.session_state.last_audio = None
                        st.session_state.recorder_key += 1
                        st.rerun()
                with col2:
                    if st.button("⬛  Clear", use_container_width=True, key="cl_err"):
                        st.session_state.chat_history = []
                        st.session_state.latencies = {}
                        st.session_state.last_audio = None
                        st.session_state.recorder_key += 1
                        st.rerun()
            else:
                with st.spinner("② Generating AI reply…"):
                    reply_text, llm_latency = generate_reply(transcript, st.session_state.chat_history)

                with st.spinner("③ Synthesising speech…"):
                    audio_bytes, mime, fname, tts_latency = tts_to_audio_file(reply_text)

                # Update history
                st.session_state.chat_history.append({"role": "user", "content": transcript})
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                total = asr_latency + llm_latency + tts_latency
                st.session_state.latencies = {
                    "ASR": f"{asr_latency:.2f}s",
                    "LLM": f"{llm_latency:.2f}s",
                    "TTS": f"{tts_latency:.2f}s",
                    "Total": f"{total:.2f}s",
                }

                # ── Latency Metrics ──────────────────────────────────────────
                st.markdown('<div class="section-header"><div class="section-icon">⚡</div><div class="section-title">Latency</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎙 ASR", st.session_state.latencies["ASR"])
                with col2:
                    st.metric("🤖 LLM", st.session_state.latencies["LLM"])
                with col3:
                    st.metric("🔊 TTS", st.session_state.latencies["TTS"])
                with col4:
                    st.metric("⏱ Total", st.session_state.latencies["Total"])

                # ── Conversation ─────────────────────────────────────────────
                st.markdown('<div class="section-header"><div class="section-icon">💬</div><div class="section-title">Conversation Log</div><div class="section-line"></div></div>', unsafe_allow_html=True)

                bubbles_html = '<div class="chat-wrap">'
                for msg in st.session_state.chat_history:
                    cls = "bubble-user" if msg["role"] == "user" else "bubble-assistant"
                    content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                    bubbles_html += f'<div class="bubble {cls}">{content}</div>'
                bubbles_html += '</div>'
                st.markdown(bubbles_html, unsafe_allow_html=True)

                # ── AI Voice Response ────────────────────────────────────────
                st.markdown('<div class="section-header"><div class="section-icon">🔊</div><div class="section-title">AI Voice Response</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="audio-panel"><div class="audio-panel-label">▶ Synthesised Reply</div>', unsafe_allow_html=True)
                st.audio(audio_bytes, format=mime)
                st.markdown('</div>', unsafe_allow_html=True)

                st.download_button(
                    "⬇  Download Audio",
                    data=audio_bytes,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )

                # ── Action Buttons ───────────────────────────────────────────
                st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎤  Record Again", use_container_width=True, type="primary", key="ra_ok"):
                        st.session_state.last_audio = None
                        st.session_state.recorder_key += 1
                        st.rerun()
                with col2:
                    if st.button("⬛  New Session", use_container_width=True, key="nc_ok"):
                        st.session_state.chat_history = []
                        st.session_state.latencies = {}
                        st.session_state.last_audio = None
                        st.session_state.recorder_key += 1
                        st.rerun()
    else:
        st.markdown('<div style="background:var(--card);border:1px solid var(--amber-dim);border-radius:10px;padding:0.9rem 1.2rem;font-size:0.78rem;color:var(--amber);margin:0.75rem 0;">⚠ Recording too short — please speak for at least 0.2 seconds.</div>', unsafe_allow_html=True)
        if st.button("🔄  Try Again", use_container_width=True, key="ra_short"):
            st.session_state.last_audio = None
            st.session_state.recorder_key += 1
            st.rerun()

# ── Debug Section ─────────────────────────────────────────────────────────────
st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-icon">🔍</div><div class="section-title">Debug Panel</div><div class="section-line"></div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.checkbox("Show chat history (JSON)"):
        st.json(st.session_state.chat_history)
with col2:
    if st.checkbox("Show latency data (JSON)"):
        st.json(st.session_state.latencies)

# Footer
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;font-size:0.6rem;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;">
  VoiceBridge · Faster-Whisper · Groq LLaMA · gTTS / Piper
</div>
""", unsafe_allow_html=True)
