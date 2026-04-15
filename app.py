# -*- coding: utf-8 -*-
"""app.ipynb

"""

import os, cv2, time, datetime, tempfile, smtplib, queue, threading, gc
from email.mime.text import MIMEText
from collections import deque

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
IMG_SIZE      = 160
FRAMES        = 20
LABELS        = {0: "Non-Violent", 1: "Violent"}
GRAPH_HISTORY = 60
CHUNK_FRAMES  = 300   # flush/GC every N frames (large-video stability)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_MAP = {
    "MobileNet_BiLSTM_Attention (95.5%)": os.path.join(BASE_DIR, "models/mobilenet_bilstm_attention.h5"),
    "CNN_LSTM_Attention (88%)": os.path.join(BASE_DIR, "models/cnn_lstm_attention.h5"),
}

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionGuard · Live",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLES  — deep navy / electric teal design system
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap');

/* ── CSS variables ── */
:root{
    --bg:        #050810;
    --surface:   #0b0f1a;
    --elevated:  #111827;
    --border:    #1e2d40;
    --border2:   #263447;
    --accent:    #06b6d4;
    --accent2:   #0ea5e9;
    --danger:    #f43f5e;
    --safe:      #10b981;
    --warn:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --muted2:    #94a3b8;
    --font-ui:   'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --font-disp: 'Syne', sans-serif;
    --radius:    10px;
    --radius-sm: 6px;
    --shadow:    0 4px 24px rgba(0,0,0,.5);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"]{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.skip-link {
    position: absolute;
    top: -40px; left: 0;
    background: var(--accent);
    color: #000;
    padding: 8px 16px;
    font-weight: 700;
    z-index: 9999;
    border-radius: 0 0 6px 0;
    text-decoration: none;
    transition: top .2s;
}
.skip-link:focus { top: 0; }

[data-testid="stSidebar"]{
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h4{
    font-family: var(--font-disp);
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted2) !important;
    margin: 16px 0 8px;
}

[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div{
    background: var(--elevated) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}

[data-testid="metric-container"]{
    background: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label{
    color: var(--muted) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: var(--font-mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-family: var(--font-disp) !important;
    font-size: 24px !important;
    color: var(--accent) !important;
    font-weight: 800 !important;
}

.stButton > button{
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 20px !important;
    transition: all .2s !important;
}
.stButton > button:hover{
    background: rgba(6,182,212,.12) !important;
    box-shadow: 0 0 20px rgba(6,182,212,.25) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stProgress"] > div > div{
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

[data-testid="stFileUploader"]{
    background: var(--elevated) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
}

::-webkit-scrollbar{ width: 6px; height: 6px; }
::-webkit-scrollbar-track{ background: var(--surface); }
::-webkit-scrollbar-thumb{ background: var(--border2); border-radius: 3px; }

@keyframes pulse{
    0%, 100% { opacity: 1; }
    50%       { opacity: .35; }
}
.pulse { animation: pulse 1.2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

st.markdown('<a class="skip-link" href="#main-content">Skip to content</a>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────────────────────

def card(title, body, color="#06b6d4", icon=""):
    st.markdown(f"""
    <div role="region" aria-label="{title}" style="
        background: var(--elevated);
        border: 1px solid {color}22;
        border-left: 3px solid {color};
        border-radius: var(--radius);
        padding: 14px 18px;
        margin-bottom: 10px;">
        <div style="font-family: var(--font-mono); font-size: 10px; letter-spacing: 2.5px;
                    color: {color}; text-transform: uppercase; margin-bottom: 5px;">
            {icon} {title}
        </div>
        <div style="font-family: var(--font-mono); font-size: 14px; color: var(--text); font-weight:500;">
            {body}
        </div>
    </div>""", unsafe_allow_html=True)


def status_bar(prob, threshold, fc):
    is_danger = prob >= threshold
    color     = "#f43f5e" if is_danger else "#10b981"
    label     = "⚠ VIOLENCE DETECTED" if is_danger else "✓ CLEAR — MONITORING"
    pulse_cls = 'class="pulse"' if is_danger else ""
    pct       = int(prob * 100)
    aria_live = 'aria-live="assertive"' if is_danger else 'aria-live="polite"'

    st.markdown(f"""
    <div {aria_live} role="status"
         style="background: var(--elevated);
                border: 1px solid {color}44;
                border-radius: var(--radius);
                padding: 14px 22px;
                margin: 10px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 20px;
                box-shadow: {'0 0 30px ' + color + '22' if is_danger else 'none'};">
        <div {pulse_cls} style="
            font-family: var(--font-disp);
            font-size: 16px;
            font-weight: 800;
            color: {color};
            letter-spacing: 2px;
            min-width: 230px;">
            {label}
        </div>
        <div role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100"
             style="flex:1; background: #1e2d40; border-radius: 4px; height: 12px; overflow:hidden;">
            <div style="width:{pct}%; background: linear-gradient(90deg, {color}bb, {color});
                        height: 100%; border-radius: 4px;
                        transition: width .4s cubic-bezier(.4,0,.2,1);"></div>
        </div>
        <div style="font-family: var(--font-mono); font-size: 12px;
                    color: var(--muted2); min-width: 150px; text-align: right;">
            {prob:.1%} &nbsp;·&nbsp; frame&nbsp;{fc}
        </div>
    </div>""", unsafe_allow_html=True)


def alert_row(ts, prob, source):
    pct = f"{prob:.1%}"
    st.markdown(f"""
    <div role="listitem"
         style="background: rgba(244,63,94,.07);
                border: 1px solid #f43f5e33;
                border-left: 3px solid #f43f5e;
                border-radius: var(--radius-sm);
                padding: 10px 16px;
                margin-bottom: 6px;
                font-family: var(--font-mono);
                font-size: 12px;
                display: flex; gap: 14px; align-items: center;">
        <span style="color:#f43f5e; font-weight:700;">🚨 ALERT</span>
        <span style="color: var(--muted2);">{ts}</span>
        <span style="color: #f59e0b;">Conf: {pct}</span>
        <span style="color: var(--accent);">{source}</span>
    </div>""", unsafe_allow_html=True)


def section_header(text, sub=""):
    st.markdown(f"""
    <div style="margin: 6px 0 18px; padding-bottom: 10px; border-bottom: 1px solid var(--border);">
        <div style="font-family: var(--font-disp); font-size: 18px; font-weight: 800;
                    color: var(--text); letter-spacing: 1px;">{text}</div>
        {"<div style='font-family:var(--font-mono);font-size:11px;color:var(--muted);margin-top:3px;letter-spacing:1px;'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  ML HELPERS
# ─────────────────────────────────────────────────────────────

def preprocess_clip(data):
    r = tf.image.resize(data.astype("float32"), (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(preprocess_input(r.numpy()), axis=0)


def infer(model, buf):
    clip  = np.array(list(buf), dtype="float32")
    probs = model.predict(preprocess_clip(clip), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return idx, float(probs[1]), probs
    
from tensorflow.keras.layers import Layer


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
from tensorflow.keras.layers import InputLayer

class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        super().__init__(*args, **kwargs)


tf.keras.utils.get_custom_objects()["InputLayer"] = FixedInputLayer

from tensorflow.keras import mixed_precision

class FixedDTypePolicy:
    def __init__(self, name="float32"):
        self.name = name

    def __call__(self, *args, **kwargs):
        return mixed_precision.Policy(self.name)

tf.keras.utils.get_custom_objects()["DTypePolicy"] = FixedDTypePolicy


@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={
                "Attention": Attention
            }
        )
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()


def send_email(cfg, prob, ts):
    if not cfg.get("enabled") or not cfg.get("to"):
        return
    try:
        msg             = MIMEText(
            f"VisionGuard detected possible violence at {ts}.\n"
            f"Confidence: {prob:.1%}\n\nThis is an automated alert."
        )
        msg["Subject"]  = f"🚨 VisionGuard Alert — {ts}"
        msg["From"]     = cfg["user"]
        msg["To"]       = cfg["to"]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(cfg["user"], cfg["password"])
            s.sendmail(cfg["user"], cfg["to"], msg.as_string())
    except Exception as e:
        st.toast(f"Email error: {e}", icon="⚠️")


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in {
    "alerts":       [],
    "prob_history": [],
    "running":      False,
    "email_cfg":    {},
    "total_frames_processed": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px;">
        <div style="font-family: var(--font-disp); font-size: 26px; font-weight: 900;
                    letter-spacing: 3px; line-height: 1.1;">
            <span style="color: var(--text);">VISION</span><span style="color: var(--danger);">GUARD</span>
        </div>
        <div style="font-family: var(--font-mono); font-size: 9px; color: var(--muted);
                    letter-spacing: 3px; margin-top: 4px; text-transform: uppercase;">
            AI · Real-Time · Surveillance
        </div>
        <div style="width: 40px; height: 2px; background: linear-gradient(90deg, var(--accent), var(--danger));
                    margin: 10px auto 0; border-radius: 1px;"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_PATH_MAP.keys()),
        label_visibility="visible",
        help="Higher accuracy models may be slower on large videos"
    )
    acc  = "95.5%" if "MobileNet" in model_choice else "88%"
    arch = "MobileNetV2 → BiLSTM → Attention" if "MobileNet" in model_choice else "CNN → LSTM → Attention"
    card("Architecture", arch, icon="🧠")
    card("Val Accuracy", acc, color="#10b981", icon="📊")

    st.markdown("---")
    st.markdown("#### ⚙️ Detection Settings")

    threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.99, value=0.50, step=0.01)
    infer_every = st.slider("Run inference every N frames", min_value=3, max_value=30, value=8, step=1)
    cooldown_sec = st.slider("Alert cooldown (seconds)", min_value=1, max_value=60, value=5, step=1)

    st.markdown("---")
    st.markdown("#### ⚡ Performance")
    max_dim = st.select_slider(
        "Max frame resolution",
        options=["360p", "480p", "720p", "1080p"],
        value="480p",
    )
    res_map = {"360p": 640, "480p": 854, "720p": 1280, "1080p": 1920}
    max_width = res_map[max_dim]

    skip_frames = st.checkbox("Smart frame skipping", value=True)

    st.markdown("---")
    st.markdown("#### 📡 Video Source")
    source_mode = st.radio("Input Source", ["🎥 Video File", "🔗 RTSP / IP Camera"])
    rtsp_url = ""
    if "RTSP" in source_mode:
        rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:port/stream")

    st.markdown("---")
    with st.expander("📧 Email Alerts", expanded=False):
        st.markdown("<small style='color:var(--muted)'>Uses Gmail SMTP (SSL). Requires App Password.</small>", unsafe_allow_html=True)
        email_on  = st.checkbox("Enable email alerts", key="email_on")
        email_to  = st.text_input("Recipient email", placeholder="recipient@example.com")
        email_usr = st.text_input("Your Gmail address", placeholder="you@gmail.com")
        email_pwd = st.text_input("Gmail App Password", type="password")
        st.session_state.email_cfg = {
            "enabled": email_on, "to": email_to,
            "user": email_usr,   "password": email_pwd,
        }

    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("Alerts Logged",  len(st.session_state.alerts))
    col_b.metric("Total Inferences", len(st.session_state.prob_history))

    if st.button("🗑️  Clear Session Data", use_container_width=True):
        st.session_state.alerts                = []
        st.session_state.prob_history          = []
        st.session_state.total_frames_processed = 0
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="main-content">', unsafe_allow_html=True)
st.markdown("""
<header role="banner" style="
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;">
    <div>
        <h1 style="font-family: var(--font-disp); font-size: 34px; font-weight: 900;
                   color: var(--text); letter-spacing: 2px; margin: 0; line-height: 1;">
            VISION<span style="color: var(--danger);">GUARD</span>
        </h1>
        <p style="font-family: var(--font-mono); font-size: 11px; color: var(--muted);
                  letter-spacing: 3px; margin: 4px 0 0; text-transform: uppercase;">
            Real-Time AI Violence Detection System
        </p>
    </div>
    <div style="font-family: var(--font-mono); font-size: 11px; color: var(--muted); text-align:right;">
        <div style="color: var(--safe);">● SYSTEM ACTIVE</div>
    </div>
</header>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────

model_file = MODEL_PATH_MAP[model_choice]

if not os.path.isfile(model_file):
    st.error(f"❌ Model not found: {model_file}")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model_safe(path):
    try:
        model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"Attention": Attention}
        )
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model_safe(model_file)

st.markdown(f"""
<div role="status" style="
    background: rgba(16,185,129,.08);
    border: 1px solid #10b98133;
    border-radius: var(--radius-sm);
    padding: 9px 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--safe);
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 10px;">
    <span>✓</span>
    <span>Model loaded: <strong>{model_choice}</strong></span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab_live, tab_demo, tab_alerts, tab_analytics = st.tabs([
    "📡  Live Analysis",
    "🎬  Demo Mode",
    "🚨  Alert Log",
    "📈  Analytics",
])


# ═════════════════════════════════════════════════════════════
#  TAB 1 — LIVE ANALYSIS
# ═════════════════════════════════════════════════════════════
with tab_live:

    if "Video" in source_mode:
        section_header("Upload & Analyse Video", "Supports .mp4 · .avi · .mov")

        up = st.file_uploader(
            "Drop your video here or click to browse",
            type=["mp4", "avi", "mov"],
            help="Maximum recommended file size: 2 GB."
        )

        speed = st.select_slider(
            "Playback speed",
            options=["0.25×", "0.5×", "1×", "2×", "Max (no delay)"],
            value="1×"
        )
        spd_map = {"0.25×": 4.0, "0.5×": 2.0, "1×": 1.0, "2×": 0.5, "Max (no delay)": 0.0}
        spd     = spd_map[speed]

        col1, col2 = st.columns(2)
        start = col1.button("▶  Start Analysis", use_container_width=True, key="vid_start")
        stop  = col2.button("⏹  Stop",           use_container_width=True, key="vid_stop")

        if up and start:
            with st.spinner("Buffering video file…"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                WRITE_CHUNK = 8 * 1024 * 1024
                while True:
                    chunk = up.read(WRITE_CHUNK)
                    if not chunk:
                        break
                    tfile.write(chunk)
                tfile.flush()

            cap   = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 25
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = min(1.0, max_width / max(orig_w, 1))
            disp_w = int(orig_w * scale)
            disp_h = int(orig_h * scale)

            frame_ph   = st.empty()
            status_ph  = st.empty()
            prog_ph    = st.progress(0, text="Initialising…")
            metrics_ph = st.empty()

            buf        = deque(maxlen=FRAMES)
            fc         = 0
            last_prob  = 0.0
            last_pred  = 0
            last_alert = 0.0
            prev_gray  = None
            infer_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or stop:
                        break

                    fc += 1
                    st.session_state.total_frames_processed += 1

                    if scale < 1.0:
                        frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if skip_frames and prev_gray is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(prev_gray, gray).mean()
                        if diff < 1.5 and len(buf) == FRAMES:
                            prev_gray = gray
                            continue
                        prev_gray = gray
                    elif skip_frames:
                        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    buf.append(cv2.resize(rgb, (160, 160)))

                    if len(buf) == FRAMES and fc % infer_every == 0:
                        last_pred, last_prob, _ = infer(model, buf)
                        infer_count += 1
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.prob_history.append(last_prob)

                        now = time.time()
                        if last_pred == 1 and last_prob >= threshold:
                            if now - last_alert >= cooldown_sec:
                                last_alert = now
                                st.session_state.alerts.append(
                                    {"ts": ts, "prob": last_prob, "source": "Video"})
                                send_email(st.session_state.email_cfg, last_prob, ts)
                                st.toast(f"🚨 Violence detected — {last_prob:.1%}", icon="🚨")

                    is_danger = last_pred == 1 and last_prob >= threshold
                    c_cv      = (244, 63, 94) if is_danger else (16, 185, 129)
                    h, w      = rgb.shape[:2]

                    overlay = rgb.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 52), (11, 15, 26), -1)
                    cv2.addWeighted(overlay, 0.82, rgb, 0.18, 0, rgb)
                    cv2.putText(
                        rgb,
                        f"{'■ VIOLENCE' if is_danger else '● CLEAR'}  {last_prob:.1%}  |  {fc:,}/{total:,}",
                        (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, c_cv, 1, cv2.LINE_AA
                    )
                    if is_danger:
                        cv2.rectangle(rgb, (0, 0), (w - 1, h - 1), (244, 63, 94), 3)

                    if fc % max(1, infer_every // 2) == 0:
                        frame_ph.image(rgb, channels="RGB", use_container_width=True)
                        with status_ph.container():
                            status_bar(last_prob, threshold, fc)
                        with metrics_ph.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Violence Prob",  f"{last_prob:.2%}")
                            m2.metric("Status",         LABELS[last_pred])
                            m3.metric("Alerts Logged",  len(st.session_state.alerts))
                            m4.metric("Progress",       f"{fc/total:.0%}")
                        prog_ph.progress(
                            min(fc / total, 1.0),
                            text=f"Frame {fc:,}/{total:,}  ·  Inferences: {infer_count:,}  ·  Alerts: {len(st.session_state.alerts)}"
                        )

                    if fc % CHUNK_FRAMES == 0:
                        gc.collect()

                    if spd > 0:
                        time.sleep((1.0 / fps_v) * spd)

            finally:
                cap.release()
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
                gc.collect()

            prog_ph.progress(1.0, text="✅ Analysis complete")
            st.success(
                f"Finished analysing **{fc:,}** frames · "
                f"**{infer_count:,}** inferences · "
                f"**{len(st.session_state.alerts)}** alert(s) logged."
            )

    else:
        section_header("RTSP / IP Camera Stream", "Live camera feed analysis")

        if not rtsp_url:
            st.info("👈  Enter your RTSP URL in the sidebar to connect.", icon="📡")
        else:
            col1, col2 = st.columns(2)
            start = col1.button("▶  Connect & Analyse", use_container_width=True, key="rtsp_start")
            stop  = col2.button("⏹  Disconnect",         use_container_width=True, key="rtsp_stop")

            if start:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    st.error("❌ Cannot open RTSP stream — verify URL, credentials, and network.", icon="🔌")
                    st.stop()

                frame_ph   = st.empty()
                status_ph  = st.empty()
                metrics_ph = st.empty()

                buf        = deque(maxlen=FRAMES)
                fc         = 0
                last_prob  = 0.0
                last_pred  = 0
                last_alert = 0.0
                reconnects = 0

                while not stop:
                    ret, frame = cap.read()
                    if not ret:
                        reconnects += 1
                        st.warning(f"⚠️ Stream lost — reconnecting… (attempt {reconnects})")
                        time.sleep(2)
                        cap = cv2.VideoCapture(rtsp_url)
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buf.append(cv2.resize(rgb, (160, 160)))
                    fc += 1

                    if len(buf) == FRAMES and fc % infer_every == 0:
                        last_pred, last_prob, _ = infer(model, buf)
                        ts  = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.prob_history.append(last_prob)
                        now = time.time()
                        if last_pred == 1 and last_prob >= threshold:
                            if now - last_alert >= cooldown_sec:
                                last_alert = now
                                st.session_state.alerts.append(
                                    {"ts": ts, "prob": last_prob, "source": "RTSP"})
                                send_email(st.session_state.email_cfg, last_prob, ts)
                                st.toast(f"🚨 {last_prob:.1%} confidence!", icon="🚨")

                    is_danger = last_pred == 1 and last_prob >= threshold
                    c_cv      = (244, 63, 94) if is_danger else (16, 185, 129)
                    h, w      = rgb.shape[:2]

                    ov = rgb.copy()
                    cv2.rectangle(ov, (0, 0), (w, 52), (11, 15, 26), -1)
                    cv2.addWeighted(ov, 0.82, rgb, 0.18, 0, rgb)
                    cv2.putText(
                        rgb,
                        f"{'■ VIOLENCE' if is_danger else '● CLEAR'}  {last_prob:.1%}  |  "
                        f"{datetime.datetime.now().strftime('%H:%M:%S')}",
                        (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, c_cv, 1, cv2.LINE_AA
                    )
                    if is_danger:
                        cv2.rectangle(rgb, (0, 0), (w - 1, h - 1), (244, 63, 94), 3)

                    frame_ph.image(rgb, channels="RGB", use_container_width=True)
                    with status_ph.container():
                        status_bar(last_prob, threshold, fc)
                    with metrics_ph.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Violence Prob", f"{last_prob:.2%}")
                        m2.metric("Status",        LABELS[last_pred])
                        m3.metric("Total Alerts",  len(st.session_state.alerts))
                        m4.metric("Frame",         fc)

                cap.release()


# ═════════════════════════════════════════════════════════════
#  TAB 2 — DEMO MODE
# ═════════════════════════════════════════════════════════════
with tab_demo:
    section_header("Demo Mode — Simulated CCTV Feed", "Plays any video as a live camera with real-time detection overlay")

    demo_video = st.file_uploader(
        "Upload demo video",
        type=["mp4", "avi", "mov"],
        key="demo_upload"
    )

    col1, col2, col3 = st.columns(3)
    cam_name   = col1.text_input("Camera Label",   value="CAM-01 · Main Entrance")
    location   = col2.text_input("Location Label", value="Building A · Floor 1")
    loop_video = col3.checkbox("Loop continuously", value=True)

    d1, d2 = st.columns(2)
    demo_start = d1.button("🔴  Start Live Demo", use_container_width=True, key="demo_start")
    demo_stop  = d2.button("⏹  Stop",             use_container_width=True, key="demo_stop")

    if demo_video and demo_start:
        with st.spinner("Buffering demo video…"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            WRITE_CHUNK = 8 * 1024 * 1024
            while True:
                chunk = demo_video.read(WRITE_CHUNK)
                if not chunk:
                    break
                tfile.write(chunk)
            tfile.flush()

        vid_col, stat_col = st.columns([2, 1])

        with vid_col:
            frame_ph  = st.empty()
            status_ph = st.empty()

        with stat_col:
            st.markdown("##### 📊 Live Stats")
            metric_ph  = st.empty()
            st.markdown("##### 🚨 Recent Alerts")
            alerts_ph  = st.empty()
            st.markdown("##### 📈 Risk Graph")
            graph_ph   = st.empty()

        buf            = deque(maxlen=FRAMES)
        fc             = 0
        last_prob      = 0.0
        last_pred      = 0
        last_alert     = 0.0
        session_alerts = []
        prob_trace     = []

        try:
            while not demo_stop:
                cap   = cv2.VideoCapture(tfile.name)
                fps_v = cap.get(cv2.CAP_PROP_FPS) or 25

                while not demo_stop:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buf.append(cv2.resize(rgb, (160,160)))
                    fc += 1

                    if len(buf) == FRAMES and fc % infer_every == 0:
                        last_pred, last_prob, _ = infer(model, buf)
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        prob_trace.append(last_prob)
                        st.session_state.prob_history.append(last_prob)

                        now = time.time()
                        if last_pred == 1 and last_prob >= threshold:
                            if now - last_alert >= cooldown_sec:
                                last_alert = now
                                alert = {"ts": ts, "prob": last_prob, "source": cam_name}
                                session_alerts.append(alert)
                                st.session_state.alerts.append(alert)
                                send_email(st.session_state.email_cfg, last_prob, ts)
                                st.toast(f"🚨 {cam_name}: {last_prob:.1%}", icon="🚨")

                    disp      = rgb.copy()
                    h, w      = disp.shape[:2]
                    is_danger = last_pred == 1 and last_prob >= threshold
                    c_cv      = (244, 63, 94) if is_danger else (16, 185, 129)

                    ov = disp.copy()
                    cv2.rectangle(ov, (0, 0), (w, 52), (11, 15, 26), -1)
                    cv2.addWeighted(ov, 0.82, disp, 0.18, 0, disp)

                    cv2.circle(disp, (18, 26), 6, (244, 63, 94), -1)
                    cv2.putText(disp, "REC", (30, 32), cv2.FONT_HERSHEY_DUPLEX, 0.5, (244, 63, 94), 1)

                    ts_now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                    cv2.putText(disp, ts_now, (w - 230, 32), cv2.FONT_HERSHEY_DUPLEX, 0.45, (88, 96, 105), 1)

                    status_txt = "VIOLENCE DETECTED" if is_danger else "MONITORING"
                    cv2.putText(disp, status_txt, (12, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.6, c_cv, 2)
                    cv2.putText(disp, f"CONF: {last_prob:.1%}", (w - 155, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.5, c_cv, 1)

                    if is_danger:
                        cv2.rectangle(disp, (0, 0), (w - 1, h - 1), (244, 63, 94), 4)

                    frame_ph.image(disp, channels="RGB", use_container_width=True)
                    with status_ph.container():
                        status_bar(last_prob, threshold, fc)
                    with metric_ph.container():
                        r1, r2 = st.columns(2)
                        r1.metric("Violence",  f"{last_prob:.1%}")
                        r2.metric("Status",    "🚨 ALERT" if is_danger else "✅ SAFE")
                        r3, r4 = st.columns(2)
                        r3.metric("Alerts",    len(session_alerts))
                        r4.metric("Frame",     fc)

                    time.sleep(1.0 / fps_v)

                cap.release()
                if not loop_video:
                    break
        finally:
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
            gc.collect()

        st.success("Demo session ended.")


# ═════════════════════════════════════════════════════════════
#  TAB 3 — ALERT LOG
# ═════════════════════════════════════════════════════════════
with tab_alerts:
    section_header("Alert Log", "All violence detection events this session")

    if not st.session_state.alerts:
        st.markdown("""
        <div style="text-align: center; padding: 60px 0; color: var(--muted);">
            <div style="font-size: 40px; margin-bottom: 12px;">🟢</div>
            No alerts recorded yet.
        </div>""", unsafe_allow_html=True)
    else:
        total_alerts = len(st.session_state.alerts)
        avg_conf     = np.mean([a["prob"] for a in st.session_state.alerts])
        max_conf     = max(a["prob"] for a in st.session_state.alerts)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Alerts",    total_alerts)
        c2.metric("Avg Confidence",  f"{avg_conf:.1%}")
        c3.metric("Peak Confidence", f"{max_conf:.1%}")

        st.markdown("---")
        for a in reversed(st.session_state.alerts):
            alert_row(a["ts"], a["prob"], a["source"])

        st.markdown("---")
        import io, csv
        buf_io = io.StringIO()
        writer = csv.DictWriter(buf_io, fieldnames=["ts", "prob", "source"])
        writer.writeheader()
        writer.writerows(st.session_state.alerts)

        st.download_button(
            "⬇  Download Alerts as CSV",
            data=buf_io.getvalue(),
            file_name=f"visionguard_alerts_{datetime.date.today()}.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════
#  TAB 4 — ANALYTICS
# ═════════════════════════════════════════════════════════════
with tab_analytics:
    section_header("Analytics Dashboard", "Detection statistics and trends")

    ph = st.session_state.prob_history
    if len(ph) < 2:
        st.markdown("""
        <div style="text-align: center; padding: 60px 0; color: var(--muted);">
            <div style="font-size: 40px; margin-bottom: 12px;">📈</div>
            Run an analysis to populate charts.
        </div>""", unsafe_allow_html=True)
    else:
        probs = ph[-GRAPH_HISTORY:]
        xs    = list(range(len(probs)))
        flags = [1 if p >= threshold else 0 for p in probs]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Inferences",    len(probs))
        s2.metric("Average Prob",  f"{np.mean(probs):.2%}")
        s3.metric("Peak Prob",     f"{max(probs):.2%}")
        s4.metric("Alert Rate",    f"{sum(flags)/len(flags):.1%}")

        st.markdown("---")

        DARK_BG  = "#0b0f1a"
        PANEL_BG = "#111827"
        GRID_CLR = "#1e2d40"
        TEXT_CLR = "#94a3b8"
        ACCENT   = "#06b6d4"
        DANGER   = "#f43f5e"
        SAFE     = "#10b981"
        WARN     = "#f59e0b"

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG)
        fig.suptitle("VisionGuard · Analytics Overview", color="#e2e8f0",
                     fontfamily="monospace", fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout(pad=3.5)

        def style_ax(ax, title):
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, color="#e2e8f0", fontsize=11, fontfamily="monospace", pad=10)
            ax.tick_params(colors=TEXT_CLR, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_CLR)
            ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

        ax = axes[0, 0]
        style_ax(ax, "Violence Probability Timeline")
        ax.plot(xs, probs, color=ACCENT, linewidth=1.6, zorder=3)
        ax.fill_between(xs, probs, alpha=0.13, color=ACCENT)
        ax.axhline(threshold, color=WARN, linestyle="--", linewidth=1.1, label=f"Threshold ({threshold:.0%})")
        d_xs = [i for i, f in enumerate(flags) if f]
        d_ys = [probs[i] for i in d_xs]
        if d_xs:
            ax.scatter(d_xs, d_ys, color=DANGER, s=28, zorder=5, label="Detected events")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

        ax2 = axes[0, 1]
        style_ax(ax2, "Probability Distribution")
        n_bins = min(25, max(5, len(probs) // 4))
        ax2.hist(probs, bins=n_bins, color=ACCENT, alpha=0.75, edgecolor=DARK_BG, linewidth=0.5)
        ax2.axvline(threshold, color=WARN, linestyle="--", linewidth=1.1)

        ax3 = axes[1, 0]
        style_ax(ax3, "Rolling Alert Rate (window=10)")
        w       = 10
        rolling = [sum(flags[max(0, i - w):i + 1]) / min(i + 1, w) for i in range(len(flags))]
        ax3.plot(xs, rolling, color=DANGER, linewidth=1.6)
        ax3.fill_between(xs, rolling, alpha=0.13, color=DANGER)
        ax3.set_ylim(0, 1)

        ax4 = axes[1, 1]
        ax4.set_facecolor(PANEL_BG)
        ax4.set_title("Breakdown", color="#e2e8f0", fontsize=11, fontfamily="monospace", pad=10)
        n_viol = sum(flags)
        n_safe = len(flags) - n_viol
        ax4.pie(
            [max(n_safe, 0.001), max(n_viol, 0.001)],
            labels=["Safe", "Violent"],
            colors=[SAFE, DANGER],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 10},
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 2.5},
            pctdistance=0.75,
        )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<footer style="
    margin-top: 50px;
    border-top: 1px solid var(--border);
    padding-top: 16px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    text-align: center;
    letter-spacing: 2px;">
    VISIONGUARD &nbsp;·&nbsp; REAL-TIME AI SURVEILLANCE
    &nbsp;·&nbsp; TensorFlow + Streamlit
</footer>
</div>
""", unsafe_allow_html=True)



