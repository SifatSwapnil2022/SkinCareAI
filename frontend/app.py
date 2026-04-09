import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from PIL import Image

import os
# API_URL = os.getenv("API_BASE", "http://backend:8000")             # docker
API_URL = os.getenv("API_BASE", "http://127.0.0.1:8001")             # local host

st.set_page_config(
    page_title="SkinAI — Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }
/* Slightly cleaner slate background to make white cards pop more */
.main { background: #F1F5F9; }
.block-container { padding: 2rem 2.5rem !important; }

/* Darker, richer sidebar for ultimate contrast */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #020617 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.15);
}
/* ALL sidebar text forced to pure white instead of gray for readability */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] strong {
    color: #FFFFFF !important;
}

/* Hero - More vibrant, saturated red-to-orange gradient */
.hero-wrap {
    background: linear-gradient(135deg, #B91C1C 0%, #DC2626 50%, #EA580C 100%);
    border-radius: 24px;
    padding: 48px 40px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(185, 28, 28, 0.40);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%);
}
.hero-wrap h1 { font-size: 3rem; font-weight: 800; margin: 0; letter-spacing: -1px; color: #ffffff !important; }
.hero-wrap p  { font-size: 1.15rem; color: #ffffff !important; opacity: 1; margin-top: 12px; font-weight: 500; }

/* Feature cards — white bg, near-black text for high contrast */
.feat-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    height: 100%;
    border-top: 4px solid #DC2626;
}
.feat-icon { font-size: 2.8rem; margin-bottom: 12px; }
.feat-card h3 { color: #111827 !important; font-size: 1rem; font-weight: 800; margin: 0 0 6px; }
.feat-card p  { color: #374151 !important; font-size: 0.85rem; margin: 0; line-height: 1.5; font-weight: 500;}

/* Metric cards — white bg, vivid accent value, pure dark label */
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    border-bottom: 4px solid #DC2626;
}
.metric-card .val { color: #B91C1C !important; font-size: 2.2rem; font-weight: 800; margin: 0; }
.metric-card .lbl { color: #111827 !important; margin: 4px 0 0; font-size: 0.85rem; font-weight: 700; }

/* Result card — white bg, high visibility dark text */
.result-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}
.result-card p { color: #111827 !important; font-weight: 500; }

.disease-badge {
    background: linear-gradient(135deg, #B91C1C, #EA580C);
    color: #ffffff !important;
    padding: 8px 22px;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 800;
    display: inline-block;
    margin: 8px 0;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(185, 28, 28, 0.3);
}

/* Confidence colours — Deepened for better readability on white */
.conf-high   { color: #065F46 !important; font-weight: 800; font-size: 1.5rem; }
.conf-medium { color: #92400E !important; font-weight: 800; font-size: 1.5rem; }
.conf-low    { color: #991B1B !important; font-weight: 800; font-size: 1.5rem; }

/* LLM sections — clearer teal bg, much darker text for readability */
.llm-block {
    background: #F0FDFA;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 10px 0;
    border-left: 6px solid #0D9488;
}
.llm-block .llm-title {
    color: #115E59 !important;
    font-weight: 800;
    font-size: 0.95rem;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.llm-block p { color: #0F172A !important; line-height: 1.75; margin: 0; font-size: 0.95rem; font-weight: 500;}

/* Severity badges — Highly visible text against tinted backgrounds */
.sev-high   { background:#FEE2E2; color:#991B1B !important; padding:5px 16px; border-radius:20px; font-weight:800; font-size:0.85rem; border: 1px solid #FCA5A5; }
.sev-medium { background:#FEF3C7; color:#92400E !important; padding:5px 16px; border-radius:20px; font-weight:800; font-size:0.85rem; border: 1px solid #FCD34D; }
.sev-low    { background:#DCFCE7; color:#166534 !important; padding:5px 16px; border-radius:20px; font-weight:800; font-size:0.85rem; border: 1px solid #86EFAC; }
.urgent-badge { background:#FEE2E2; color:#991B1B !important; padding:5px 16px; border-radius:20px; font-weight:800; font-size:0.85rem; margin-left:8px; border: 1px solid #FCA5A5; }

/* History rows — solid borders, dark text */
.hist-row {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #0D9488;
}
.hist-row b { color: #111827 !important; font-size: 0.95rem; font-weight: 800; }
.hist-row small { color: #374151 !important; font-weight: 500; }

/* Buttons - Bolder gradients and shadows */
.stButton > button {
    background: linear-gradient(135deg, #B91C1C, #EA580C) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    width: 100%;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(185, 28, 28, 0.45) !important;
}

/* Inputs - Darker border so the box is clearly visible */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #94A3B8 !important; 
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
    color: #0F172A !important;
    background: #ffffff !important;
    font-weight: 500 !important;
}
.stTextInput > div > div > input:focus { border-color: #DC2626 !important; box-shadow: 0 0 0 2px rgba(220,38,38,0.2) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 700 !important;
    color: #1F2937 !important;
}

/* Sidebar nav buttons */
.nav-btn > button {
    background: transparent !important;
    color: #E2E8F0 !important;
    text-align: left !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
    border: none !important;
    box-shadow: none !important;
}
.nav-btn-active > button {
    background: rgba(220, 38, 38, 0.25) !important;
    color: #FDA4AF !important;
    font-weight: 800 !important;
}

/* Disclaimer strip - Deeper yellow/orange contrast */
.disclaimer {
    background: #FEF9C3;
    border: 2px solid #CA8A04;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #854D0E !important;
    margin-top: 14px;
    font-weight: 600;
}

/* Page headings */
h2.page-title { color: #B91C1C !important; font-weight: 800; margin: 0; }
p.page-sub   { color: #374151 !important; margin: 4px 0 0; font-weight: 500; }

/* Auth card heading */
.auth-title { color: #B91C1C !important; font-weight: 800; font-size: 2rem; }
.auth-sub   { color: #374151 !important; font-weight: 500; }

/* Profile card inner text */
.profile-card-name  { color: #ffffff !important; font-size: 1.15rem; font-weight: 800; margin: 12px 0 6px; }
.profile-card-email { color: rgba(255,255,255,0.95) !important; margin: 0; font-size: 0.95rem; font-weight: 500; }

/* General body text in main area */
.main p, .block-container p { color: #111827; font-weight: 500; }

div[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid rgba(0,0,0,0.1) !important; }

/* Selectbox text */
.stSelectbox label { color: #111827 !important; font-weight: 700; }

/* Section subheadings written with st.markdown ##### */
h5 { color: #0F172A !important; font-weight: 800; font-size: 1.05rem; }

/* Upload widget label */
.stFileUploader label { color: #111827 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


#  Session State 
defaults = {
    "token": None, "user_name": None, "user_email": None,
    "user_id": None, "page": "landing", "last_result": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# API Helpers 
def api_post(endpoint, data=None, files=None, json_body=None, auth=False):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    elif st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        if json_body:
            r = requests.post(f"{API_URL}{endpoint}", json=json_body,
                              headers=headers, timeout=90)
        else:
            r = requests.post(f"{API_URL}{endpoint}", data=data,
                              files=files, headers=headers, timeout=90)
        try:
            return r.json(), r.status_code
        except Exception:
            return {"detail": r.text}, r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500

def api_get(endpoint, auth=False):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = requests.get(f"{API_URL}{endpoint}", headers=headers, timeout=30)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500

def api_delete(endpoint):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        r = requests.delete(f"{API_URL}{endpoint}", headers=headers, timeout=30)
        return r.status_code
    except:
        return 500

def conf_class(c):
    if c >= 0.75: return "conf-high"
    if c >= 0.50: return "conf-medium"
    return "conf-low"

def sev_badge(sev):
    s = (sev or "").lower()
    if s == "high":   return '<span class="sev-high">🔴 High Severity</span>'
    if s == "medium": return '<span class="sev-medium">🟡 Medium Severity</span>'
    return '<span class="sev-low">🟢 Low Severity</span>'



# PAGE: LANDING

def page_landing():
    st.markdown("""
    <div class="hero-wrap">
        <h1>🏥 SkinCare AI.</h1>
        <p>AI-Powered Skin Disease Detection &amp; Intelligent Health Advisor</p>
        <p style="font-size:0.95rem;margin-top:6px;color:#ffffff;">
            Upload → Detect → Get personalized AI recommendations instantly
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    feats = [
        ("🔬", "4 AI Models",      "EfficientNetB0, MobileNetV2, ResNet50 & YOLOv8"),
        ("🧠", "LLM Advisor",      "Groq-powered personalized dermatology advice"),
        ("📊", "Full Analytics",   "Confidence scores, probability charts & history"),
        ("📄", "PDF Reports",      "Download detailed professional analysis reports"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], feats):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1.5, 1, 1.5])
    with mid:
        if st.button("🚀 Get Started — Sign In OR Sign Up"):
            st.session_state.page = "auth"
            st.rerun()

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> SkinCare AI. is for informational purposes only and does not replace
        professional medical advice. Always consult a certified dermatologist for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)



# PAGE: AUTH

def page_auth():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px;">
            <h2 class="auth-title">🏥 SkinCare AI.</h2>
            <p class="auth-sub">Your AI-powered skin health companion</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email    = st.text_input("Email Address", placeholder="you@email.com", key="li_email")
            password = st.text_input("Password", placeholder="Your password",
                                     type="password", key="li_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="btn_login"):
                if email and password:
                    with st.spinner("Signing in..."):
                        resp, code = api_post("/auth/login",
                                              data={"username": email, "password": password})
                    if code == 200:
                        st.session_state.token      = resp["access_token"]
                        st.session_state.user_name  = resp["user_name"]
                        st.session_state.user_email = resp["user_email"]
                        st.session_state.user_id    = resp["user_id"]
                        st.session_state.page       = "analyze"
                        st.success(f"Welcome back, {resp['user_name']}! 👋")
                        st.rerun()
                    else:
                        st.error(resp.get("detail", "Login failed. Check your credentials."))
                else:
                    st.warning("Please fill in all fields.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            name      = st.text_input("Full Name",        placeholder="John Doe",         key="su_name")
            email_r   = st.text_input("Email Address",    placeholder="you@email.com",    key="su_email")
            password_r = st.text_input("Password",        placeholder="Minimum 6 characters",
                                       type="password",   key="su_pass")
            password_c = st.text_input("Confirm Password", placeholder="Repeat your password",
                                       type="password",   key="su_conf")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", key="btn_signup"):
                if name and email_r and password_r and password_c:
                    if password_r != password_c:
                        st.error("Passwords do not match.")
                    elif len(password_r) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        with st.spinner("Creating your account..."):
                            resp, code = api_post("/auth/signup",
                                                  json_body={"name": name,
                                                             "email": email_r,
                                                             "password": password_r})
                        if code == 200:
                            st.session_state.token      = resp["access_token"]
                            st.session_state.user_name  = resp["user_name"]
                            st.session_state.user_email = resp["user_email"]
                            st.session_state.user_id    = resp["user_id"]
                            st.session_state.page       = "analyze"
                            st.success(f"Account created! Welcome, {resp['user_name']}! 🎉")
                            st.rerun()
                        else:
                            st.error(resp.get("detail", "Signup failed. Try again."))
                else:
                    st.warning("Please fill in all fields.")



# DISPLAY RESULT

def display_result(resp):
    disease     = resp.get("disease", "Unknown")
    confidence  = resp.get("confidence", 0)
    model_used  = resp.get("model_used", "")
    analysis_id = resp.get("analysis_id", "")
    severity    = resp.get("severity", "")
    urgent      = resp.get("see_doctor_urgently", False)

    conf_pct = f"{confidence * 100:.1f}%"
    cc       = conf_class(confidence)

    urgent_html = '<span class="urgent-badge">🚨 See Doctor Urgently</span>' if urgent else ""
    st.markdown(f"""
    <div class="result-card">
        <p style="color:#718096;font-size:0.78rem;font-weight:700;letter-spacing:1px;margin:0;text-transform:uppercase;">Detected Condition</p>
        <div class="disease-badge">{disease}</div>
        <br>
        <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;margin-top:4px;">
            <div>
                <p style="color:#718096;font-size:0.78rem;font-weight:700;letter-spacing:1px;margin:0 0 4px;text-transform:uppercase;">Confidence</p>
                <span class="{cc}">{conf_pct}</span>
            </div>
            <div>
                <p style="color:#718096;font-size:0.78rem;font-weight:700;letter-spacing:1px;margin:0 0 4px;text-transform:uppercase;">Severity</p>
                <div style="margin-top:4px;">{sev_badge(severity)} {urgent_html}</div>
            </div>
        </div>
        <p style="color:#718096;font-size:0.80rem;margin:16px 0 0;">🤖 Model: <b style="color:#2d3748;">{model_used}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Probability chart
    all_preds = resp.get("all_predictions", {})
    if all_preds:
        sorted_preds = dict(sorted(all_preds.items(), key=lambda x: x[1], reverse=True))
        colors = ["#c0392b" if i == 0 else "#2c7a7b" for i in range(len(sorted_preds))]
        fig = go.Figure(go.Bar(
            x=list(sorted_preds.values()),
            y=list(sorted_preds.keys()),
            orientation='h',
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in sorted_preds.values()],
            textposition='outside',
            textfont=dict(size=11, color="#2d3748")
        ))
        fig.update_layout(
            title=dict(text="Class Probability Distribution", font=dict(size=14, color="#2D3748")),
            xaxis=dict(range=[0, 1.2], showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                       tickformat='.0%', title="", tickfont=dict(color="#4a5568")),
            yaxis=dict(title="", tickfont=dict(color="#2d3748")),
            height=360,
            margin=dict(l=10, r=70, t=45, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=11, color="#2d3748")
        )
        st.plotly_chart(fig, use_container_width=True)

    # LLM Sections
    llm_sections = [
        ("💊", "recommendations", "Recommendations"),
        ("👣", "next_steps",      "Next Steps"),
        ("🌿", "tips",            "Daily Care Tips"),
    ]
    for icon, key, label in llm_sections:
        content = resp.get(key, "")
        if content:
            formatted = content.replace("\n", "<br>")
            st.markdown(f"""
            <div class="llm-block">
                <div class="llm-title">{icon} {label}</div>
                <p>{formatted}</p>
            </div>
            """, unsafe_allow_html=True)

    # PDF + New Analysis buttons
    if analysis_id:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download PDF Report", key="pdf_btn"):
                with st.spinner("Generating PDF..."):
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    r = requests.get(f"{API_URL}/report/{analysis_id}", headers=headers, timeout=30)
                if r.status_code == 200:
                    st.download_button(
                        label="⬇️ Save PDF Report",
                        data=r.content,
                        file_name=f"SkinCare AI._Report_{analysis_id[:8]}.pdf",
                        mime="application/pdf",
                        key="dl_pdf"
                    )
                else:
                    st.error("Failed to generate PDF.")
        with col2:
            if st.button("🔄 New Analysis", key="new_btn"):
                st.session_state.last_result = None
                st.rerun()

    st.markdown("""
    <div class="disclaimer">
        ⚠️ This is AI-generated analysis for informational purposes only.
        Please consult a certified dermatologist for proper medical advice.
    </div>
    """, unsafe_allow_html=True)



# PAGE: ANALYZE

def page_analyze():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#c0392b;font-weight:800;margin:0;">🔬 Skin Analysis</h2>
        <p style="color:#4a5568;margin:4px 0 0;font-size:1rem;">Upload a skin image and select your preferred AI model</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### 📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose a skin image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit photo of the affected skin area"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)

        st.markdown("##### 🤖 Select AI Model")
        model_info = {
            "EfficientNetB0": "⚡ Fast & highly accurate",
            "MobileNetV2":    "📱 Lightweight — Great for quick analysis",
            "ResNet50":       "🏋️ Deep network — High accuracy on complex cases",
            "YOLOv8":         "🎯 YOLO-based — Real-time detection ",
        }
        model_name = st.selectbox("Choose Model", list(model_info.keys()))
        st.info(model_info[model_name])

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Skin Image")

    with col_right:
        if analyze_btn:
            if not uploaded:
                st.error("Please upload an image first!")
            else:
                with st.spinner("🧠 AI is analyzing your image... This may take a moment."):
                    uploaded.seek(0)
                    resp, code = api_post(
                        "/analyze_skin",
                        data={"model_name": model_name},
                        files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                        auth=True
                    )
                if code == 200:
                    st.session_state.last_result = resp
                    display_result(resp)
                else:
                    st.error(f"Analysis failed: {resp.get('detail', 'Unknown error')}")

        elif st.session_state.last_result:
            display_result(st.session_state.last_result)
        else:
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;background:#f7fafc;border-radius:20px;border:2px dashed #cbd5e0;">
                <div style="font-size:5rem;">🔬</div>
                <h3 style="color:#2d3748;font-weight:700;">Upload an image to begin</h3>
                <p style="color:#4a5568;font-size:0.95rem;">Your analysis results will appear here</p>
            </div>
            """, unsafe_allow_html=True)



# PAGE: DASHBOARD

def page_dashboard():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#c0392b;font-weight:800;margin:0;">📊 Dashboard</h2>
        <p style="color:#4a5568;margin:4px 0 0;font-size:1rem;">Your skin health analytics overview</p>
    </div>
    """, unsafe_allow_html=True)

    history, code = api_get("/history", auth=True)
    if code != 200 or not history:
        st.info("No analyses yet. Go to **Analyze** to get started!")
        return

    total       = len(history)
    avg_conf    = sum(h["confidence"] for h in history) / total
    top_disease = max(set(h["disease"] for h in history),
                      key=lambda d: sum(1 for h in history if h["disease"] == d))
    models_used = len(set(h["model_used"] for h in history))

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [total, f"{avg_conf*100:.1f}%", top_disease[:20]+"…" if len(top_disease)>20 else top_disease, models_used],
        ["Total Analyses", "Avg Confidence", "Most Common Condition", "Models Used"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        disease_counts = {}
        for h in history:
            disease_counts[h["disease"]] = disease_counts.get(h["disease"], 0) + 1
        fig_pie = go.Figure(go.Pie(
            labels=list(disease_counts.keys()),
            values=list(disease_counts.values()),
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Pastel),
            textinfo='percent',
            hoverinfo='label+value'
        ))
        fig_pie.update_layout(
            title=dict(text="Disease Distribution", font=dict(size=14, color="#2D3748")),
            font=dict(family="Inter", color="#2d3748"),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(size=10, color="#2d3748")),
            height=340,
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        model_counts = {}
        for h in history:
            model_counts[h["model_used"]] = model_counts.get(h["model_used"], 0) + 1
        fig_bar = go.Figure(go.Bar(
            x=list(model_counts.keys()),
            y=list(model_counts.values()),
            marker_color=["#c0392b", "#2c7a7b", "#d69e2e", "#553c9a"][:len(model_counts)],
            text=list(model_counts.values()),
            textposition='outside',
            textfont=dict(color="#2d3748")
        ))
        fig_bar.update_layout(
            title=dict(text="Model Usage", font=dict(size=14, color="#2D3748")),
            font=dict(family="Inter", color="#2d3748"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=340,
            margin=dict(t=50, b=10, l=10, r=10),
            xaxis=dict(tickfont=dict(color="#2d3748")),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', tickfont=dict(color="#2d3748"))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Confidence over time
    dates = [h.get("timestamp", "")[:10] for h in reversed(history)]
    confs = [round(h["confidence"] * 100, 1) for h in reversed(history)]
    fig_line = go.Figure(go.Scatter(
        x=dates, y=confs,
        mode='lines+markers',
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=7, color="#c0392b"),
        fill='tozeroy',
        fillcolor='rgba(192,57,43,0.08)'
    ))
    fig_line.update_layout(
        title=dict(text="Confidence Score Over Time", font=dict(size=14, color="#2D3748")),
        font=dict(family="Inter", color="#2d3748"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickfont=dict(color="#4a5568")),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                   title="Confidence (%)", range=[0, 110], tickfont=dict(color="#4a5568")),
        height=300,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_line, use_container_width=True)



# PAGE: HISTORY

def page_history():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#c0392b;font-weight:800;margin:0;">📋 Analysis History</h2>
        <p style="color:#4a5568;margin:4px 0 0;font-size:1rem;">All your past skin analyses</p>
    </div>
    """, unsafe_allow_html=True)

    history, code = api_get("/history", auth=True)
    if code != 200 or not history:
        st.info("No analyses yet. Go to **Analyze** to get started!")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search by condition", placeholder="e.g. Eczema, Melanoma")
    with col2:
        model_filter = st.selectbox("Filter by Model",
                                    ["All"] + list(set(h["model_used"] for h in history)))

    filtered = [h for h in history
                if (not search or search.lower() in h["disease"].lower())
                and (model_filter == "All" or h["model_used"] == model_filter)]

    st.markdown(f"<p style='color:#2d3748;font-size:0.90rem;font-weight:600;'><b>{len(filtered)}</b> records found</p>",
                unsafe_allow_html=True)

    for h in filtered:
        conf     = h["confidence"]
        conf_pct = f"{conf*100:.1f}%"
        date_str = h.get("timestamp", "")[:19].replace("T", " ")
        cc       = conf_class(conf)

        col_a, col_b, col_c, col_d, col_e = st.columns([3, 1.5, 1.5, 0.8, 0.8])
        with col_a:
            st.markdown(f"""
            <div class="hist-row">
                <b>{h['disease']}</b><br>
                <small>🕐 {date_str}</small>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<br><span class='{cc}'>{conf_pct}</span>", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"<br><small style='color:#4a5568;font-size:0.88rem;'>🤖 {h['model_used']}</small>",
                        unsafe_allow_html=True)
        with col_d:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📄", key=f"pdf_{h['_id']}", help="Download PDF"):
                with st.spinner(""):
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    r = requests.get(f"{API_URL}/report/{h['_id']}", headers=headers, timeout=30)
                if r.status_code == 200:
                    st.download_button(
                        label="⬇️ Save",
                        data=r.content,
                        file_name=f"SkinCare AI._Report_{h['_id'][:8]}.pdf",
                        mime="application/pdf",
                        key=f"dl_{h['_id']}"
                    )
        with col_e:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_{h['_id']}", help="Delete"):
                status = api_delete(f"/history/{h['_id']}")
                if status == 200:
                    st.success("Deleted!")
                    st.rerun()



# PAGE: PROFILE

def page_profile():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#c0392b;font-weight:800;margin:0;">👤 Profile</h2>
        <p style="color:#4a5568;margin:4px 0 0;font-size:1rem;">Your account information &amp; statistics</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#c0392b,#e67e22);
                    border-radius:20px;padding:44px 20px;text-align:center;
                    box-shadow:0 10px 30px rgba(192,57,43,0.25);">
            <div style="font-size:4.5rem;">👤</div>
            <h3 class="profile-card-name">{st.session_state.user_name}</h3>
            <p class="profile-card-email">{st.session_state.user_email}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        profile, code = api_get("/auth/profile", auth=True)
        if code == 200:
            st.markdown("##### Account Details")
            for k, v in [
                ("👤 Name",         profile.get("name", "")),
                ("📧 Email",        profile.get("email", "")),
                ("📅 Member Since", profile.get("created_at", "")[:10]),
            ]:
                ck, cv = st.columns([1, 2])
                with ck: st.markdown(f"**{k}**")
                with cv: st.markdown(str(v))
                st.divider()

        history, _ = api_get("/history", auth=True)
        if isinstance(history, list) and history:
            st.markdown("##### 📊 Your Statistics")
            s1, s2, s3 = st.columns(3)
            with s1: st.metric("Total Analyses", len(history))
            with s2:
                avg = sum(h["confidence"] for h in history) / len(history)
                st.metric("Avg Confidence", f"{avg*100:.1f}%")
            with s3:
                models = len(set(h["model_used"] for h in history))
                st.metric("Models Used", models)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### ⚠️ Session")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("🚪 Sign Out"):
            for key in ["token", "user_name", "user_email", "user_id", "last_result"]:
                st.session_state[key] = None
            st.session_state.page = "landing"
            st.rerun()



# SIDEBAR

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:24px 0 16px;">
            <h2 style="color:#ff7675;margin:0;font-weight:800;font-size:1.6rem;">🏥 SkinCare AI.</h2>
            <p style="color:#a0aec0;font-size:0.78rem;margin:4px 0 0;">AI Skin Health Platform</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.token:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.07);border-radius:14px;
                        padding:16px;margin:8px 0 16px;text-align:center;">
                <div style="font-size:2.2rem;">👤</div>
                <b style="font-size:0.95rem;color:#f0f0f0;">{st.session_state.user_name}</b><br>
                <small style="color:#a0aec0;font-size:0.78rem;">{st.session_state.user_email}</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            pages = {
                "🔬  Analyze":    "analyze",
                "📊  Dashboard":  "dashboard",
                "📋  History":    "history",
                "👤  Profile":    "profile",
            }
            for label, page_key in pages.items():
                is_active = st.session_state.page == page_key
                css_class = "nav-btn-active" if is_active else "nav-btn"
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.page = page_key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("---")
            if st.button("🚀 Get Started", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="padding:10px 4px;">
            <p style="color:#a0aec0;font-size:0.78rem;font-weight:700;margin:0 0 8px;letter-spacing:0.8px;text-transform:uppercase;">Detectable Conditions</p>
            <p style="color:#cbd5e0;font-size:0.80rem;line-height:2;margin:0;">
                Eczema • Melanoma • Atopic Dermatitis<br>
                Basal Cell Carcinoma • Psoriasis<br>
                Seborrheic Keratoses • Ringworm<br>
                Warts &amp; Molluscum • Melanocytic Nevi
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════
render_sidebar()

if not st.session_state.token:
    if st.session_state.page == "auth":
        page_auth()
    else:
        page_landing()
else:
    page_map = {
        "analyze":   page_analyze,
        "dashboard": page_dashboard,
        "history":   page_history,
        "profile":   page_profile,
    }
    page_map.get(st.session_state.page, page_analyze)()