import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import PlantDiseaseCNN

st.set_page_config(page_title="LeafScan AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #0d1f17;
    font-family: 'DM Sans', sans-serif;
}

@keyframes float1 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    33%       { transform: translate(30px, -30px) scale(1.05); }
    66%       { transform: translate(-20px, 20px) scale(0.95); }
}
@keyframes float2 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    33%       { transform: translate(-40px, 20px) scale(1.08); }
    66%       { transform: translate(25px, -25px) scale(0.92); }
}
@keyframes float3 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50%       { transform: translate(20px, 30px) scale(1.06); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    70%  { transform: scale(1);    box-shadow: 0 0 0 12px rgba(34,197,94,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
@keyframes scanline {
    0%   { top: 0%; opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.bg-canvas {
    position: fixed;
    inset: 0;
    overflow: hidden;
    z-index: 0;
    pointer-events: none;
}
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.5;
}
.orb-1 {
    width: 700px; height: 700px;
    background: radial-gradient(circle, #22c55e, #15803d);
    top: -200px; left: -150px;
    animation: float1 12s ease-in-out infinite;
}
.orb-2 {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #16a34a, #064e3b);
    top: 20%; right: -150px;
    animation: float2 15s ease-in-out infinite;
}
.orb-3 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #4ade80, #166534);
    bottom: -150px; left: 25%;
    animation: float3 18s ease-in-out infinite;
}
.orb-4 {
    width: 350px; height: 350px;
    background: radial-gradient(circle, #86efac, #15803d);
    top: 55%; left: 5%;
    animation: float1 20s ease-in-out infinite reverse;
}

.main-wrap {
    position: relative;
    z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    animation: fadeSlideUp 0.7s ease both;
}

.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3.5rem;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1rem;
    font-weight: 500;
    color: #86efac;
    letter-spacing: 0.5px;
}
.nav-logo-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-ring 2s infinite;
}
.nav-badge {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #86efac;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
}

.hero-section {
    text-align: center;
    margin-bottom: 3rem;
}
.hero-tag {
    display: inline-block;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    color: #86efac;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    margin-bottom: 1.8rem;
}
.hero-h1 {
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 300;
    line-height: 1.1;
    letter-spacing: -1.5px;
    color: #f0fdf4;
    margin-bottom: 1.2rem;
}
.hero-h1 span {
    font-weight: 700;
    color: #4ade80;
}
.hero-p {
    color: #86efac;
    font-size: 1rem;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto 2.5rem;
    line-height: 1.8;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 3rem;
}
.hstat { text-align: center; }
.hstat-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0fdf4;
    letter-spacing: -0.5px;
}
.hstat-lbl {
    font-size: 0.65rem;
    color: #86efac;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3), 0 1px 0 rgba(255,255,255,0.05) inset;
}

.section-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.75rem;
}

.result-status-healthy {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #86efac;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 0.4rem 1rem; border-radius: 999px;
    margin-bottom: 1rem;
}
.result-status-disease {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    color: #fca5a5;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 0.4rem 1rem; border-radius: 999px;
    margin-bottom: 1rem;
}

.disease-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f0fdf4;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.disease-sub {
    color: #86efac;
    font-size: 0.8rem;
    font-weight: 300;
    margin-bottom: 1.5rem;
}

.metrics-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.metric-pill {
    flex: 1;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
}
.metric-pill-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #4ade80;
    letter-spacing: -0.5px;
}
.metric-pill-lbl {
    font-size: 0.6rem;
    color: #86efac;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
}

.bar-row {
    margin-bottom: 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    backdrop-filter: blur(10px);
}
.bar-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 1rem;
}
.bar-name { color: #d1fae5; font-weight: 500; }
.bar-pct  { color: #4ade80; font-weight: 700; }

.scanning-wrap {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    justify-content: center;
}
.scan-line {
    position: absolute;
    left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
    animation: scanline 1.5s ease-in-out;
    z-index: 10;
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #4ade80;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-txt  { font-size: 0.85rem; line-height: 1.7; font-weight: 300; color: #86efac; }

section[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px dashed rgba(255,255,255,0.15) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 24px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #86efac !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: rgba(34,197,94,0.1) !important;
    border: 1px solid rgba(34,197,94,0.25) !important;
    color: #86efac !important;
    border-radius: 10px !important;
}
small {
    color: #4ade80 !important;
}
}
.stProgress > div > div {
    background: linear-gradient(90deg, #15803d, #22c55e, #86efac) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: rgba(34,197,94,0.1) !important;
    border-radius: 999px !important;
    height: 7px !important;
}
footer, #MainMenu, header { visibility: hidden; }
</style>

<div class="bg-canvas">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <div class="orb orb-4"></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(r'C:\plant_disease_project\best_plant_model.pth', map_location=device)
    classes    = checkpoint['classes']
    model      = PlantDiseaseCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes, device

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, classes, device = load_model()

st.markdown("""
<div class="main-wrap">
    <div class="nav">
        <div class="nav-logo">
            <div class="nav-logo-dot"></div>
            LeafScan AI
        </div>
        <div class="nav-badge">⚡ Live · RTX 4060</div>
    </div>
    <div class="hero-section">
        <div class="hero-tag">🌿 CNN · ResNet18 · PlantVillage</div>
        <div class="hero-h1">Detect Plant<br/><span>Disease Instantly</span></div>
        <div class="hero-p">Upload any leaf photo. Our AI model identifies diseases across 15 plant classes with near-perfect accuracy.</div>
        <div class="hero-stats">
            <div class="hstat"><div class="hstat-val">99.4%</div><div class="hstat-lbl">Accuracy</div></div>
            <div class="hstat"><div class="hstat-val">15</div><div class="hstat-lbl">Classes</div></div>
            <div class="hstat"><div class="hstat-val">20k+</div><div class="hstat-lbl">Images trained</div></div>
            <div class="hstat"><div class="hstat-val">&lt;1s</div><div class="hstat-lbl">Inference</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-eyebrow">Upload Leaf Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.markdown('<div class="scanning-wrap"><div class="scan-line"></div>', unsafe_allow_html=True)
        st.image(image, width=300)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🍃</div>
            <div class="empty-txt">Drop a JPG or PNG here<br/>to scan for diseases</div>
        </div>""", unsafe_allow_html=True)
        
with right:
    if uploaded:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]

        confidence, predicted = probs.max(0)
        disease    = classes[predicted.item()]
        confidence = confidence.item() * 100
        is_healthy = "healthy" in disease.lower()

        status_html  = f'<div class="result-status-{"healthy" if is_healthy else "disease"}">{"✅ Healthy" if is_healthy else "⚠️ Disease Detected"}</div>'
        disease_clean = disease.replace('_', ' ')

        st.markdown(f"""
        <div class="card" style="animation: fadeSlideUp 0.5s ease both;">
            <p class="section-eyebrow">Scan Result</p>
            {status_html}
            <div class="disease-title">{disease_clean}</div>
            <div class="disease-sub">Identified via ResNet18 · PlantVillage dataset</div>
            <div class="metrics-row">
                <div class="metric-pill">
                    <div class="metric-pill-val">{confidence:.1f}%</div>
                    <div class="metric-pill-lbl">Confidence</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-val">{len(classes)}</div>
                    <div class="metric-pill-lbl">Classes</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-val">99%</div>
                    <div class="metric-pill-lbl">Model Acc</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-eyebrow" style="margin-top:1.5rem">Top Predictions</p>', unsafe_allow_html=True)
        top5 = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[:5]
        for idx, prob in top5:
            label = classes[idx].replace('_', ' ')
            pct   = prob * 100
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-meta">
                    <span class="bar-name">{label}</span>
                    <span class="bar-pct">{pct:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)
            st.progress(prob)
    else:
        st.markdown("""
        <div class="card">
            <div class="empty-state">
                <div class="empty-icon">🔬</div>
                <div class="empty-txt">Results will appear here<br/>after uploading a leaf image</div>
            </div>
        </div>""", unsafe_allow_html=True)