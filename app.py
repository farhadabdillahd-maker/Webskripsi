import streamlit as st
import pandas as pd
import re
import joblib

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Klasifikasi Tingkat Kejahatan",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# DARK MODE
# =========================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# =========================================
# DARK MODE TOGGLE
# =========================================
st.sidebar.markdown("### 🎨 Theme")

dark_toggle = st.sidebar.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode
)

st.session_state.dark_mode = dark_toggle

# =========================================
# COLOR MODE
# =========================================
if st.session_state.dark_mode:

    bg_main = "#0f172a"
    bg_sidebar = "#111827"
    bg_card = "#172033"
    text_color = "#f8fafc"
    sub_text = "#94a3b8"
    border = "#334155"
    hover = "#1e40af"
    upload_bg = "#1e3a8a"

else:

    bg_main = "#f4f7fb"
    bg_sidebar = "#ffffff"
    bg_card = "#ffffff"
    text_color = "#0f172a"
    sub_text = "#64748b"
    border = "#e5e7eb"
    hover = "#eff6ff"
    upload_bg = "#dbeafe"

# =========================================
# CUSTOM CSS
# =========================================
st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

:root {{

    --text-main: {text_color};
    --text-sub: {sub_text};
    --bg-card: {bg_card};
    --border: {border};

}}

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background: {bg_main};
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: transparent !important;
}}

section[data-testid="stSidebar"] {{
    background: {bg_sidebar};
    border-right: 1px solid {border};
    width: 320px !important;
}}

[data-testid="collapsedControl"] {{
    display: block !important;
    background: white;
    border-radius: 12px;
    padding: 8px;
}}

.logo-container {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 40px;
}}

.logo-icon {{
    width: 70px;
    height: 70px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 34px;
    color: white;
}}

.logo-title {{
    font-size: 30px;
    font-weight: 800;
    color: {text_color};
    line-height: 1;
}}

.logo-sub {{
    color: {sub_text};
    font-size: 14px;
}}

.menu-title {{
    font-size: 13px;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 10px;
}}

.stRadio > div {{
    gap: 12px;
}}

.stRadio label {{
    padding: 14px 18px;
    border-radius: 18px;
    transition: 0.3s;
    font-weight: 600;
}}

.stRadio label:hover {{
    background: {hover};
}}

[data-testid="stFileUploader"] {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 24px;
    padding: 20px;
}}

.main-card {{
    background: {bg_card};
    border-radius: 40px;
    padding: 40px;
}}

.hero {{
    display: flex;
    align-items: center;
    gap: 35px;
}}

.hero-icon {{
    width: 140px;
    height: 140px;
    border-radius: 40px;
    background: {"#0f172a" if st.session_state.dark_mode else "#ffffff"};
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 75px;
}}

.hero-title {{
    font-size: 65px;
    font-weight: 800;
    color: {text_color};
    line-height: 1.1;
}}

.hero-subtitle {{
    font-size: 28px;
    font-weight: 700;
    color: {text_color};
}}

.hero-desc {{
    color: {sub_text};
    font-size: 18px;
    line-height: 1.8;
}}

.blue-line {{
    width: 90px;
    height: 7px;
    border-radius: 20px;
    background: #2563eb;
    margin: 20px 0;
}}

.info-box {{
    margin-top: 35px;
    background: {upload_bg};
    border-radius: 35px;
    padding: 35px;
}}

.info-title {{
    font-size: 26px;
    font-weight: 700;
    color: {text_color};
}}

.info-desc {{
    color: {sub_text};
}}

.stButton > button {{
    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color: white !important;
    border: none;
    border-radius: 18px;
    height: 55px;
    width: 100%;
    font-size: 16px;
    font-weight: 700;
}}

.stTextArea textarea {{
    background: {bg_card} !important;
    color: {text_color} !important;
    border: 1px solid {border} !important;
}}

[data-testid="metric-container"] {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 24px;
    padding: 20px;
}}

[data-testid="metric-container"] * {{
    color: {text_color} !important;
}}

[data-testid="stDataFrame"] {{
    border-radius: 20px;
    overflow: hidden;
}}

thead tr th {{
    background: #2563eb !important;
    color: white !important;
}}

tbody tr td {{
    background: {bg_card} !important;
    color: {text_color} !important;
}}

h1,h2,h3,h4,h5,h6,p,span,label,div {{
    color: {text_color} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {text_color} !important;
}}

/* =========================================
UPLOAD FIX DARK MODE
========================================= */

[data-testid="stFileUploader"] * {{
    color: var(--text-main) !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    color: var(--text-main) !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] * {{
    color: var(--text-main) !important;
}}

small {{
    color: var(--text-sub) !important;
}}

.stRadio label p {{
    color: var(--text-main) !important;
}}

[data-testid="stDataFrameToolbar"] * {{
    color: var(--text-main) !important;
}}

.glideDataEditor * {{
    color: var(--text-main) !important;
}}

[data-testid="stWidgetLabel"] * {{
    color: var(--text-main) !important;
}}

[data-testid="stFileUploader"] small {{
    color: var(--text-sub) !important;
}}

.row_heading {{
    color: var(--text-main) !important;
}}

[data-testid="stMarkdownContainer"] * {{
    color: var(--text-main) !important;
}}

[data-baseweb="tooltip"] {{
    color: white !important;
}}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.markdown("""
<div class="logo-container">

<div class="logo-icon">
🛡️
</div>

<div>
<div class="logo-title">
KLASIFIKASI
</div>

<div class="logo-sub">
TINGKAT KEJAHATAN
</div>
</div>

</div>

<div class="menu-title">
MENU
</div>
""", unsafe_allow_html=True)

# =========================================
# MENU
# =========================================
menu = st.sidebar.radio(
    "",
    [
        "📂 Upload Dataset",
        "🧹 Preprocessing",
        "📊 Klasifikasi Naïve Bayes",
        "📈 Prediksi"
    ]
)

# =========================================
# UPLOAD
# =========================================
st.sidebar.markdown("""
<div class="menu-title" style="margin-top:30px;">
DATASET
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)
