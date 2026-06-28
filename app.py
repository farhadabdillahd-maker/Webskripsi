import streamlit as st
import pandas as pd
import re
import joblib
import nltk

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

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS PREMIUM DASHBOARD
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

/* HIDE STREAMLIT */

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

/* Header tetap tampil agar tombol menu  muncul */
header[data-testid="stHeader"]{
    background: transparent;
}

/* BACKGROUND */

.stApp{
    background:
    linear-gradient(
    135deg,
    #f8fafc,
    #eef2ff,
    #ffffff
    );
}

/* SIDEBAR */

section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #0f172a 0%,
        #1e293b 40%,
        #2563eb 100%
    );
    color:white;
    border:none;

    /* Rounded Sidebar */
    border-top-right-radius:30px;
    border-bottom-right-radius:30px;
    overflow:hidden;
}

/* HERO */

.hero{
    background:white;
    padding:35px;
    border-radius:28px;
    box-shadow:
    0 10px 35px rgba(0,0,0,.05);
    margin-bottom:25px;
}

.hero-title{
    font-size:42px;
    font-weight:800;
    color:#0f172a;
}

.hero-sub{
    color:#64748b;
    font-size:16px;
    line-height:1.8;
}

/* KPI */

[data-testid="metric-container"]{
    background:white;
    border:none;
    border-radius:20px;
    padding:20px;
    box-shadow:
    0 8px 25px rgba(0,0,0,.05);
}

/* BUTTON */

.stButton button{
    width:100%;
    height:55px;
    border:none;
    border-radius:18px;
    font-weight:700;
    color:white;
    background:linear-gradient(
        135deg,
        #06b6d4,
        #2563eb,
        #7c3aed
    );
    box-shadow:0 10px 25px rgba(37,99,235,.35);
}

/* DATAFRAME */

[data-testid="stDataFrame"]{
    border-radius:18px;
    overflow:hidden;
}

/* UPLOADER */

[data-testid="stFileUploader"]{
    border-radius:18px;
    background:#f8fafc;
    padding:15px;
}

/* SUCCESS */

.stSuccess{
    border-radius:16px;
}

/* WARNING */

.stWarning{
    border-radius:16px;
}

/* INFO */

.stInfo{
    border-radius:16px;
}

/* CUSTOM CARD */

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:
    0 10px 25px rgba(0,0,0,.05);
}

/* SIDEBAR LOGO */

.logo-box{
    text-align:center;
    margin-top:-80px !important;
    margin-bottom:10px !important;
}

/* Hilangkan jarak atas bawaan sidebar */
section[data-testid="stSidebar"] > div{
    padding-top:0rem !important;
}

/* Logo Sidebar */
[data-testid="stSidebar"] img{
    display:block !important;
    margin-left:auto !important;
    margin-right:auto !important;
    margin-top:-95px !important;
    filter:drop-shadow(0 0 15px rgba(59,130,246,.4));
}

/* Naikkan judul */
.logo-title{
    margin-top:-10px !important;
    color:white !important;
    font-size:24px !important;
    font-weight:800 !important;
}

/* Naikkan subtitle */
.logo-sub{
    margin-top:-5px !important;
    color:#cbd5e1 !important;
}

.logo-icon{
    font-size:45px;
}

.logo-title{
    font-size:18px;
    font-weight:800;
    color:#111827;
}

.logo-sub{
    font-size:12px;
    color:#6b7280;
}


/* Sidebar Modern */
[data-testid="stFileUploader"]{
    background:rgba(255,255,255,.12);
    backdrop-filter:blur(15px);
    border:1px solid rgba(255,255,255,.15);
    border-radius:22px;
    padding:20px;
}

div[role="radiogroup"] label{
    background:rgba(255,255,255,.12);
    padding:12px;
    border-radius:12px;
    margin-bottom:8px;
}

div[role="radiogroup"] label:hover{
    background:rgba(255,255,255,.20);
}

.logo-title{
    color:white !important;
    font-size:22px;
}

.logo-sub{
    color:#cbd5e1 !important;
}


/* TEKS FILE UPLOAD AGAR TERLIHAT */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p{
    color:#0f172a !important;
    font-weight:700 !important;
}

/* CARD FILE YANG SUDAH TERUPLOAD */
[data-testid="stFileUploader"] section{
    background:linear-gradient(
        135deg,
        #ffffff,
        #eff6ff
    ) !important;

    border:1px solid #bfdbfe !important;
    border-radius:14px !important;
    box-shadow:0 5px 15px rgba(37,99,235,.15);
    padding:10px;
}

/* TOMBOL HAPUS FILE */
[data-testid="stFileUploader"] svg{
    color:#2563eb !important;
}

/* TEKS MENU SIDEBAR */
div[role="radiogroup"] label p{
    color:#ffffff !important;
    opacity:1 !important;
    font-weight:700 !important;
}


/* ========================= */
/* PREMIUM BUTTON ANIMATION */
/* ========================= */

.stButton button{
    transition:all .25s ease !important;
}

.stButton button:hover{
    transform:translateY(-3px) scale(1.05) !important;
    box-shadow:
        0 0 25px rgba(37,99,235,.60),
        0 0 50px rgba(124,58,237,.40) !important;
}

.stButton button:active{
    transform:scale(0.97) !important;
}

/* MENU CARD */
div[role="radiogroup"] label{
    width:100% !important;
    min-width:100% !important;
    height:55px !important;
    transition:all .25s ease !important;
}

div[role="radiogroup"] label:hover{
    transform:translateX(6px) scale(1.05) !important;
    box-shadow:
        0 0 25px rgba(37,99,235,.55),
        0 0 45px rgba(124,58,237,.35) !important;
}

div[role="radiogroup"] label p{
    color:white !important;
    font-weight:700 !important;
}


[data-testid="stSidebar"] img{
    display:block;
    margin-left:auto;
    margin-right:auto;
    filter:drop-shadow(0 0 15px rgba(59,130,246,.4));
}


[data-testid="stSidebar"] img{
    display:block !important;
    margin:auto !important;
}



/* ===== Corner Accent Border ===== */
section[data-testid="stSidebar"]{
position:relative!important;
overflow:hidden;
border-radius:32px!important;
background:#0b1224!important;
}
section[data-testid="stSidebar"]::before{
content:"";
position:absolute;
inset:12px;
border-radius:30px;
pointer-events:none;
background:
linear-gradient(#2bd9ff,#2bd9ff) left top/60px 3px no-repeat,
linear-gradient(#2bd9ff,#2bd9ff) left top/3px 60px no-repeat,
linear-gradient(#ff2b5f,#ff2b5f) right top/60px 3px no-repeat,
linear-gradient(#ff2b5f,#ff2b5f) right top/3px 60px no-repeat,
linear-gradient(#ff2b5f,#ff2b5f) left bottom/60px 3px no-repeat,
linear-gradient(#ff2b5f,#ff2b5f) left bottom/3px 60px no-repeat,
linear-gradient(#2bd9ff,#2bd9ff) right bottom/60px 3px no-repeat,
linear-gradient(#2bd9ff,#2bd9ff) right bottom/3px 60px no-repeat;
filter:drop-shadow(0 0 8px #2bd9ff) drop-shadow(0 0 8px #ff2b5f);
}
section[data-testid="stSidebar"]::after{
content:"";
position:absolute;
inset:12px;
border-radius:30px;
pointer-events:none;
background:
linear-gradient(90deg,transparent,#2bd9ff,transparent) top/180% 3px no-repeat,
linear-gradient(180deg,transparent,#ff2b5f,transparent) right/3px 180% no-repeat,
linear-gradient(90deg,transparent,#2bd9ff,transparent) bottom/180% 3px no-repeat,
linear-gradient(180deg,transparent,#ff2b5f,transparent) left/3px 180% no-repeat;
animation:borderRun 3.5s linear infinite;
filter:drop-shadow(0 0 10px #2bd9ff) drop-shadow(0 0 10px #ff2b5f);
}
@keyframes borderRun{
0%{background-position:-140% 0,100% -140%,140% 100%,0 140%;}
25%{background-position:140% 0,100% 140%,140% 100%,0 140%;}
50%{background-position:140% 0,100% 140%,-140% 100%,0 140%;}
75%{background-position:140% 0,100% 140%,-140% 100%,0 -140%;}
100%{background-position:-140% 0,100% -140%,140% 100%,0 140%;}
}

/* ============================= */
/* POLICE HOVER EFFECT MENU      */
/* ============================= */

div[role="radiogroup"] label{
    position:relative;
    overflow:hidden;
    transition:all .35s ease!important;
}

div[role="radiogroup"] label::before{
    content:"";
    position:absolute;
    inset:0;
    background:
      linear-gradient(90deg,transparent 0 35%,rgba(0,220,255,.95) 48%,transparent 60%),
      linear-gradient(90deg,transparent 0 65%,rgba(255,30,90,.95) 78%,transparent 90%);
    background-size:250% 100%,250% 100%;
    opacity:0;
    transition:.25s;
}

div[role="radiogroup"] label:hover::before{
    opacity:1;
    animation:policeSweep .8s linear infinite;
}

div[role="radiogroup"] label:hover{
    transform:translateX(8px) scale(1.04)!important;
    border:1px solid rgba(255,255,255,.35);
    box-shadow:
      -12px 0 18px rgba(0,220,255,.55),
       12px 0 18px rgba(255,30,90,.55),
       0 0 24px rgba(255,255,255,.08)!important;
}

@keyframes policeSweep{
    from{background-position:-150% 0,150% 0;}
    to{background-position:150% 0,-150% 0;}
}


/* ===== POPPINS MENU & BUTTON FONT ===== */
.stButton button,
div[role="radiogroup"] label,
div[role="radiogroup"] label p{
    font-family:'Poppins',sans-serif !important;
    font-weight:700 !important;
    letter-spacing:.2px;
}


/* Center MENU button */
section[data-testid="stSidebar"] .stButton>button{
margin:auto!important;
max-width:150px!important;
display:block!important;
}

/* Menu animation */
div[role="radiogroup"]{
animation:menuIn .45s cubic-bezier(.2,.8,.2,1);
transform-origin:top center;
}
@keyframes menuIn{
0%{opacity:0;transform:translateY(-20px) scale(.92);}
60%{opacity:1;transform:translateY(6px) scale(1.02);}
100%{opacity:1;transform:translateY(0) scale(1);}
}
div[role="radiogroup"] label{
animation:itemIn .45s ease both;
}
div[role="radiogroup"] label:nth-child(1){animation-delay:.05s;}
div[role="radiogroup"] label:nth-child(2){animation-delay:.10s;}
div[role="radiogroup"] label:nth-child(3){animation-delay:.15s;}
div[role="radiogroup"] label:nth-child(4){animation-delay:.20s;}
div[role="radiogroup"] label:nth-child(5){animation-delay:.25s;}
@keyframes itemIn{
from{opacity:0;transform:translateX(-25px);}
to{opacity:1;transform:translateX(0);}
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO DASHBOARD
# =====================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🚔 Crime Analytics Dashboard
</div>

<br>

<div class="hero-sub">
PENERAPAN MACHINE LEARNING MENGGUNAKAN ALGORITMA NAÏVE BAYES UNTUK KLASIFIKASI TINGKAT KEJAHATAN DI POLRES PASAMAN.
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div class="logo-box"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns([1,3,1])

with col2:
    st.image(
        "assets/logo.png",
        width=165
    )

st.sidebar.markdown("""
<div class="logo-title" style="text-align:center;">
CRIME ANALYTICS
</div>

<div class="logo-sub" style="text-align:center;">
Naïve Bayes Dashboard
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div style='height:60px'></div>",
    unsafe_allow_html=True
)

if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

_c1,_c2,_c3 = st.sidebar.columns([1,2,1])
with _c2:
    btn_clicked = st.button("☰ MENU", use_container_width=True)
if btn_clicked:
    st.session_state.show_menu = not st.session_state.show_menu

menu = None

if st.session_state.show_menu:
    menu = st.sidebar.radio(
        "",
        [
            "📂 Upload Dataset",
            "🧹 Preprocessing",
            "🤖 Klasifikasi",
            "🔍 Prediksi",
            "ℹ️ About"
        ]
    )

if menu is None:
    st.info("Klik tombol ☰ MENU untuk membuka navigasi.")
    st.stop()

uploaded_file = None
if menu in ["📂 Upload Dataset","🧹 Preprocessing","🤖 Klasifikasi"]:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )


# =====================================================
# MENU PREDIKSI TANPA UPLOAD DATASET
# =====================================================
if menu == "🔍 Prediksi" and uploaded_file is None:

    st.markdown("""
    <div class="card">
    <h2>🔍 Prediksi Tingkat Kejahatan</h2>
    <p>Prediksi dapat digunakan tanpa upload dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model = joblib.load("model_naive_bayes.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")

        try:
            stop_words = set(stopwords.words("indonesian"))
        except:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("indonesian"))

        stemmer = StemmerFactory().create_stemmer()

        malam_keywords = [
            "malam","subuh","dini hari","tengah malam",
            "larut malam","jam 1","jam 2","jam 3","jam 4","jam 5"
        ]

        input_text = st.text_area("Masukkan Judul Berita", height=150)

        if st.button("🚀 Prediksi"):
            if input_text.strip():
                detected = False
                txt = input_text.lower()

                for k in malam_keywords:
                    if k in txt:
                        prediction = "Kasus Malam"
                        detected = True
                        break

                if not detected:
                    txt = re.sub(r"[^\w\s]", "", txt)
                    tokens = [w for w in txt.split() if w not in stop_words]
                    tokens = [stemmer.stem(w) for w in tokens]
                    vector = tfidf.transform([" ".join(tokens)])
                    prediction = model.predict(vector)[0]

                st.success(f"Hasil Prediksi : {prediction}")
            else:
                st.warning("Masukkan judul berita terlebih dahulu.")
    except:
        st.error("Model belum tersedia. Jalankan menu Klasifikasi terlebih dahulu untuk membuat model.")



# =====================================================
# MENU ABOUT
# =====================================================
if menu == "ℹ️ About":
    st.markdown("""
    <div class="card">
        <h2>👨‍💻 About Developer</h2>
        <p>Informasi pengembang aplikasi.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### 👤 Profil Pengembang

**NOBP** : **22101152630058**

**Nama** : **FARHAD ABDILLAH DARNAZ**

**Jurusan** : **TEKNIK INFORMATIKA**

Aplikasi ini merupakan implementasi algoritma **Naïve Bayes** untuk klasifikasi tingkat kejahatan di **Polres Pasaman** menggunakan **Python, Streamlit, TF‑IDF, dan Sastrawi**.
""")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================

if menu in ["📂 Upload Dataset","🧹 Preprocessing","🤖 Klasifikasi"]:

    if uploaded_file is None:
        st.info("Silakan upload dataset CSV untuk menggunakan menu ini.")
        st.stop()


    df = pd.read_csv(uploaded_file)

    # =====================================================
    # VALIDASI KOLOM
    # =====================================================

    if "Judul Media Nasional" not in df.columns:

        st.error(
            "Kolom 'Judul Media Nasional' tidak ditemukan!"
        )

        st.stop()

    # =====================================================
    # AMBIL KOLOM
    # =====================================================

    df = df[
        [
            "Judul Media Nasional"
        ]
    ]

    # =====================================================
    # STOPWORD
    # =====================================================

    try:

        stop_words = set(
            stopwords.words("indonesian")
        )

    except:

        nltk.download("stopwords")

        stop_words = set(
            stopwords.words("indonesian")
        )

    # =====================================================
    # STEMMER
    # =====================================================

    factory = StemmerFactory()

    stemmer = factory.create_stemmer()

    # =====================================================
    # PREPROCESSING FUNCTION
    # =====================================================

    def case_folding(text):

        return str(text).lower()

    def tokenizing(text):

        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

        return text.split()

    def stopword_removal(tokens):

        return [
            word
            for word in tokens
            if word not in stop_words
        ]

    def stemming(tokens):

        return [
            stemmer.stem(word)
            for word in tokens
        ]

    # =====================================================
    # AUTO LABEL
    # =====================================================

    malam_keywords = [

        "malam",
        "subuh",
        "dini hari",
        "tengah malam",
        "larut malam",
        "jam 1",
        "jam 2",
        "jam 3",
        "jam 4",
        "jam 5"

    ]

    def auto_label(text):

        text = str(text).lower()

        for keyword in malam_keywords:

            if keyword in text:

                return "Kasus Malam"

        return "Kasus Umum"

    # =====================================================
    # PREPROCESSING
    # =====================================================

    with st.spinner(
        "Melakukan preprocessing dataset..."
    ):

        df["Case Folding"] = df[
            "Judul Media Nasional"
        ].apply(case_folding)

        df["Tokenizing"] = df[
            "Case Folding"
        ].apply(tokenizing)

        df["Stopword Removal"] = df[
            "Tokenizing"
        ].apply(stopword_removal)

        df["Stemming"] = df[
            "Stopword Removal"
        ].apply(stemming)

        df["Final Text"] = df[
            "Stemming"
        ].apply(
            lambda x: " ".join(x)
        )

        df["Label"] = df[
            "Judul Media Nasional"
        ].apply(auto_label)

    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    total_data = len(df)

    total_malam = len(
        df[
            df["Label"] == "Kasus Malam"
        ]
    )

    total_umum = len(
        df[
            df["Label"] == "Kasus Umum"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Total Data",
            f"{total_data:,}"
        )

    with col2:

        st.metric(
            "🌙 Kasus Malam",
            f"{total_malam:,}"
        )

    with col3:

        st.metric(
            "☀️ Kasus Umum",
            f"{total_umum:,}"
        )

    with col4:

        st.metric(
            "📊 Status",
            "Loaded"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # DISTRIBUSI LABEL
    # =====================================================

    chart_col1, chart_col2 = st.columns([2,1])

    with chart_col1:

        st.markdown("""
        <div class="card">
        <h3>Distribusi Dataset</h3>
        </div>
        """, unsafe_allow_html=True)

        distribusi = df["Label"].value_counts()

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        distribusi.plot(
            kind="bar",
            ax=ax
        )

        plt.xlabel("Label")

        plt.ylabel("Jumlah")

        plt.title(
            "Distribusi Kasus"
        )

        st.pyplot(fig)

    with chart_col2:

        st.markdown("""
        <div class="card">

        <h4>Informasi Dataset</h4>

        <br>

        <b>Kolom:</b><br>
        Judul Media Nasional

        <br><br>

        <b>Metode:</b><br>
        Naïve Bayes

        <br><br>

        <b>Ekstraksi Fitur:</b><br>
        TF-IDF

        <br><br>

        <b>Stemming:</b><br>
        Sastrawi

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # =====================================================
# MENU UPLOAD DATASET
# =====================================================

    if menu == "📂 Upload Dataset":

        st.markdown("""
        <div class="card">
        <h2>📂 Dataset Awal</h2>
        <p>Dataset berita kriminal yang berhasil diupload.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            df[
                ["Judul Media Nasional"]
            ],
            use_container_width=True,
            height=500
        )

        st.success(
            f"Dataset berhasil dimuat ({len(df)} data)"
        )

    # =====================================================
    # MENU PREPROCESSING
    # =====================================================

    elif menu == "🧹 Preprocessing":

        st.markdown("""
        <div class="card">
        <h2>🧹 Tahapan Preprocessing</h2>
        <p>
        Menampilkan seluruh proses preprocessing
        yang dilakukan sebelum klasifikasi.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Case Folding",
                "Tokenizing",
                "Stopword",
                "Stemming",
                "Labeling"
            ]
        )

        # =====================================
        # CASE FOLDING
        # =====================================

        with tab1:

            st.subheader(
                "Case Folding"
            )

            st.info(
                "Mengubah seluruh huruf menjadi huruf kecil."
            )

            st.dataframe(
                df[
                    [
                        "Judul Media Nasional",
                        "Case Folding"
                    ]
                ],
                use_container_width=True,
                height=500
            )

        # =====================================
        # TOKENIZING
        # =====================================

        with tab2:

            st.subheader(
                "Tokenizing"
            )

            st.info(
                "Memecah kalimat menjadi token kata."
            )

            st.dataframe(
                df[
                    [
                        "Case Folding",
                        "Tokenizing"
                    ]
                ],
                use_container_width=True,
                height=500
            )

        # =====================================
        # STOPWORD
        # =====================================

        with tab3:

            st.subheader(
                "Stopword Removal"
            )

            st.info(
                "Menghapus kata-kata yang tidak memiliki makna penting."
            )

            st.dataframe(
                df[
                    [
                        "Tokenizing",
                        "Stopword Removal"
                    ]
                ],
                use_container_width=True,
                height=500
            )

        # =====================================
        # STEMMING
        # =====================================

        with tab4:

            st.subheader(
                "Stemming"
            )

            st.info(
                "Mengubah kata menjadi bentuk dasar."
            )

            st.dataframe(
                df[
                    [
                        "Stopword Removal",
                        "Stemming"
                    ]
                ],
                use_container_width=True,
                height=500
            )

        # =====================================
        # LABELING
        # =====================================

        with tab5:

            st.subheader(
                "Pelabelan Dataset"
            )

            st.info(
                "Pelabelan otomatis Kasus Malam dan Kasus Umum."
            )

            st.dataframe(
                df[
                    [
                        "Judul Media Nasional",
                        "Label"
                    ]
                ],
                use_container_width=True,
                height=500
            )

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "Kasus Malam",
                    total_malam
                )

            with col_b:

                st.metric(
                    "Kasus Umum",
                    total_umum
                )
    # =====================================================
    # MENU KLASIFIKASI
    # =====================================================

    elif menu == "🤖 Klasifikasi":

        st.markdown("""
        <div class="card">
        <h2>🤖 Klasifikasi Naïve Bayes</h2>
        <p>
        Proses training model menggunakan
        TF-IDF dan Multinomial Naïve Bayes.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # DATA
        # =====================================

        X = df["Final Text"]

        y = df["Label"]

        # =====================================
        # TF-IDF
        # =====================================

        with st.spinner(
            "Melakukan ekstraksi fitur TF-IDF..."
        ):

            tfidf = TfidfVectorizer()

            X_tfidf = tfidf.fit_transform(X)

        # =====================================
        # SPLIT DATA
        # =====================================

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42
        )

        # =====================================
        # INFO SPLIT
        # =====================================

        split1, split2, split3 = st.columns(3)

        with split1:

            st.metric(
                "Training Data",
                len(y_train)
            )

        with split2:

            st.metric(
                "Testing Data",
                len(y_test)
            )

        with split3:

            st.metric(
                "Vocabulary",
                X_tfidf.shape[1]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # TRAINING
        # =====================================

        with st.spinner(
            "Training Naïve Bayes..."
        ):

            model = MultinomialNB()

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

        # =====================================
        # METRIK
        # =====================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        # =====================================
        # KPI METRIK
        # =====================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{precision:.4f}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{recall:.4f}"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # RINGKASAN MODEL
        # =====================================

        st.markdown("""
        <div class="card">

        <h3>📊 Ringkasan Model</h3>

        <ul>
        <li>Algoritma : Multinomial Naïve Bayes</li>
        <li>Ekstraksi Fitur : TF-IDF</li>
        <li>Data Split : 80% Training - 20% Testing</li>
        <li>Random State : 42</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # PREDIKSI SAMPLE
        # =====================================

        hasil_df = pd.DataFrame({

            "Actual": y_test.values,
            "Prediction": y_pred

        })

        st.subheader(
            "📋 Hasil Prediksi Testing"
        )

        st.dataframe(
            hasil_df,
            use_container_width=True,
            height=300
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # SIMPAN MODEL
        # =====================================

        joblib.dump(
            model,
            "model_naive_bayes.pkl"
        )

        joblib.dump(
            tfidf,
            "tfidf_vectorizer.pkl"
        )

        st.success(
            "✅ Model berhasil dilatih dan disimpan"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # CONFUSION MATRIX
        # =====================================

        st.markdown("""
        <div class="card">
        <h3>📊 Confusion Matrix</h3>
        <p>
        Visualisasi hasil prediksi model
        terhadap data testing.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(7,5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            linewidths=1,
            linecolor="white"
        )

        plt.title(
            "Confusion Matrix"
        )

        plt.xlabel(
            "Prediksi"
        )

        plt.ylabel(
            "Aktual"
        )

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # CLASSIFICATION REPORT
        # =====================================

        st.markdown("""
        <div class="card">
        <h3>📄 Classification Report</h3>
        <p>
        Evaluasi lengkap precision,
        recall, f1-score dan support.
        </p>
        </div>
        """, unsafe_allow_html=True)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # VISUALISASI METRIK
        # =====================================

        st.markdown("""
        <div class="card">
        <h3>📈 Visualisasi Performa Model</h3>
        </div>
        """, unsafe_allow_html=True)

        metric_df = pd.DataFrame({

            "Metric":[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            "Value":[
                accuracy,
                precision,
                recall,
                f1
            ]

        })

        fig2, ax2 = plt.subplots(
            figsize=(8,4)
        )

        sns.barplot(
            data=metric_df,
            x="Metric",
            y="Value",
            ax=ax2
        )

        plt.ylim(0,1)

        plt.ylabel(
            "Score"
        )

        plt.xlabel(
            ""
        )

        plt.title(
            "Performa Model"
        )

        st.pyplot(fig2)

        st.markdown("<br>", unsafe_allow_html=True)

            # =====================================
        # STATUS MODEL
        # =====================================

        if accuracy >= 0.90:

            st.success(
                "🔥 Model memiliki performa sangat baik."
            )

        elif accuracy >= 0.80:

            st.info(
                "✅ Model memiliki performa baik."
            )

        else:

            st.warning(
                "⚠️ Model masih perlu ditingkatkan."
            )

    # =====================================================
    # MENU PREDIKSI
    # =====================================================


    # =====================================================
    # MENU ABOUT
    # =====================================================

    elif menu == "ℹ️ About":

        st.markdown("""
        <div class="card">
        <h2>ℹ️ About Developer</h2>
        <p>Informasi pengembang aplikasi.</p>
        </div>
        """, unsafe_allow_html=True)

        c1,c2=st.columns([1,2])
        with c2:
            st.markdown("""
### 👨‍💻 Developer Profile

**NOBP** : 22101152630058

**Nama** : FARHAD ABDILLAH DARNAZ

**Jurusan** : TEKNIK INFORMATIKA

Aplikasi ini dibuat sebagai implementasi algoritma **Naïve Bayes** untuk klasifikasi tingkat kejahatan pada Polres Pasaman menggunakan Python, Streamlit, dan TF‑IDF.
""")

    elif menu == "🔍 Prediksi":

        st.markdown("""
        <div class="card">
        <h2>🔍 Prediksi Tingkat Kejahatan</h2>

        <p>
        Masukkan judul berita kriminal,
        sistem akan melakukan preprocessing
        dan klasifikasi secara otomatis.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        try:
            stop_words = set(stopwords.words("indonesian"))
        except:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("indonesian"))

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        malam_keywords = [
            "malam",
            "subuh",
            "dini hari",
            "tengah malam",
            "larut malam",
            "jam 1",
            "jam 2",
            "jam 3",
            "jam 4",
            "jam 5"
        ]

    # =====================================
    # LOAD MODEL
    # =====================================

    try:

        model = joblib.load(
            "model_naive_bayes.pkl"
        )

        tfidf = joblib.load(
            "tfidf_vectorizer.pkl"
        )

        input_text = st.text_area(
            "Masukkan Judul Berita",
            height=150,
            placeholder="Contoh: Polisi menangkap pelaku pencurian pada dini hari..."
        )

        if st.button("🚀 Prediksi"):

            if not input_text.strip():

                st.warning(
                    "Masukkan judul berita terlebih dahulu."
                )

            else:

                input_lower = input_text.lower()

                detected = False

                # ==========================
                # RULE BASED
                # ==========================

                for keyword in malam_keywords:

                    if keyword in input_lower:

                        prediction = "Kasus Malam"
                        detected = True
                        break

                # ==========================
                # MACHINE LEARNING
                # ==========================

                if not detected:

                    text = input_text.lower()

                    text = re.sub(
                        r"[^\w\s]",
                        "",
                        text
                    )

                    tokens = text.split()

                    tokens = [
                        word
                        for word in tokens
                        if word not in stop_words
                    ]

                    tokens = [
                        stemmer.stem(word)
                        for word in tokens
                    ]

                    final_text = " ".join(tokens)

                    vector = tfidf.transform(
                        [final_text]
                    )

                    prediction = model.predict(
                        vector
                    )[0]

                st.success(
                    f"Hasil Prediksi : {prediction}"
                )

    except:

        st.error("""
Model belum tersedia.

Silakan jalankan menu 🤖 Klasifikasi sekali
agar file berikut dibuat:

• model_naive_bayes.pkl
• tfidf_vectorizer.pkl
""")

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # PIE CHART DISTRIBUSI
        # =====================================

        st.markdown("""
        <div class="card">
        <h3>🥧 Distribusi Label Dataset</h3>
        </div>
        """, unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(
            figsize=(6,6)
        )

        distribusi = df["Label"].value_counts()

        ax3.pie(
            distribusi.values,
            labels=distribusi.index,
            autopct="%1.1f%%"
        )

        ax3.set_title(
            "Distribusi Kasus"
        )

        st.pyplot(fig3)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # DOWNLOAD REPORT
        # =====================================

        csv_report = report_df.to_csv(
            index=True
        )

        st.download_button(
            label="📥 Download Classification Report",
            data=csv_report,
            file_name="classification_report.csv",
            mime="text/csv"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # DOWNLOAD DATASET
        # =====================================

        csv_dataset = df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Dataset Hasil",
            data=csv_dataset,
            file_name="dataset_hasil.csv",
            mime="text/csv"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h3>📚 Top 20 Kata Terbanyak</h3>
        <p>
        Kata yang paling sering muncul setelah preprocessing.
        </p>
        </div>
        """, unsafe_allow_html=True)

        all_words = []

        for text in df["Final Text"]:

            all_words.extend(
                text.split()
            )

        word_freq = pd.Series(
            all_words
        ).value_counts().head(20)

        freq_df = pd.DataFrame({

            "Kata": word_freq.index,
            "Frekuensi": word_freq.values

        })

        st.dataframe(
            freq_df,
            use_container_width=True
        )
