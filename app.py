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
    page_icon="🚔",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

/* IMPORT FONT */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* HIDE STREAMLIT */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #f8fbff 0%,
        #eef4ff 100%
    );
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
    width: 320px !important;
}

/* SIDEBAR CONTENT */
.sidebar-content {
    padding-top: 10px;
}

/* LOGO */
.logo-box {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 40px;
}

.logo-icon {
    width: 65px;
    height: 65px;
    background: linear-gradient(135deg,#2563eb,#3b82f6);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    color: white;
    box-shadow: 0px 10px 25px rgba(37,99,235,0.3);
}

.logo-text h1 {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    line-height: 1;
}

.logo-text p {
    margin: 0;
    color: #64748b;
    font-size: 15px;
}

/* MENU TITLE */
.menu-title {
    font-size: 14px;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* CARD */
.custom-card {
    background: rgba(255,255,255,0.9);
    padding: 35px;
    border-radius: 35px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.04);
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
}

/* TITLE */
.main-title {
    font-size: 72px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 10px;
}

.sub-title {
    font-size: 28px;
    color: #1e293b;
    font-weight: 700;
    margin-bottom: 20px;
}

.desc {
    color: #64748b;
    font-size: 18px;
    line-height: 1.8;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color: white;
    border: none;
    border-radius: 18px;
    height: 55px;
    width: 100%;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s;
    box-shadow: 0px 10px 25px rgba(37,99,235,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb
    );
}

/* METRIC */
[data-testid="metric-container"] {
    background: white;
    border-radius: 25px;
    padding: 25px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.04);
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 25px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    padding: 20px;
    border-radius: 25px;
    border: 1px solid #e5e7eb;
}

/* TEXT AREA */
textarea {
    border-radius: 18px !important;
}

/* SUCCESS */
.stSuccess {
    border-radius: 18px;
}

/* ALERT */
.stAlert {
    border-radius: 20px;
}

/* RADIO */
.stRadio > div {
    gap: 12px;
}

/* SECTION TITLE */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #94a3b8;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* INFO BOX */
.info-box {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #dbeafe
    );
    border-radius: 30px;
    padding: 35px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 20px;
}

.info-text h2 {
    color: #0f172a;
    font-size: 30px;
    margin-bottom: 10px;
}

.info-text p {
    color: #64748b;
    font-size: 18px;
}

.info-icon {
    font-size: 80px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR HEADER
# =========================================
st.sidebar.markdown("""
<div class="sidebar-content">

<div class="logo-box">

<div class="logo-icon">
🛡️
</div>

<div class="logo-text">
<h1>KLASIFIKASI</h1>
<p>TINGKAT KEJAHATAN</p>
</div>

</div>

<div class="menu-title">
MENU
</div>

</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR MENU
# =========================================
menu = st.sidebar.radio(
    "",
    [
        "📂 Upload Dataset",
        "🧹 Preprocessing",
        "🤖 Klasifikasi Naïve Bayes",
        "🔍 Prediksi"
    ]
)

# =========================================
# FILE UPLOADER
# =========================================
st.sidebar.markdown("""
<div class="section-title">
DATASET
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="custom-card">

<div style="display:flex;align-items:center;gap:30px;">

<div style="
width:130px;
height:130px;
border-radius:35px;
background:white;
display:flex;
align-items:center;
justify-content:center;
font-size:70px;
box-shadow:0px 10px 25px rgba(0,0,0,0.05);
">
🚔
</div>

<div>

<div class="main-title">
KLASIFIKASI TINGKAT KEJAHATAN
</div>

<div style="
width:90px;
height:8px;
border-radius:20px;
background:#2563eb;
margin-bottom:20px;
"></div>

<div class="sub-title">
Naïve Bayes - Polres Pasaman
</div>

<div class="desc">
Sistem Machine Learning menggunakan algoritma Naïve Bayes
untuk klasifikasi tingkat kejahatan berdasarkan berita kriminal.
</div>

</div>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================
# JIKA FILE ADA
# =========================================
if uploaded_file is not None:

    # READ CSV
    df = pd.read_csv(uploaded_file)

    # VALIDASI KOLOM
    if "Judul Media Nasional" not in df.columns:

        st.error("Kolom 'Judul Media Nasional' tidak ditemukan!")

        st.stop()

    # AMBIL KOLOM
    df = df[["Judul Media Nasional"]]

    # STOPWORD
    stop_words = set(stopwords.words('indonesian'))

    # STEMMER
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # CASE FOLDING
    def case_folding(text):

        return str(text).lower()

    # TOKENIZING
    def tokenizing(text):

        text = re.sub(r'[^\w\s]', '', text)

        return text.split()

    # STOPWORD REMOVAL
    def stopword_removal(tokens):

        return [
            word for word in tokens
            if word not in stop_words
        ]

    # STEMMING
    def stemming(tokens):

        return [
            stemmer.stem(word)
            for word in tokens
        ]

    # KEYWORD MALAM
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

    # AUTO LABEL
    def auto_label(text):

        text = str(text).lower()

        for keyword in malam_keywords:

            if keyword in text:

                return "Kasus Malam"

        return "Kasus Umum"

    # PREPROCESSING
    df["Case Folding"] = df["Judul Media Nasional"].apply(
        case_folding
    )

    df["Tokenizing"] = df["Case Folding"].apply(
        tokenizing
    )

    df["Stopword Removal"] = df["Tokenizing"].apply(
        stopword_removal
    )

    df["Stemming"] = df["Stopword Removal"].apply(
        stemming
    )

    df["Final Text"] = df["Stemming"].apply(
        lambda x: " ".join(x)
    )

    df["Label"] = df["Judul Media Nasional"].apply(
        auto_label
    )

    # =========================================
    # MENU DATASET
    # =========================================
    if menu == "📂 Upload Dataset":

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

        st.header("📂 Dataset Awal")

        st.dataframe(
            df[["Judul Media Nasional"]],
            use_container_width=True
        )

        st.success("Dataset berhasil diupload!")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # PREPROCESSING
    # =========================================
    elif menu == "🧹 Preprocessing":

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

        st.header("🧹 Preprocessing Text")

        st.subheader("1. Case Folding")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Case Folding"
                ]
            ],
            use_container_width=True
        )

        st.subheader("2. Tokenizing")

        st.dataframe(
            df[
                [
                    "Case Folding",
                    "Tokenizing"
                ]
            ],
            use_container_width=True
        )

        st.subheader("3. Stopword Removal")

        st.dataframe(
            df[
                [
                    "Tokenizing",
                    "Stopword Removal"
                ]
            ],
            use_container_width=True
        )

        st.subheader("4. Stemming")

        st.dataframe(
            df[
                [
                    "Stopword Removal",
                    "Stemming"
                ]
            ],
            use_container_width=True
        )

        st.subheader("5. Pelabelan Dataset")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Label"
                ]
            ],
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # KLASIFIKASI
    # =========================================
    elif menu == "🤖 Klasifikasi Naïve Bayes":

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

        st.header("🤖 Klasifikasi Naïve Bayes")

        X = df["Final Text"]

        y = df["Label"]

        # TF-IDF
        tfidf = TfidfVectorizer()

        X_tfidf = tfidf.fit_transform(X)

        # SPLIT DATA
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42
        )

        # MODEL
        model = MultinomialNB()

        # TRAINING
        model.fit(X_train, y_train)

        # PREDIKSI
        y_pred = model.predict(X_test)

        # METRIK
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        # METRIC UI
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")

        # CONFUSION MATRIX
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(figsize=(6,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )

        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")

        st.pyplot(fig)

        # REPORT
        st.subheader("📄 Classification Report")

        report = classification_report(
            y_test,
            y_pred
        )

        st.text(report)

        # SAVE MODEL
        joblib.dump(
            model,
            "model_naive_bayes.pkl"
        )

        joblib.dump(
            tfidf,
            "tfidf_vectorizer.pkl"
        )

        st.success("Model berhasil disimpan!")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # PREDIKSI
    # =========================================
    elif menu == "🔍 Prediksi":

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

        st.header("🔍 Prediksi Tingkat Kejahatan")

        try:

            model = joblib.load(
                "model_naive_bayes.pkl"
            )

            tfidf = joblib.load(
                "tfidf_vectorizer.pkl"
            )

            input_text = st.text_area(
                "Masukkan Judul Berita"
            )

            if st.button("Prediksi"):

                input_lower = input_text.lower()

                detected = False

                for keyword in malam_keywords:

                    if keyword in input_lower:

                        prediction = "Kasus Malam"

                        detected = True

                        break

                if not detected:

                    text = input_text.lower()

                    text = re.sub(
                        r'[^\w\s]',
                        '',
                        text
                    )

                    tokens = text.split()

                    tokens = [
                        word for word in tokens
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
                    f"Hasil Prediksi: {prediction}"
                )

        except:

            st.warning(
                "Silakan lakukan klasifikasi terlebih dahulu!"
            )

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# BELUM UPLOAD
# =========================================
else:

    st.markdown("""
    <div class="info-box">

    <div class="info-text">
        <h2>Silakan upload dataset CSV terlebih dahulu.</h2>
        <p>
        Pastikan file berformat CSV dan sesuai dengan struktur data.
        </p>
    </div>

    <div class="info-icon">
        📄
    </div>

    </div>
    """, unsafe_allow_html=True)
