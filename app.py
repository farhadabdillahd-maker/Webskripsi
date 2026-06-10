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
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

/* GOOGLE FONT */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
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
        #f7faff 0%,
        #eef4ff 100%
    );
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
    width: 330px !important;
}

/* SIDEBAR CONTENT */
.sidebar-container {
    padding-top: 10px;
}

/* LOGO */
.logo-container {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 45px;
}

.logo-icon {
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
    box-shadow: 0px 10px 30px rgba(37,99,235,0.35);
}

.logo-title {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}

.logo-sub {
    color: #64748b;
    font-size: 15px;
    margin-top: 4px;
}

/* MENU TEXT */
.menu-label {
    font-size: 14px;
    color: #94a3b8;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* RADIO */
.stRadio > div {
    gap: 12px;
}

/* RADIO BUTTON */
.stRadio label {
    background: transparent;
    padding: 14px 18px;
    border-radius: 18px;
    transition: 0.3s;
    font-weight: 500;
    color: #334155;
}

.stRadio label:hover {
    background: #eff6ff;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 20px;
    border-radius: 24px;
}

/* MAIN CARD */
.main-card {
    background: rgba(255,255,255,0.85);
    padding: 40px;
    border-radius: 40px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.04);
    backdrop-filter: blur(10px);
}

/* HERO */
.hero {
    display: flex;
    align-items: center;
    gap: 35px;
}

/* HERO ICON */
.hero-icon {
    width: 140px;
    height: 140px;
    background: white;
    border-radius: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 75px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
}

/* HERO TITLE */
.hero-title {
    font-size: 74px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 14px;
}

/* BLUE LINE */
.blue-line {
    width: 90px;
    height: 7px;
    background: #2563eb;
    border-radius: 20px;
    margin-bottom: 24px;
}

/* SUBTITLE */
.hero-subtitle {
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 18px;
}

/* DESCRIPTION */
.hero-desc {
    color: #64748b;
    font-size: 19px;
    line-height: 1.8;
}

/* INFO BOX */
.info-box {
    margin-top: 35px;
    background: linear-gradient(
        135deg,
        #eff6ff,
        #dbeafe
    );
    border-radius: 35px;
    padding: 35px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.info-left {
    display: flex;
    align-items: center;
    gap: 25px;
}

.info-circle {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: #2563eb;
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 34px;
    font-weight: bold;
}

.info-title {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

.info-desc {
    color: #64748b;
    font-size: 18px;
    margin-top: 8px;
}

.csv-icon {
    font-size: 90px;
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
    box-shadow: 0px 10px 25px rgba(37,99,235,0.3);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* METRIC */
[data-testid="metric-container"] {
    background: white;
    border-radius: 24px;
    padding: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.04);
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 25px;
    overflow: hidden;
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

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.markdown("""
<div class="sidebar-container">

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

<div class="menu-label">
MENU
</div>

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
# FILE UPLOADER
# =========================================
st.sidebar.markdown("""
<div class="menu-label" style="margin-top:30px;">
DATASET
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# =========================================
# HERO SECTION
# =========================================
st.markdown("""
<div class="main-card">

<div class="hero">

<div class="hero-icon">
🚔
</div>

<div>

<div class="hero-title">
KLASIFIKASI TINGKAT KEJAHATAN
</div>

<div class="blue-line"></div>

<div class="hero-subtitle">
Naïve Bayes - Polres Pasaman
</div>

<div class="hero-desc">
Sistem Machine Learning menggunakan algoritma Naïve Bayes
untuk klasifikasi tingkat kejahatan berdasarkan berita kriminal.
</div>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================
# JIKA BELUM UPLOAD
# =========================================
if uploaded_file is None:

    st.markdown("""
    <div class="info-box">

    <div class="info-left">

    <div class="info-circle">
    i
    </div>

    <div>

    <div class="info-title">
    Silakan upload dataset CSV terlebih dahulu.
    </div>

    <div class="info-desc">
    Pastikan file berformat CSV dan sesuai dengan struktur data.
    </div>

    </div>

    </div>

    <div class="csv-icon">
    📄
    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================
# JIKA FILE ADA
# =========================================
else:

    # READ CSV
    df = pd.read_csv(uploaded_file)

    # VALIDASI
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

    # STOPWORD
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

    # LABEL
    def auto_label(text):

        text = str(text).lower()

        for keyword in malam_keywords:

            if keyword in text:

                return "Kasus Malam"

        return "Kasus Umum"

    # PREPROCESSING
    df["Case Folding"] = df["Judul Media Nasional"].apply(case_folding)

    df["Tokenizing"] = df["Case Folding"].apply(tokenizing)

    df["Stopword Removal"] = df["Tokenizing"].apply(stopword_removal)

    df["Stemming"] = df["Stopword Removal"].apply(stemming)

    df["Final Text"] = df["Stemming"].apply(
        lambda x: " ".join(x)
    )

    df["Label"] = df["Judul Media Nasional"].apply(auto_label)

    # =========================================
    # MENU DATASET
    # =========================================
    if menu == "📂 Upload Dataset":

        st.subheader("📂 Dataset Awal")

        st.dataframe(
            df[["Judul Media Nasional"]],
            use_container_width=True
        )

    # =========================================
    # PREPROCESSING
    # =========================================
    elif menu == "🧹 Preprocessing":

        st.subheader("🧹 Preprocessing Text")

        st.write("### 1. Case Folding")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Case Folding"
                ]
            ],
            use_container_width=True
        )

        st.write("### 2. Tokenizing")

        st.dataframe(
            df[
                [
                    "Case Folding",
                    "Tokenizing"
                ]
            ],
            use_container_width=True
        )

        st.write("### 3. Stopword Removal")

        st.dataframe(
            df[
                [
                    "Tokenizing",
                    "Stopword Removal"
                ]
            ],
            use_container_width=True
        )

        st.write("### 4. Stemming")

        st.dataframe(
            df[
                [
                    "Stopword Removal",
                    "Stemming"
                ]
            ],
            use_container_width=True
        )

        st.write("### 5. Pelabelan Dataset")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Label"
                ]
            ],
            use_container_width=True
        )

    # =========================================
    # KLASIFIKASI
    # =========================================
    elif menu == "📊 Klasifikasi Naïve Bayes":

        st.subheader("📊 Klasifikasi Naïve Bayes")

        X = df["Final Text"]

        y = df["Label"]

        # TF-IDF
        tfidf = TfidfVectorizer()

        X_tfidf = tfidf.fit_transform(X)

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42
        )

        # MODEL
        model = MultinomialNB()

        model.fit(X_train, y_train)

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

        # METRIC
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2f}")

        col2.metric("Precision", f"{precision:.2f}")

        col3.metric("Recall", f"{recall:.2f}")

        col4.metric("F1-Score", f"{f1:.2f}")

        # CM
        st.write("### 📊 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

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
        st.write("### 📄 Classification Report")

        report = classification_report(
            y_test,
            y_pred
        )

        st.text(report)

        # SAVE
        joblib.dump(
            model,
            "model_naive_bayes.pkl"
        )

        joblib.dump(
            tfidf,
            "tfidf_vectorizer.pkl"
        )

        st.success("Model berhasil disimpan!")

    # =========================================
    # PREDIKSI
    # =========================================
    elif menu == "📈 Prediksi":

        st.subheader("📈 Prediksi Tingkat Kejahatan")

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

# CLOSE MAIN CARD
st.markdown("</div>", unsafe_allow_html=True)
