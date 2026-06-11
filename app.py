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

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="Klasifikasi Tingkat Kejahatan",
    page_icon="🚔",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1, h2, h3 {
    color: #0f172a;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# TITLE
# ====================================
st.title("🚔 KLASIFIKASI TINGKAT KEJAHATAN")
st.subheader("Naïve Bayes - Polres Pasaman")

st.write("""
Sistem Machine Learning menggunakan algoritma Naïve Bayes
untuk klasifikasi tingkat kejahatan berdasarkan berita kriminal.
""")

# ====================================
# SIDEBAR MENU
# ====================================
menu = st.sidebar.radio(
    "📌 Pilih Menu",
    [
        "Upload Dataset",
        "Preprocessing",
        "Klasifikasi Naïve Bayes",
        "Prediksi"
    ]
)

# ====================================
# UPLOAD DATASET
# ====================================
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# ====================================
# JIKA FILE ADA
# ====================================
if uploaded_file is not None:

    # ====================================
    # READ CSV
    # ====================================
    df = pd.read_csv(uploaded_file)

    # ====================================
    # VALIDASI KOLOM
    # ====================================
    if "Judul Media Nasional" not in df.columns:

        st.error("Kolom 'Judul Media Nasional' tidak ditemukan!")

        st.stop()

    # ====================================
    # AMBIL KOLOM
    # ====================================
    df = df[["Judul Media Nasional"]]

    # ====================================
    # STOPWORD & STEMMER
    # ====================================
    # ====================================
# STOPWORD & STEMMER
# ====================================
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

    # ====================================
    # CASE FOLDING
    # ====================================
    def case_folding(text):

        return str(text).lower()

    # ====================================
    # TOKENIZING
    # ====================================
    def tokenizing(text):

        text = re.sub(r'[^\w\s]', '', text)

        return text.split()

    # ====================================
    # STOPWORD REMOVAL
    # ====================================
    def stopword_removal(tokens):

        return [
            word for word in tokens
            if word not in stop_words
        ]

    # ====================================
    # STEMMING
    # ====================================
    def stemming(tokens):

        return [
            stemmer.stem(word)
            for word in tokens
        ]

    # ====================================
    # AUTO LABELING
    # ====================================
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

    # ====================================
    # PREPROCESSING
    # ====================================
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

    # ====================================
    # MENU UPLOAD DATASET
    # ====================================
    if menu == "Upload Dataset":

        st.header("📂 Dataset Awal")

        st.dataframe(
            df[["Judul Media Nasional"]]
        )

        st.success("Dataset berhasil diupload!")

    # ====================================
    # MENU PREPROCESSING
    # ====================================
    elif menu == "Preprocessing":

        st.header("🧹 Preprocessing Text")

        # CASE FOLDING
        st.subheader("1. Case Folding")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Case Folding"
                ]
            ]
        )

        # TOKENIZING
        st.subheader("2. Tokenizing")

        st.dataframe(
            df[
                [
                    "Case Folding",
                    "Tokenizing"
                ]
            ]
        )

        # STOPWORD REMOVAL
        st.subheader("3. Stopword Removal")

        st.dataframe(
            df[
                [
                    "Tokenizing",
                    "Stopword Removal"
                ]
            ]
        )

        # STEMMING
        st.subheader("4. Stemming")

        st.dataframe(
            df[
                [
                    "Stopword Removal",
                    "Stemming"
                ]
            ]
        )

        # LABELING
        st.subheader("5. Pelabelan Dataset")

        st.dataframe(
            df[
                [
                    "Judul Media Nasional",
                    "Label"
                ]
            ]
        )

    # ====================================
    # MENU KLASIFIKASI
    # ====================================
    elif menu == "Klasifikasi Naïve Bayes":

        st.header("🤖 Klasifikasi Naïve Bayes")

        X = df["Final Text"]

        y = df["Label"]

        # ====================================
        # TF-IDF
        # ====================================
        tfidf = TfidfVectorizer()

        X_tfidf = tfidf.fit_transform(X)

        # ====================================
        # SPLIT DATA
        # ====================================
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42
        )

        # ====================================
        # MODEL
        # ====================================
        model = MultinomialNB()

        # ====================================
        # TRAINING
        # ====================================
        model.fit(X_train, y_train)

        # ====================================
        # PREDIKSI
        # ====================================
        y_pred = model.predict(X_test)

        # ====================================
        # METRIK
        # ====================================
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        # ====================================
        # TAMPILKAN METRIK
        # ====================================
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{accuracy:.2f}"
        )

        col2.metric(
            "Precision",
            f"{precision:.2f}"
        )

        col3.metric(
            "Recall",
            f"{recall:.2f}"
        )

        col4.metric(
            "F1-Score",
            f"{f1:.2f}"
        )

        # ====================================
        # CONFUSION MATRIX
        # ====================================
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(figsize=(5,4))

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

        # ====================================
        # CLASSIFICATION REPORT
        # ====================================
        st.subheader("📄 Classification Report")

        report = classification_report(
            y_test,
            y_pred
        )

        st.text(report)

        # ====================================
        # SIMPAN MODEL
        # ====================================
        joblib.dump(
            model,
            "model_naive_bayes.pkl"
        )

        joblib.dump(
            tfidf,
            "tfidf_vectorizer.pkl"
        )

        st.success("Model berhasil disimpan!")

    # ====================================
    # MENU PREDIKSI
    # ====================================
    elif menu == "Prediksi":

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

                # RULE-BASED MALAM
                for keyword in malam_keywords:

                    if keyword in input_lower:

                        prediction = "Kasus Malam"

                        detected = True

                        break

                # JIKA TIDAK ADA KEYWORD
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

# ====================================
# JIKA BELUM UPLOAD
# ====================================
else:

    st.info(
        "Silakan upload dataset CSV terlebih dahulu."
    )
