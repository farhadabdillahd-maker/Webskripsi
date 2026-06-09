import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
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

# # =========================
# # DOWNLOAD NLTK
# # =========================
# nltk.download('punkt')
# nltk.download('stopwords')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Klasifikasi Tingkat Kejahatan",
    page_icon="🚔",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

h1, h2, h3 {
    color: #1f3c88;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🚔 KLASIFIKASI TINGKAT KEJAHATAN")
st.subheader("Naïve Bayes - Polres Pasaman")

st.write("""
Sistem Machine Learning menggunakan algoritma Naïve Bayes
untuk klasifikasi tingkat kejahatan berdasarkan berita kriminal.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Dashboard",
        "Preprocessing",
        "Training & Evaluasi",
        "Prediksi"
    ]
)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =========================
    # AMBIL KOLOM
    # =========================
    if "Judul Media Nasional" not in df.columns:
        st.error("Kolom 'Judul Media Nasional' tidak ditemukan!")
        st.stop()

    df = df[["Judul Media Nasional"]]

    # =========================
    # AUTO LABELING
    # =========================
    malam_keywords = [
        "malam",
        "subuh",
        "dini hari",
        "tengah malam",
        "jam 2",
        "jam 3",
        "jam 4"
    ]

    def auto_label(text):
        text = str(text).lower()

        for keyword in malam_keywords:
            if keyword in text:
                return "Kasus Malam"

        return "Kasus Umum"

    df["Label"] = df["Judul Media Nasional"].apply(auto_label)

    # =========================
    # PREPROCESSING
    # =========================
    stop_words = set(stopwords.words('indonesian'))

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocessing(text):

        # case folding
        text = text.lower()

        # hapus angka
        text = re.sub(r'\d+', '', text)

        # hapus tanda baca
        text = re.sub(r'[^\w\s]', '', text)

        # tokenizing
        tokens = text.split()

        # stopword removal
        tokens = [word for word in tokens if word not in stop_words]

        # stemming
        tokens = [stemmer.stem(word) for word in tokens]

        return " ".join(tokens)

    df["Preprocessing"] = df["Judul Media Nasional"].apply(preprocessing)

    # =========================
    # DASHBOARD
    # =========================
    if menu == "Dashboard":

        st.header("📊 Dashboard Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Jumlah Data", len(df))

        with col2:
            st.metric("Jumlah Kolom", len(df.columns))

        st.write("### Dataset")
        st.dataframe(df)

        # =========================
        # DISTRIBUSI KELAS
        # =========================
        st.write("### Distribusi Kelas")

        label_counts = df["Label"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            label_counts,
            labels=label_counts.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig)

    # =========================
    # PREPROCESSING PAGE
    # =========================
    elif menu == "Preprocessing":

        st.header("🧹 Hasil Preprocessing")

        preview_df = pd.DataFrame({
            "Teks Asli": df["Judul Media Nasional"],
            "Hasil Preprocessing": df["Preprocessing"],
            "Label": df["Label"]
        })

        st.dataframe(preview_df)

    # =========================
    # TRAINING
    # =========================
    elif menu == "Training & Evaluasi":

        st.header("🤖 Training Naïve Bayes")

        X = df["Preprocessing"]
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

        # =========================
        # METRIC DISPLAY
        # =========================
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")

        # =========================
        # CONFUSION MATRIX
        # =========================
        st.write("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

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

        # =========================
        # CLASSIFICATION REPORT
        # =========================
        st.write("### Classification Report")

        report = classification_report(y_test, y_pred)

        st.text(report)

        # =========================
        # SAVE MODEL
        # =========================
        joblib.dump(model, 'model/model_naive_bayes.pkl')
        joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

        st.success("Model berhasil disimpan!")

    # =========================
    # PREDIKSI
    # =========================
    elif menu == "Prediksi":

        st.header("🔍 Prediksi Tingkat Kejahatan")

        try:
            model = joblib.load('model/model_naive_bayes.pkl')
            tfidf = joblib.load('model/tfidf_vectorizer.pkl')

            input_text = st.text_area(
                "Masukkan Judul Berita"
            )

            if st.button("Prediksi"):

                processed_text = preprocessing(input_text)

                vector = tfidf.transform([processed_text])

                prediction = model.predict(vector)[0]

                st.success(f"Hasil Prediksi: {prediction}")

        except:
            st.warning("Silakan training model terlebih dahulu!")

else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
