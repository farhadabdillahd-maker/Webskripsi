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

from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from datetime import datetime
from io import BytesIO


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
# HERO LANDING PAGE
# =====================================================
def show_home():
    show_home()
    st.stop()

if menu is None:
    st.stop()


uploaded_file = None

if "uploaded_dataset" not in st.session_state:
    st.session_state.uploaded_dataset = None

if menu == "Upload Dataset":
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"],
        key="dashboard_upload"
    )
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        st.session_state.uploaded_dataset = uploaded_file

if menu in ["Preprocessing","Klasifikasi"]:
    uploaded_file = st.session_state.uploaded_dataset



# =====================================================
# MENU PREDIKSI TANPA UPLOAD DATASET
# =====================================================
if menu == "Prediksi" and uploaded_file is None:

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

                # ================= PDF SURAT =================
                def generate_police_pdf(judul, hasil):
                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=A4)
                    w, h = A4

                    try:
                        c.drawImage(ImageReader("assets/logo_polri.png"),1.5*cm,h-3.7*cm,width=2.4*cm,height=2.4*cm,mask='auto')
                    except:
                        pass
                    try:
                        c.drawImage(ImageReader("assets/logo_polda_sumbar.png"),w-3.9*cm,h-3.7*cm,width=2.4*cm,height=2.4*cm,mask='auto')
                    except:
                        pass

                    c.setFont("Helvetica-Bold",12)
                    c.drawCentredString(w/2,h-1.5*cm,"KEPOLISIAN NEGARA REPUBLIK INDONESIA")
                    c.drawCentredString(w/2,h-2.1*cm,"DAERAH SUMATERA BARAT")
                    c.drawCentredString(w/2,h-2.7*cm,"RESOR PASAMAN")
                    c.setFont("Helvetica",10)
                    c.drawCentredString(w/2,h-3.3*cm,"Jln. Jend. Sudirman No. 1 Lubuk Sikaping 26311")
                    c.setLineWidth(1.2)
                    c.line(1.5*cm,h-3.75*cm,w-1.5*cm,h-3.75*cm)
                    c.setLineWidth(0.5)
                    c.line(1.5*cm,h-3.9*cm,w-1.5*cm,h-3.9*cm)

                    nomor = "B/001/RESKRIM/%s" % datetime.now().strftime("%m/%Y")
                    tanggal = datetime.now().strftime("%d %B %Y")

                    y = h-4.5*cm
                    c.setFont("Helvetica-Bold",14)
                    c.drawCentredString(w/2,y,"LAPORAN HASIL KLASIFIKASI")
                    y -= 1*cm

                    x0=2*cm
                    table_w=w-4*cm
                    row_h=0.8*cm
                    col1=6*cm

                    c.setFont("Helvetica-Bold",11)
                    c.rect(x0,y-row_h,table_w,row_h)
                    c.line(x0+col1,y,x0+col1,y-row_h)
                    c.drawCentredString(x0+col1/2,y-0.55*cm,"Parameter")
                    c.drawCentredString(x0+col1+(table_w-col1)/2,y-0.55*cm,"Keterangan")

                    rows=[
                        ("Nomor Surat",nomor),
                        ("Input Teks",judul[:90]),
                        ("Hasil Prediksi",hasil),
                    ]
                    c.setFont("Helvetica",11)
                    yy=y-row_h
                    for p,v in rows:
                        c.rect(x0,yy-row_h,table_w,row_h)
                        c.line(x0+col1,yy,x0+col1,yy-row_h)
                        c.drawString(x0+0.2*cm,yy-0.55*cm,p)
                        c.drawString(x0+col1+0.2*cm,yy-0.55*cm,str(v))
                        yy-=row_h
                    y=yy-1*cm
                    c.drawString(2*cm,y,"Demikian laporan hasil klasifikasi ini dibuat untuk dipergunakan sebagaimana mestinya.")
                    y -= 2*cm
                    c.drawRightString(w-2*cm,y,"Pasaman, "+tanggal)
                    y -= 0.8*cm
                    c.drawRightString(w-2*cm,y,"Kepala Sat Reskrim")
                    y -= 2.5*cm
                    c.drawRightString(w-2*cm,y,"(................................)")
                    c.save()
                    pdf = buffer.getvalue()
                    buffer.close()
                    return pdf

                pdf = generate_police_pdf(input_text, prediction)

                st.download_button(
                    "📄 Download Surat Hasil Prediksi (PDF)",
                    data=pdf,
                    file_name="Surat_Hasil_Klasifikasi.pdf",
                    mime="application/pdf"
                )

            else:
                st.warning("Masukkan judul berita terlebih dahulu.")
    except:
        st.error("Model belum tersedia. Jalankan menu Klasifikasi terlebih dahulu untuk membuat model.")



# =====================================================
# MENU ABOUT
# =====================================================
if menu == "About":
    st.markdown("""
    <div class="card">
        <h1 style="text-align:center;">ℹ️ ABOUT APLIKASI</h1>
        <p style="text-align:center;">Informasi pengembang, penelitian, dan teknologi yang digunakan.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📘 Judul Penelitian")
    st.markdown("""
    <div class="card">
    <b>Penerapan Machine Learning Menggunakan Algoritma Naïve Bayes Untuk Klasifikasi Tingkat Kejahatan di Polres Pasaman</b>
    </div>
    """, unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Perkenalan")
        from pathlib import Path
        foto_path = Path(__file__).parent / "assets" / "FOTO.png"
        if foto_path.exists():
            st.image(str(foto_path), width=180)
        else:
            st.warning("assets/FOTO.png tidak ditemukan")
        st.markdown("""
        <div class="card">
        <b>Nama</b> : Farhad Abdillah Darnaz<br><br>
        <b>NOBP</b> : 22101152630058<br><br>
        <b>Program Studi</b> : Teknik Informatika<br><br>
        <b>Universitas</b> : Universitas Putra Indonesia YPTK Padang
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🛠️ Aplikasi yang Digunakan")
        st.markdown("""
        <div class="card">
        • Python<br>
        • Streamlit<br>
        • Pandas<br>
        • Scikit-learn<br>
        • Naïve Bayes<br>
        • TF-IDF Vectorizer<br>
        • Sastrawi<br>
        • NLTK<br>
        • Matplotlib<br>
        • Seaborn<br>
        • Joblib
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ℹ️ Informasi Aplikasi")
    st.info("Aplikasi ini dibuat sebagai media klasifikasi tingkat kejahatan berdasarkan judul berita menggunakan algoritma Naïve Bayes.")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================

if menu in ["Upload Dataset","Preprocessing","Klasifikasi"]:

    if uploaded_file is None:
        st.info("Silakan upload dataset CSV untuk menggunakan menu ini.")
        st.stop()


    # Pastikan file dapat dibaca ulang setiap perpindahan menu
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

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

    if menu == "Upload Dataset":

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

    elif menu == "Preprocessing":

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

    elif menu == "Klasifikasi":

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

    elif menu == "About":

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

    elif menu == "Prediksi":

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
