import streamlit as st
import joblib
import re
import os

st.set_page_config(page_title="AI Fraud Investigator", page_icon="🕵️‍♀️")

def bersihkan_dan_normalisasi(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks.strip()

st.title("🕵️‍♀️ Autonomous Fraud Agent")
st.markdown("""
Selamat datang! Agen AI ini dilatih menggunakan algoritma *Logistic Regression* untuk mendeteksi manipulasi psikologis (*Social Engineering*) pada pesan teks. 
Silakan uji coba dengan memasukkan pesan SMS, Email, atau DM Medsos.
""")

@st.cache_resource
def load_ai_brain():
    jalur_vektor = 'models/tfidf_vectorizer.pkl'
    jalur_model = 'models/fraud_detector_model.pkl'
    
    if os.path.exists(jalur_vektor) and os.path.exists(jalur_model):
        vectorizer = joblib.load(jalur_vektor)
        model = joblib.load(jalur_model)
        return vectorizer, model
    else:
        return None, None

vectorizer, model = load_ai_brain()

if vectorizer is None or model is None:
    st.error("⚠️ Sistem gagal memuat memori AI. Pastikan file .pkl ada di folder models/")
else:
    pesan_baru = st.text_area("Masukkan teks yang ingin diinvestigasi:", height=150)
    
    if st.button("Jalankan Investigasi 🔍"):
        if pesan_baru.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            pesan_bersih = bersihkan_dan_normalisasi(pesan_baru)
            pesan_vektor = vectorizer.transform([pesan_bersih])
            probabilitas = model.predict_proba(pesan_vektor)[0]
            
            prob_normal = probabilitas[0] * 100
            prob_fraud = probabilitas[1] * 100
            
            st.divider()
            st.subheader("Hasil Analisis Sistem:")
            
            if prob_fraud > 60.0:
                st.error(f"🚨 **DIBLOKIR! (INDIKASI PENIPUAN)**")
                st.write(f"Tingkat Keyakinan AI: **{prob_fraud:.2f}%**")
                st.progress(int(prob_fraud))
            else:
                st.success(f"✅ **AMAN (KATEGORI NORMAL)**")
                st.write(f"Tingkat Keyakinan AI: **{prob_normal:.2f}%**")
                st.progress(int(prob_normal))
st.markdown("""
    <style>
    /* Mengganti warna latar belakang tombol */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff1a1a;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    /* Mempercantik font judul */
    h1 {
        font-family: 'Courier New', Courier, monospace;
        color: #1e3d59;
    }
    </style>
""", unsafe_allow_html=True)                