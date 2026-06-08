import joblib
import re
import os

def bersihkan_dan_normalisasi(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks.strip()

def mulai_investigasi():
    print("=== MENGHIDUPKAN AGEN INVESTIGATOR PERBANKAN ===")
    
    jalur_vektor = 'models/tfidf_vectorizer.pkl'
    jalur_model = 'models/fraud_detector_model.pkl'
    
    if not os.path.exists(jalur_vektor) or not os.path.exists(jalur_model):
        print("[!] ERROR: Otak AI tidak ditemukan. Jalankan model_engine.py dulu!")
        return
        
    print("[*] Memuat kamus kosakata (TF-IDF)...")
    vectorizer = joblib.load(jalur_vektor)
    
    print("[*] Memuat memori logika (Logistic Regression)...")
    model = joblib.load(jalur_model)
    
    print("\n[SUCCESS] Agen siap menerima laporan!")
    print("-" * 50)
    
    while True:
        pesan_baru = input("\n[?] Masukkan pesan SMS yang ingin dicek (ketik 'exit' untuk keluar):\n> ")
        
        if pesan_baru.lower() == 'exit':
            print("Mematikan agen... Selamat beristirahat, Arsitek!")
            break
            
        if not pesan_baru.strip():
            continue
            
        pesan_bersih = bersihkan_dan_normalisasi(pesan_baru)
        
        pesan_vektor = vectorizer.transform([pesan_bersih])
        
        probabilitas = model.predict_proba(pesan_vektor)[0]
        prob_normal = probabilitas[0] * 100
        prob_fraud = probabilitas[1] * 100
        
        print("\n--- HASIL INVESTIGASI AGEN ---")
        print(f"Teks Bersih : '{pesan_bersih}'")
        if prob_fraud > 60.0:
            print(f"[!] STATUS    : 🚨 DIBLOKIR! (PENIPUAN / FRAUD)")
            print(f"[!] TINGKAT KEYAKINAN : {prob_fraud:.2f}%")
        else:
            print(f"[*] STATUS    : ✅ AMAN (NORMAL)")
            print(f"[*] TINGKAT KEYAKINAN : {prob_normal:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    mulai_investigasi()