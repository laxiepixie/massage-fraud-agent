import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def jalankan_pelatihan_model():
    print("=== MENGHIDUPKAN MESIN PELATIHAN AI ===")
    
    file_data_bersih = 'data/processed/clean_banking_dataset_v2.csv'
    folder_model = 'models/'
    
    try:
        df = pd.read_csv(file_data_bersih)
        df = df.dropna(subset=['clean_text', 'label'])
        print(f"[*] Memuat {df.shape[0]} baris data siap latih.")
        
        X = df['clean_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[*] Data dipecah: {X_train.shape[0]} Training, {X_test.shape[0]} Testing.")
        
        print("[*] Melakukan Vektorisasi TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        os.makedirs(folder_model, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(folder_model, 'tfidf_vectorizer.pkl'))
        
       
        print("\n[*] Menyuapkan data ke Algoritma (Logistic Regression)...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        print("[*] Menguji performa AI pada data Ujian (Testing)...")
        y_pred = model.predict(X_test_vec)
        
        print("\n=== RAPOR KECERDASAN AI (CLASSIFICATION REPORT) ===")
        print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud/Banking (1)']))
        
        joblib.dump(model, os.path.join(folder_model, 'fraud_detector_model.pkl'))
        print(f"\n[SUCCESS] Model otak AI berhasil disimpan di folder {folder_model}")
        print("=== PROSES TRAINING SELESAI. AGEN AI SIAP DIGUNAKAN! ===")
        
    except Exception as e:
        print(f"[!] ERROR SISTEM: {e}")

if __name__ == "__main__":
    jalankan_pelatihan_model()