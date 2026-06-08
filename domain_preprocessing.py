import pandas as pd
import re
import os
from deep_translator import GoogleTranslator
from tqdm import tqdm 

def bersihkan_dan_normalisasi(teks):
    """Mesin Pencuci Teks (NLP)"""
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks.strip()

def terjemahkan_ke_indonesia(teks):
    """Penerjemah Universal (Auto-Detect -> ID)"""
    if pd.isna(teks) or str(teks).strip() == "":
        return ""
    try:
        return GoogleTranslator(source='auto', target='id').translate(str(teks))
    except Exception:
        return ""  
    
def jalankan_pipeline_multibahasa():
    print("=== MENGHIDUPKAN PABRIK DATA V2.0: GLOBAL THREAT INTELLIGENCE ===")
    
    jalur_internasional = 'data/raw/final_dataset_output.csv'
    jalur_lokal = 'data/raw/dataset_sms_spam_v2.csv' 
    jalur_output = 'data/processed/clean_banking_dataset_v2.csv'
    
    try:
        print("[*] 1. Menggali taktik penipuan global...")
        df_int = pd.read_csv(jalur_internasional)
        
        df_banking = df_int[df_int['scam_type'] == 'banking'].copy()
        
        bahasa_target = ['English', 'Russian', 'French', 'Italian']
        list_sampel = []
        
        for lang in bahasa_target:
            df_lang = df_banking[df_banking['language'] == lang]
            df_sample = df_lang.sample(n=min(200, len(df_lang)), random_state=42)
            list_sampel.append(df_sample)
            print(f"    -> Terambil {len(df_sample)} baris taktik dari bahasa {lang}")
            
        df_scam_global = pd.concat(list_sampel, ignore_index=True)
        
        print("\n[*] 2. Menerjemahkan taktik global ke Bahasa Indonesia...")
        tqdm.pandas(desc="Translating Fraud Data")
        df_scam_global['text_id'] = df_scam_global['text'].progress_apply(terjemahkan_ke_indonesia)
        
        df_fraud_final = pd.DataFrame({'text': df_scam_global['text_id'], 'label': 1})
        df_fraud_final = df_fraud_final[df_fraud_final['text'] != ''] # Buang yang gagal translate
        
        print("\n[*] 3. Mengambil data normal lokal...")
        df_loc = pd.read_csv(jalur_lokal)
        df_normal_raw = df_loc[df_loc['label'] == 'normal'].copy()
        df_normal_final = pd.DataFrame({'text': df_normal_raw['Teks'], 'label': 0})
        
        print("[*] 4. Menyatukan data (Data Fusion)...")
        df_gabungan = pd.concat([df_fraud_final, df_normal_final], ignore_index=True)
        
        print("[*] 5. Mencuci dan menormalisasi teks (NLP)...")
        df_gabungan['clean_text'] = df_gabungan['text'].apply(bersihkan_dan_normalisasi)
        df_gabungan = df_gabungan[df_gabungan['clean_text'] != '']
        
        os.makedirs(os.path.dirname(jalur_output), exist_ok=True)
        df_gabungan[['clean_text', 'label']].to_csv(jalur_output, index=False)
        print(f"\n[SUCCESS] {df_gabungan.shape[0]} Data Bebas Bias Bahasa berhasil disimpan di: {jalur_output}")
        print("=== PIPA V2.0 SELESAI ===")

    except Exception as e:
        print(f"[!] ERROR SISTEM: {e}")

if __name__ == "__main__":
    jalankan_pipeline_multibahasa()