import pandas as pd

def jalankan_eda():
    print("=== MENGHIDUPKAN MESIN ANALISIS (EDA) ===")
    
    try:
        file_path = 'data/raw/final_dataset_output.csv'
        df = pd.read_csv(file_path)
        
        print(f"[*] Berhasil memuat {df.shape[0]} baris dan {df.shape[1]} kolom data.")
        
        kolom_penting = ['text', 'scam_type', 'lure_principles', 'language']
        
        df_bersih = df[kolom_penting].dropna(subset=['text', 'scam_type'])
        
        print(f"[*] Sisa data bersih setelah disaring: {df_bersih.shape[0]} baris.\n")
        
        print("=== TOP 5 MODUS PENIPUAN (SCAM TYPE) ===")
        print(df_bersih['scam_type'].value_counts().head(5))
        print("-" * 40)
        
        print("\n=== TOP 5 TEKNIK MANIPULASI PSIKOLOGIS ===")
        print(df_bersih['lure_principles'].value_counts().head(5))
        print("-" * 40)
        
        print("\n=== DISTRIBUSI BAHASA ===")
        print(df_bersih['language'].value_counts().head(5))
        print("-" * 40)
        
    except FileNotFoundError:
        print("[!] ERROR: File final_dataset_output.csv tidak ditemukan di folder data/raw/")
    except Exception as e:
        print(f"[!] ERROR SISTEM: {e}")

if __name__ == "__main__":
    jalankan_eda()