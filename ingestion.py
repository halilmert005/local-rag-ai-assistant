import json
import os
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

# Veritabanının oluşturulacağı yeni yol: data/chunking.db
DB_PATH = os.path.join("data", "chunking.db")


def setup_database():
    """Initializes the SQLite database and creates the necessary documents table."""
    # Data klasörü yoksa oluştur (güvenlik için)
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table for storing text chunks and their embeddings
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS belgeler
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       metin TEXT,
                       vektor TEXT
                   )
                   ''')
    conn.commit()
    return conn, cursor


# Fonksiyon ismini main'den process_file'a çevirdik ve parametre ekledik
def process_file(file_path):
    """Reads a single file, chunks it, generates embeddings, and stores in SQLite."""
    print(f"Data ingestion starting for: {file_path}")

    # 1. Initialize the database
    conn, cursor = setup_database()

    # 2. Initialize SDK and Embedding Model
    try:
        config = Configuration(app_name="rag-asistan")
        FoundryLocalManager.initialize(config)
    except Exception:
        # Eğer sistem ui.py tarafından zaten başlatılmışsa, hata verme ve sessizce atla
        pass

    manager = FoundryLocalManager.instance

    print("\nPreparing Embedding Model (qwen3-embedding-0.6b)...")
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    # ÇÖZÜM 1: Model arka planda çökmüş olsa bile işlemden hemen önce yeniden yüklenmesini garanti eder
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    # Dosya uzantısını kontrol et
    file_extension = os.path.splitext(file_path)[1].lower()
    content = ""

    # Tablo destekli PDF okuma mantığı
    if file_extension == '.pdf':
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # 1. Sayfadaki standart metni al
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n\n"

                    # 2. Tabloları bul ve Yapay Zekanın anlayacağı Markdown formatına (sütunlu) çevir
                    tables = page.extract_tables()
                    for table in tables:
                        if table:  # Eğer sayfada tablo varsa
                            for row_idx, row in enumerate(table):
                                # Hücrelerdeki satır atlamalarını temizle ve yan yana '|' ile diz
                                clean_row = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
                                content += "| " + " | ".join(clean_row) + " |\n"

                                # Tablo başlıklarının altına çizgi ekle (Yapay zeka bu çizgilerden tablo olduğunu anlar)
                                if row_idx == 0:
                                    content += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                            content += "\n\n"
        except ImportError:
            return "Sistem Hatası: PDF yükleyebilmek için pdfplumber gerekli. Terminale 'pip install pdfplumber' yazarak kurun."
    else:
        # Metin (.txt, .md vb.) dosyaları için okuma mantığı
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp1254', errors='ignore') as f:
                content = f.read()

    # Elde edilen metni (content) belirli uzunluktaki parçalara bölen fonksiyon
    def get_chunks(text, chunk_size=1000, overlap=100):
        chunks_list = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks_list.append(text[i:i + chunk_size])
        return chunks_list

    # Karakter sayısına göre bölüyoruz
    chunks = get_chunks(content)
    print(f"\nFile read. Processing {len(chunks)} chunks.")

    # 4. Generate embeddings and save to database
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        try:
            # Generate embedding vector for current text chunk
            response = embed_client.generate_embeddings([chunk])
            vector_list = response.data[0].embedding

            # Serialize vector list to JSON string for SQLite storage
            vector_json = json.dumps(vector_list)

            # Insert processed chunk and vector into the database
            cursor.execute(
                "INSERT INTO belgeler (metin, vektor) VALUES (?, ?)",
                (chunk, vector_json)
            )
        except Exception as e:
            print(f"ERROR: Failed to process chunk {idx + 1}: {e}")

    # Commit changes and clean up resources
    conn.commit()
    conn.close()
    embed_model.unload()

    # UI tarafında göstermek için bir mesaj döndürüyoruz
    file_name = file_path.split("\\")[-1].split("/")[-1]  # Çapraz platform uyumlu dosya adı alma
    return f"Success: '{file_name}' vectorized and saved to chunking.db"


if __name__ == "__main__":
    print("Toplu vektörizasyon işlemi başlatılıyor...")

    # Hedef klasörlerimizi belirliyoruz
    hedef_klasorler = [
        os.path.join("data", "sample_docs"),
        os.path.join("data", "uploads")
    ]

    islenen_dosya_sayisi = 0

    for klasor in hedef_klasorler:
        # Eğer klasör yoksa oluştur (Örn: projeyi yeni klonladıysan uploads olmayabilir)
        if not os.path.exists(klasor):
            os.makedirs(klasor, exist_ok=True)
            print(f"Bilgi: '{klasor}' klasörü oluşturuldu.")
            continue  # İçi boş olduğu için diğer klasöre geç

        # Klasördeki dosyaları tara
        for dosya_adi in os.listdir(klasor):
            dosya_yolu = os.path.join(klasor, dosya_adi)

            # Sadece dosya olanları ve ilgili uzantıları işle
            if os.path.isfile(dosya_yolu) and dosya_yolu.lower().endswith(('.pdf', '.txt', '.md')):
                sonuc = process_file(dosya_yolu)
                print(sonuc)
                islenen_dosya_sayisi += 1

    if islenen_dosya_sayisi == 0:
        print("\nİşlem bitti: Belirtilen klasörlerde işlenecek uygun belge bulunamadı.")
    else:
        print(f"\nİşlem bitti: Toplam {islenen_dosya_sayisi} belge veritabanına eklendi.")