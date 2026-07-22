import json
import os
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager


def setup_database():
    """Initializes the SQLite database and creates the necessary documents table."""
    conn = sqlite3.connect("veritabani.sqlite")
    cursor = conn.cursor()

    # Create table for storing text chunks and their embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS belgeler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metin TEXT,
            vektor TEXT
        )
    ''')
    conn.commit()
    return conn, cursor


def main():
    print("Veri Alımı (Ingestion) Sistemi Başlatılıyor...")

    # 1. Initialize the database
    conn, cursor = setup_database()

    # 2. Initialize SDK and Embedding Model
    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("\nGömme Modeli (qwen3-embedding-0.6b) hazırlanıyor...")
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download(lambda p: print(f"\rİndiriliyor: %{p:.0f}", end=""))
    embed_model.load()

    client = embed_model.get_embedding_client()

    # 3. Read and Chunk Markdown Files
    data_folder = "data"
    if not os.path.exists(data_folder):
        print(f"\nHATA: '{data_folder}' klasörü bulunamadı!")
        return

    # Filter only .md files from the directory
    md_files = [f for f in os.listdir(data_folder) if f.endswith('.md')]

    if not md_files:
        print(f"\nHATA: '{data_folder}' klasöründe .md dosyası bulunamadı!")
        return

    # Iterate through each file and process chunks
    for file_name in md_files:
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split document by double line breaks (paragraphs)
        chunks = content.split("\n\n")
        print(f"\n{file_name} okundu. {len(chunks)} parça işleniyor...")

        # 4. Generate embeddings and save to database
        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
                
            try:
                # Generate embedding vector for current text chunk
                response = client.generate_embeddings([chunk])
                vector_list = response.data[0].embedding

                # Serialize vector list to JSON string for SQLite storage
                vector_json = json.dumps(vector_list)

                # Insert processed chunk and vector into the database
                cursor.execute(
                    "INSERT INTO belgeler (metin, vektor) VALUES (?, ?)",
                    (chunk, vector_json)
                )
            except Exception as e:
                print(f"HATA: {file_name} - Parça {idx + 1} işlenirken hata oluştu: {e}")

    # Commit changes and clean up resources
    conn.commit()
    conn.close()
    embed_model.unload()

    print("\n✅ İşlem Tamamlandı! Tüm metinler vektörleştirilip 'veritabani.sqlite' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()