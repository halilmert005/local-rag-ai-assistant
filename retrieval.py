import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager


def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_relevant_chunks(query, embed_client=None, db_path="chunking.db", top_k=3):
    """
    Takes a user query, generates its embedding, and returns the top_k
    most similar text chunks from the SQLite database.
    """
    # 1. ADIM: Vektör modelini bul ve işleme başlamadan hemen önce yükle
    try:
        manager = FoundryLocalManager.instance
        embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        embed_model.load()  # ÇÖZÜM: Arka planda çökmüş olsa bile modeli zorla belleğe alır
        embed_client = embed_model.get_embedding_client()
    except Exception as e:
        print(f"Hata: Model yüklenirken bir sorun oluştu - {e}")
        return []

    # 2. ADIM: Soruyu vektörleştir
    try:
        response = embed_client.generate_embeddings([str(query)])
        query_vector = response.data[0].embedding
    except Exception as e:
        print(f"Hata: Soru vektörleştirilemedi - {e}")
        return []

    # 3. ADIM: Veritabanından belgeleri çek
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Veritabanı şeması ingestion.py tarafında belirlendiği için SQL sorgusu sabit bırakıldı.
        cursor.execute("SELECT metin, vektor FROM belgeler")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Hata: Veritabanına ulaşılamadı - {e}")
        return []

    # 4. ADIM: Kosinüs benzerliğini hesapla
    similarities = []
    for row in rows:
        chunk_text = row[0]
        chunk_vector = json.loads(row[1])

        score = cosine_similarity(query_vector, chunk_vector)
        similarities.append((score, chunk_text))

    # 5. ADIM: En yüksek skorluları sırala ve seç
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = [item[1] for item in similarities[:top_k]]

    # 6. ADIM: İşlem bitince RAM'de yer açmak için vektör modelini bellekten çıkar
    try:
        embed_model.unload()
    except Exception as e:
        pass  # Kapatırken oluşacak ufak hataları görmezden gel

    return top_results


if __name__ == "__main__":
    print("Geri getirme testi başlatılıyor.")

    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    client = embed_model.get_embedding_client()

    test_query = "Kripto paralarda riskler nelerdir?"
    print(f"\nSoru: '{test_query}'")
    print("En alakalı 2 metin aranıyor.")

    retrieved_chunks = get_relevant_chunks(test_query, client, top_k=2)

    for index, chunk in enumerate(retrieved_chunks):
        print(f"--- Sonuç {index + 1} ---")
        print(chunk)
        print("-" * 20)

    embed_model.unload()