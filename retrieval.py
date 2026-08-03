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


def get_relevant_chunks(query, embed_client, db_path="chunking.db", top_k=2):
    """embed_client dışarıdan hazır olarak gelir, burada model yükleme/kapatma YAPILMAZ."""
    try:
        response = embed_client.generate_embeddings([str(query)])
        query_vector = response.data[0].embedding
    except Exception as e:
        print(f"Hata: Soru vektörleştirilemedi - {e}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT metin, vektor FROM belgeler")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Hata: Veritabanına ulaşılamadı - {e}")
        return []

    similarities = []
    for row in rows:
        chunk_text = row[0]
        chunk_vector = json.loads(row[1])
        score = cosine_similarity(query_vector, chunk_vector)
        similarities.append((score, chunk_text))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in similarities[:top_k]]


if __name__ == "__main__":
    print("Geri getirme testi başlatılıyor.")

    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    client = embed_model.get_embedding_client()

    test_query = "Politika faizi nedir?"
    print(f"\nSoru: '{test_query}'")
    print("En alakalı 2 metin aranıyor.")

    retrieved_chunks = get_relevant_chunks(test_query, client, top_k=2)

    for index, chunk in enumerate(retrieved_chunks):
        print(f"--- Sonuç {index + 1} ---")
        print(chunk)
        print("-" * 20)

    embed_model.unload()