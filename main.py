import sys
import os
from foundry_local_sdk import Configuration, FoundryLocalManager

# Import our custom retrieval logic
from retrieval import get_relevant_chunks


def send_message(client, user_message, context=""):
    system_rules = f"""Sen bir RAG (Retrieval-Augmented Generation) asistanısın.
Görevlerini sadece sana verilen <baglam> etiketleri içindeki metinlere dayanarak yerine getirmelisin.

<kurallar>
- <baglam> içinde sana birden fazla bilgi verilmiş olabilir. SADECE kullanıcının sorusuyla doğrudan ilgili olan bilgiyi seç.
- Soruyu cevaplamak için gereksiz olan diğer bilgileri (örneğin sorulmayan diğer faiz türlerini) kesinlikle cevaba ekleme, filtrele.
- Akıcı, kurallı ve dilbilgisi hatası olmayan, doğal bir Türkçe kullan ("Política" gibi yabancı veya uydurma kelimeler kullanma).
- Eğer sorunun cevabı <baglam> içinde yoksa, sadece "Bu bilgiye sahip değilim." yaz.
</kurallar>

<baglam>
{context}
</baglam>
"""

    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": user_message}
    ]

    response = client.complete_chat(messages)
    return response.choices[0].message.content


def main():
    print("Sistem başlatılıyor.")

    # 1. SDK Configuration and Manager Initialization
    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2. Select, Download, and Load Models
    # We need both the chat model for answers and the embedding model for search
    print("\nYapay zeka sohbet motoru (qwen2.5-1.5b) hazırlanıyor.")
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download(lambda p: print(f"\rİndiriliyor (Sohbet): %{p:.0f}", end=""))
    print("\nSohbet modeli belleğe (RAM) yükleniyor.")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nGömme motoru (qwen3-embedding-0.6b) hazırlanıyor.")
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download(lambda p: print(f"\rİndiriliyor (Gömme): %{p:.0f}", end=""))
    print("\nGömme modeli belleğe (RAM) yükleniyor.")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    print("\nYapay zeka hazır. Finans, kripto veya diğer konulardaki sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'q' veya 'çıkış' yazın.\n")

    # 3. Interactive CLI Loop (Replacing the hardcoded test)
    while True:
        user_query = input("Soru: ")

        if user_query.lower() in ['q', 'quit', 'çıkış']:
            break

        if not user_query.strip():
            continue

        print("Veri tabanında aranıyor ve cevap üretiliyor.\n")

        # Retrieve relevant context from SQLite using embedding search
        # DÜZELTME: Veritabanı yolunu yeni klasör yapısına uyumlu hale getirdik
        db_path = os.path.join("data", "chunking.db")
        relevant_chunks = get_relevant_chunks(user_query, embed_client, db_path=db_path, top_k=3)

        # Combine retrieved chunks into a single text block
        context_text = "\n\n".join(relevant_chunks)

        # Generate the final answer using the retrieved context
        answer = send_message(chat_client, user_query, context=context_text)

        print("-----Modelin Cevabı-----")
        print(answer)
        print("----------------------\n")

    # 4. Resource Cleanup
    print("\nİşlem tamamlandı, sistem kaynakları serbest bırakılıyor.")
    chat_model.unload()
    embed_model.unload()


if __name__ == "__main__":
    main()