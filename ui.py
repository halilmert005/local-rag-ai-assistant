import os
import shutil
import gradio as gr
import ingestion
from retrieval import get_relevant_chunks
from main import send_message
from foundry_local_sdk import Configuration, FoundryLocalManager

# --- YAPAY ZEKA MODELLERİ (GLOBAL DEĞİŞKENLER) ---
chat_client = None
embed_client = None


def initialize_models():
    """Arayüz başlatılırken yapay zeka modellerini VRAM'e (GPU) yükler."""
    global chat_client, embed_client
    print("Modeller yükleniyor, lütfen bekleyin.")

    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    result = manager.download_and_register_eps()
    print(f"EP kayıt sonucu: {result.success}, {result.status}")

    # 1. SOHBET MODELİ (CHAT)
    print("\nSohbet modeli hazırlanıyor...")
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")

    # İsminin içinde 'cuda' geçen GPU varyantını bul
    chat_cuda_variant = next((v for v in chat_model.variants if "cuda" in v.id.lower()), None)
    if chat_cuda_variant:
        print(f"Sohbet modeli için GPU varyantı seçildi: '{chat_cuda_variant.id}'")
        chat_model.select_variant(chat_cuda_variant)
    else:
        print("Uyarı: Sohbet modeli için CUDA varyantı bulunamadı, CPU kullanılacak.")

    chat_model.download(lambda p: print(f"\rİndiriliyor (Sohbet): %{p:.0f}", end=""))
    print("\nSohbet modeli VRAM'e yükleniyor...")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    # 2. GÖMME MODELİ (EMBEDDING)
    print("\nGömme motoru hazırlanıyor...")
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")

    # İsminin içinde 'cuda' geçen GPU varyantını bul
    embed_cuda_variant = next((v for v in embed_model.variants if "cuda" in v.id.lower()), None)
    if embed_cuda_variant:
        print(f"Gömme modeli için GPU varyantı seçildi: '{embed_cuda_variant.id}'")
        embed_model.select_variant(embed_cuda_variant)
    else:
        print("Uyarı: Gömme modeli için CUDA varyantı bulunamadı, CPU kullanılacak.")

    embed_model.download(lambda p: print(f"\rİndiriliyor (Gömme): %{p:.0f}", end=""))
    print("\nGömme modeli VRAM'e yükleniyor...")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    print("\nTüm modeller GPU üzerinde hazır! Arayüzü kullanabilirsiniz.")


# --- BÖLÜM 1: ARKA PLAN FONKSİYONLARI ---

def process_document(file):
    if file is None:
        return "Lütfen dosyayı yükleyin."

    # 1. Yüklenen dosyayı kalıcı olarak saklayacağımız klasörü belirliyoruz
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # 2. Dosyanın orijinal ismini alıp yeni yolu oluşturuyoruz
    original_filename = os.path.basename(file.name)
    save_path = os.path.join(upload_dir, original_filename)

    # 3. Dosyayı Gradio'nun geçici klasöründen bizim uploads klasörümüze kopyalıyoruz
    shutil.copy(file.name, save_path)

    # 4. Ingestion betiğine yeni kalıcı dosya yolunu gönderiyoruz
    result_message = ingestion.process_file(save_path)
    return result_message


def query_rag(user_input, chat_history):
    if not user_input or not user_input.strip():
        return "", chat_history
    
    # 1. SQLite'ta benzerlik araması yap (Veritabanı yolunu data/chunking.db olarak güncelledik)
    db_path = os.path.join("data", "chunking.db")
    relevant_chunks = get_relevant_chunks(user_input, embed_client, db_path=db_path, top_k=2)

    # Parçaları birleştir
    context_text = "\n\n".join(relevant_chunks)

    # 2. LLM'e prompt gönder
    if not context_text.strip():
        bot_response = "Bu konu hakkında veritabanında yeterli bilgi bulamadım."
    else:
        bot_response = send_message(chat_client, user_input, context=context_text)

    # 3. Geçmişi sözlük formatında güncelle
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_response})

    return "", chat_history


# --- BÖLÜM 2: ARAYÜZ İSKELETİ ---

with gr.Blocks() as demo:
    gr.Markdown("# Local RAG AI Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Belge Yükleme ve İşleme")
            upload_file = gr.File(label="PDF veya Metin Belgesi Yükle")
            process_button = gr.Button("Veritabanına İşle", variant="primary")
            system_status = gr.Textbox(label="Sistem Durumu", interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("### Asistan ile Sohbet")
            chatbot = gr.Chatbot(label="RAG Assistant", height=500)
            user_message = gr.Textbox(label="Mesajınızı yazın",
                                      placeholder="Yüklenen belgelerle ilgili ne sormak istersiniz?")
            clear_button = gr.ClearButton([user_message, chatbot], value="Sohbeti Temizle")

    # --- BÖLÜM 3: TETİKLEYİCİLER ---
    process_button.click(fn=process_document, inputs=upload_file, outputs=system_status)
    user_message.submit(fn=query_rag, inputs=[user_message, chatbot], outputs=[user_message, chatbot])

if __name__ == "__main__":
    # Arayüz açılmadan önce modelleri RAM'e yükle
    initialize_models()
    demo.launch(theme=gr.themes.Soft())