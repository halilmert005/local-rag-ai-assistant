import gradio as gr
import ingestion
from retrieval import get_relevant_chunks
from main import send_message
from foundry_local_sdk import Configuration, FoundryLocalManager

# --- YKAPAY ZEKA MODELLERİ (GLOBAL DEĞİŞKENLER) ---
chat_client = None
embed_client = None


def initialize_models():
    """Arayüz başlatılırken yapay zeka modellerini RAM'e yükler."""
    global chat_client, embed_client
    print("Modeller yükleniyor, lütfen bekleyin.")

    config = Configuration(app_name="rag-asistan")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Sohbet modeli
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    # Vektör/Arama modeli
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()
    print("Modeller hazır! Arayüzü kullanabilirsiniz.")


# --- BÖLÜM 1: ARKA PLAN FONKSİYONLARI ---

def process_document(file):
    if file is None:
        return "Lütfen dosyayı yükleyin."
    file_path = file.name
    result_message = ingestion.process_file(file_path)
    return result_message


def query_rag(user_input, chat_history):
    # 1. SQLite'ta benzerlik araması yap (chunking.db üzerinden)
    relevant_chunks = get_relevant_chunks(user_input, embed_client, db_path="chunking.db", top_k=3)

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
    gr.Markdown("# 🧠 Local RAG AI Assistant")

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