# Local RAG AI Assistant

Bu proje, kendi bilgisayarınızda internete ihtiyaç duymadan belgeleriniz (PDF vb.) üzerinde soru-cevap yapabileceğiniz bir RAG (Retrieval-Augmented Generation) asistanıdır. 

Projeyi baştan sona Qwen 1.5B / 2.5-1.5B gibi küçük parametreli modeller (SLM) ile en verimli şekilde çalışacak formata (Stable v1.0) getirdim.

## Neler Var?

* **Gradio Arayüzü:** Sistemi terminalden kurtarıp web arayüzüne taşıdım. Belge yükleme (ingestion) ve sohbet işlemleri artık doğrudan bu arayüz üzerinden yapılıyor.
* **XML Prompting ile Halüsinasyon Kontrolü:** Küçük modeller bağlamdan çok çabuk koptuğu veya sistem promptunu sızdırdığı için `<kurallar>` ve `<baglam>` şeklinde katı bir XML etiketleme yapısı kurdum. Bu sayede model kafasına göre uydurmak yerine sadece verdiğimiz metne sadık kalıyor.
* **Altyapı Detayları:**
  * Vektör depolama ve arama işlemleri için **SQLite** kullanılıyor.
  * PDF'lerden düzgün metin çıkarabilmek için **pdfplumber** entegre edildi.
  * Modelin belleğe yüklenmesi ve yaşam döngüsü **FoundryLocalManager** üzerinden yönetiliyor.

## Kurulum

Projeyi klonladıktan sonra proje dizininde şu adımları izleyebilirsiniz:

1. Sanal ortamı aktif edin:
   ```bash
   .venv\Scripts\Activate