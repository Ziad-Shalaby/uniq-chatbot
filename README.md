# UniQ AI Chatbot 🎓
### المساعد الذكي · كلية الحاسبات وعلوم البيانات · جامعة الإسكندرية

Arabic RAG chatbot grounded in the official faculty bylaws (اللائحة الداخلية).

---

## Quick Start — Local Machine

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open `http://localhost:8501` — click **"تحميل النموذج"** in the sidebar.

---

## Options by Hardware

| Your Setup | Model | Settings |
|---|---|---|
| **NVIDIA GPU (≥8GB)** | Qwen2.5-7B-Instruct | 4-bit ON |
| **NVIDIA GPU (<8GB)** | Qwen2.5-3B-Instruct | 4-bit ON |
| **CPU only** | Qwen2.5-3B-Instruct | CPU toggle ON |
| **Google Colab T4** | Qwen2.5-7B-Instruct | 4-bit ON, use notebook |

> ⚠️ CPU mode is slow (~30–60s per response). Consider Colab for best speed.

---

## Google Colab

Open `UniQ_Colab.ipynb` in Colab:
1. Set Runtime → **T4 GPU**
2. Run all cells
3. Paste your free [ngrok token](https://ngrok.com)
4. Click the public URL that appears

---

## Files

```
uniq-streamlit/
├── app.py                  ← Streamlit UI (Arabic RTL chat interface)
├── rag_engine.py           ← RAG pipeline (embeddings + ChromaDB + Qwen)
├── knowledge_base/
│   ├── bylaws_structured.md            ← Structured bylaws text (Arabic)
│   └── لائحة_كلية_الحاسبات.pdf         ← Original PDF
├── vector_store/           ← Auto-created: ChromaDB index
├── UniQ_Colab.ipynb        ← Google Colab notebook
└── requirements.txt
```

---

## Adding More Documents

Drop any `.pdf`, `.txt`, or `.md` file into `knowledge_base/` before starting — it will be auto-ingested on first run.

You can also upload documents at runtime using the sidebar file uploader.

---

## How It Works

```
Student question (Arabic)
         ↓
multilingual-e5-large embeddings
         ↓
ChromaDB similarity search → top 5 bylaws chunks
         ↓
Qwen2.5-7B + chunks injected as context
         ↓
Grounded Arabic answer
```
