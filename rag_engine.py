"""
UniQ RAG Engine
================
Uses:
  - sentence-transformers (multilingual-e5-small) for embeddings (~120MB)
  - ChromaDB for vector search
  - Groq API for generation (FREE tier available at console.groq.com)
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("uniq.rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KB_DIR = Path(__file__).parent / "knowledge_base"

SYSTEM_PROMPT = """أنت "يونيك بوت" — المساعد الذكي الرسمي لنظام UniQ في كلية الحاسبات وعلوم البيانات، جامعة الإسكندرية.

مهمتك:
- الإجابة بدقة واحترافية على أسئلة الطلاب حول اللائحة الأكاديمية والإجراءات
- الاستناد فقط إلى المعلومات الموجودة في السياق المتاح أدناه
- الرد باللغة العربية دائماً (فصحى أو عامية حسب سؤال الطالب)
- تقديم إجابات واضحة ومنظمة

قواعد مهمة:
1. إذا لم تجد المعلومة في السياق، قل: "لا تتوفر لديّ معلومات كافية. يُرجى التواصل مع شؤون الطلاب مباشرة."
2. لا تخترع أرقاماً أو نصوصاً غير موجودة في السياق
3. إذا كان السؤال عن خطوات إجراء، رتبها بشكل واضح
4. كن مختصراً ومفيداً

السياق من اللائحة الداخلية:
{context}
"""

FALLBACK_AR = "عذراً، لم أجد معلومات كافية في اللائحة للإجابة على سؤالك. يُرجى التواصل مع مكتب شؤون الطلاب مباشرةً."

INTENT_MAP = {
    "withdrawal":   ["سحب القيد", "انسحاب", "withdrawal"],
    "add_drop":     ["إضافة", "حذف", "add drop", "تسجيل مادة"],
    "gpa":          ["معدل", "gpa", "cgpa", "درجات", "حساب المعدل"],
    "graduation":   ["تخرج", "graduation", "ساعة معتمدة", "شرف", "honors"],
    "probation":    ["متعثر", "مراقبة", "probation", "فصل", "إنذار"],
    "attendance":   ["غياب", "مواظبة", "حضور", "attendance", "حرمان"],
    "registration": ["تسجيل", "registration", "مقررات", "فصل دراسي"],
    "programs":     ["برنامج", "تخصص", "program", "قسم"],
    "courses":      ["مقرر", "مادة", "course", "ساعات"],
}

SUGGESTED_ACTIONS = {
    "withdrawal":   ["التحدث مع المرشد الأكاديمي"],
    "add_drop":     ["مراجعة العبء الدراسي"],
    "gpa":          ["مراجعة كشف الدرجات"],
    "graduation":   ["مراجعة متطلبات التخرج"],
    "registration": ["التواصل مع المرشد الأكاديمي"],
}

# Groq free models (pick best for Arabic)
GROQ_MODEL = "llama-3.1-8b-instant"   # free, fast, good Arabic support


class TextChunker:
    def __init__(self, chunk_size=800):
        self.chunk_size = chunk_size

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        raw_chunks = self._merge(paragraphs)
        result = []
        source = metadata.get("source", "unknown")
        for i, chunk in enumerate(raw_chunks):
            cid = hashlib.md5(f"{source}::{i}::{chunk[:40]}".encode()).hexdigest()
            result.append({
                "id": cid,
                "content": chunk,
                "metadata": {**metadata, "chunk_index": i, "chunk_id": cid},
            })
        return result

    def _merge(self, paragraphs):
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                current = para[:self.chunk_size]
        if current:
            chunks.append(current)
        return chunks


class RAGEngine:
    """Lightweight RAG: small embedding model + Groq free API for generation."""

    def __init__(self, model_name=GROQ_MODEL, use_4bit=False):
        self.groq_model = GROQ_MODEL
        self.embedder = None
        self.vector_store = None
        self._chroma = None
        self._ready = False
        self._api_key = os.environ.get("GROQ_API_KEY", "")

    def initialize(self):
        logger.info("Initializing RAG engine (Groq free API)")
        self._load_embedder()
        self._load_vector_store()
        self._ingest_knowledge_base()
        self._ready = True
        logger.info("RAG engine ready")

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (multilingual-e5-small ~120MB)...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        logger.info("Embedder loaded")

    def _load_vector_store(self):
        import chromadb
        from chromadb.config import Settings as CS
        logger.info("Initializing ChromaDB...")
        self._chroma = chromadb.PersistentClient(
            path=str(Path(__file__).parent / "vector_store"),
            settings=CS(anonymized_telemetry=False),
        )
        self.vector_store = self._chroma.get_or_create_collection(
            name="uniq_bylaws_v2",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store ready ({self.vector_store.count()} chunks)")

    def _ingest_knowledge_base(self):
        if self.vector_store.count() > 0:
            logger.info(f"Already indexed {self.vector_store.count()} chunks — skipping")
            return
        logger.info("Ingesting knowledge base...")
        total = 0
        for fpath in sorted(KB_DIR.iterdir()):
            if fpath.suffix.lower() in {".pdf", ".txt", ".md"}:
                try:
                    n = self.ingest_file(str(fpath))
                    total += n
                    logger.info(f"  {fpath.name}: {n} chunks")
                except Exception as e:
                    logger.warning(f"  Failed {fpath.name}: {e}")
        logger.info(f"Total chunks indexed: {total}")

    def query(self, question: str, history: List[Dict] = None) -> Dict[str, Any]:
        intent = self._detect_intent(question)
        sources = self._retrieve(question)
        fallback = not sources or sources[0]["score"] < 0.30

        if fallback:
            return {"answer": FALLBACK_AR, "sources": [], "intent": intent,
                    "suggested_actions": SUGGESTED_ACTIONS.get(intent, []), "fallback": True}

        context = self._build_context(sources)
        answer = self._generate(question, context, history or [])

        return {
            "answer": answer,
            "sources": [{"title": s.get("title", "اللائحة"),
                         "content_preview": s["content"][:150],
                         "score": round(s["score"], 3),
                         "source": s.get("source", "")} for s in sources],
            "intent": intent,
            "suggested_actions": SUGGESTED_ACTIONS.get(intent, []),
            "fallback": False,
        }

    def _retrieve(self, question: str) -> List[Dict]:
        emb = self.embedder.encode(f"query: {question}", normalize_embeddings=True).tolist()
        count = self.vector_store.count()
        if count == 0:
            return []
        results = self.vector_store.query(
            query_embeddings=[emb], n_results=min(5, count),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = 1.0 - dist
            if score >= 0.25:
                chunks.append({"content": doc, "title": meta.get("title", "اللائحة"),
                                "source": meta.get("source", ""), "score": score})
        return sorted(chunks, key=lambda x: x["score"], reverse=True)

    def _build_context(self, sources: List[Dict]) -> str:
        parts, total = [], 0
        for i, s in enumerate(sources, 1):
            chunk = f"[مصدر {i}: {s['title']}]\n{s['content']}"
            if total + len(chunk) > 5000:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str, history: List[Dict]) -> str:
        if not self._api_key:
            return (
                "⚠️ يرجى إضافة GROQ_API_KEY في Secrets الخاصة بالتطبيق.\n\n"
                "احصل على مفتاح مجاني من: https://console.groq.com"
            )

        from groq import Groq
        client = Groq(api_key=self._api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        for turn in history[-6:]:
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def ingest_file(self, file_path: str) -> int:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext in {".txt", ".md"}:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported: {ext}")

        chunker = TextChunker(chunk_size=700)
        chunks = chunker.split(text, {"source": Path(file_path).name, "title": Path(file_path).stem})
        if not chunks:
            return 0

        texts = [f"passage: {c['content']}" for c in chunks]
        embeddings = self.embedder.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        self.vector_store.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def _extract_pdf(self, path: str) -> str:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t.strip())
        return "\n\n".join(pages)

    def _detect_intent(self, question: str) -> str:
        q = question.lower()
        for intent, keywords in INTENT_MAP.items():
            if any(kw.lower() in q for kw in keywords):
                return intent
        return "general"

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.vector_store.count() if self.vector_store else 0,
            "embedding_model": "intfloat/multilingual-e5-small",
            "llm_model": f"Groq / {self.groq_model}",
            "use_4bit": False,
            "use_cpu": True,
        }

    def reset_knowledge_base(self):
        self._chroma.delete_collection("uniq_bylaws_v2")
        self.vector_store = self._chroma.get_or_create_collection(
            name="uniq_bylaws_v2", metadata={"hnsw:space": "cosine"})
