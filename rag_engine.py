"""
UniQ RAG Engine
================
Self-contained RAG pipeline.
Runs on:
  - Local CPU  (slow but works on any machine, use Qwen2.5-3B)
  - Local GPU  (fast, use Qwen2.5-7B)
  - Google Colab T4 (use 4-bit + Qwen2.5-7B)
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("uniq.rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KB_DIR = Path(__file__).parent / "knowledge_base"

# ── Arabic system prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """أنت "يونيك بوت" — المساعد الذكي الرسمي لنظام UniQ في كلية الحاسبات وعلوم البيانات، جامعة الإسكندرية.

مهمتك:
- الإجابة بدقة واحترافية على أسئلة الطلاب حول اللائحة الأكاديمية والإجراءات
- الاستناد فقط إلى المعلومات الموجودة في السياق المتاح أدناه
- الرد باللغة العربية دائماً (فصحى أو عامية حسب سؤال الطالب)
- تقديم إجابات واضحة ومنظمة

قواعد مهمة:
1. إذا لم تجد المعلومة في السياق أدناه، قل صراحةً: "لا تتوفر لديّ معلومات كافية حول هذا الموضوع. يُرجى التواصل مع شؤون الطلاب مباشرة."
2. لا تخترع أرقاماً أو نصوصاً غير موجودة في السياق
3. إذا كان السؤال عن خطوات إجراء، رتبها بشكل واضح
4. كن مختصراً ومفيداً

السياق من اللائحة الداخلية:
{context}
"""

FALLBACK_AR = "عذراً، لم أجد معلومات كافية في اللائحة للإجابة على سؤالك بدقة. يُرجى التواصل مع مكتب شؤون الطلاب مباشرةً."

INTENT_MAP = {
    "withdrawal":    ["سحب القيد", "انسحاب", "withdrawal", "w تقدير"],
    "add_drop":      ["إضافة", "حذف", "add drop", "تسجيل مادة"],
    "gpa":           ["معدل", "gpa", "cgpa", "درجات", "حساب المعدل"],
    "graduation":    ["تخرج", "graduation", "ساعة معتمدة", "شرف", "honors"],
    "probation":     ["متعثر", "مراقبة", "probation", "فصل", "إنذار"],
    "attendance":    ["غياب", "مواظبة", "حضور", "attendance", "حرمان"],
    "registration":  ["تسجيل", "registration", "مقررات", "فصل دراسي"],
    "payment":       ["مصاريف", "رسوم", "دفع", "payment"],
    "programs":      ["برنامج", "تخصص", "program", "قسم", "برامج"],
    "courses":       ["مقرر", "مادة", "course", "ساعات", "كود"],
}

SUGGESTED_ACTIONS = {
    "withdrawal":   ["تقديم طلب انسحاب", "التحدث مع المرشد الأكاديمي"],
    "add_drop":     ["فتح نموذج الحذف والإضافة", "مراجعة العبء الدراسي"],
    "gpa":          ["حساب المعدل", "مراجعة كشف الدرجات"],
    "graduation":   ["مراجعة متطلبات التخرج", "حساب الساعات المتبقية"],
    "registration": ["الانتقال لصفحة التسجيل", "التواصل مع المرشد الأكاديمي"],
}


class TextChunker:
    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

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
                current = para if len(para) <= self.chunk_size else para[:self.chunk_size]
        if current:
            chunks.append(current)
        return chunks


class RAGEngine:
    """Complete RAG pipeline: load models → embed docs → query."""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", use_4bit=True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_cpu = os.environ.get("USE_CPU", "0") == "1"
        self.embedder = None
        self.vector_store = None
        self.llm = None
        self.tokenizer = None
        self._ready = False

    def initialize(self):
        logger.info(f"Initializing RAG engine (model={self.model_name}, 4bit={self.use_4bit}, cpu={self.use_cpu})")
        self._load_embedder()
        self._load_vector_store()
        self._ingest_knowledge_base()
        self._load_llm()
        self._ready = True
        logger.info("✅ RAG engine ready")

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        logger.info("✅ Embedder loaded")

    def _load_vector_store(self):
        import chromadb
        from chromadb.config import Settings as CS
        logger.info("Initializing ChromaDB...")
        self._chroma = chromadb.PersistentClient(
            path=str(Path(__file__).parent / "vector_store"),
            settings=CS(anonymized_telemetry=False),
        )
        self.vector_store = self._chroma.get_or_create_collection(
            name="uniq_bylaws",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"✅ Vector store ready ({self.vector_store.count()} chunks)")

    def _ingest_knowledge_base(self):
        """Ingest all documents from knowledge_base/ if not already indexed."""
        if self.vector_store.count() > 0:
            logger.info(f"Knowledge base already has {self.vector_store.count()} chunks — skipping ingestion")
            return

        logger.info("Ingesting knowledge base documents...")
        total = 0
        for fpath in KB_DIR.iterdir():
            if fpath.suffix.lower() in {".pdf", ".txt", ".md"}:
                try:
                    n = self.ingest_file(str(fpath))
                    total += n
                    logger.info(f"  ✅ {fpath.name}: {n} chunks")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to ingest {fpath.name}: {e}")

        logger.info(f"✅ Total chunks ingested: {total}")

    def _load_llm(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading LLM: {self.model_name} (cpu={self.use_cpu})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        if self.use_cpu:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
        elif self.use_4bit and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        self.llm.eval()
        logger.info("✅ LLM loaded")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, history: List[Dict] = None) -> Dict[str, Any]:
        intent = self._detect_intent(question)
        sources = self._retrieve(question)
        fallback = not sources or sources[0]["score"] < 0.30

        if fallback:
            return {
                "answer": FALLBACK_AR,
                "sources": [], "intent": intent,
                "suggested_actions": SUGGESTED_ACTIONS.get(intent, []),
                "fallback": True,
            }

        context = self._build_context(sources)
        answer = self._generate(question, context, history or [])

        return {
            "answer": answer,
            "sources": [
                {
                    "title": s.get("title", "اللائحة"),
                    "content_preview": s["content"][:150],
                    "score": round(s["score"], 3),
                    "source": s.get("source", ""),
                }
                for s in sources
            ],
            "intent": intent,
            "suggested_actions": SUGGESTED_ACTIONS.get(intent, []),
            "fallback": False,
        }

    def _retrieve(self, question: str) -> List[Dict]:
        emb = self.embedder.encode(
            f"query: {question}", normalize_embeddings=True
        ).tolist()

        count = self.vector_store.count()
        if count == 0:
            return []

        results = self.vector_store.query(
            query_embeddings=[emb],
            n_results=min(5, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist
            if score >= 0.25:
                chunks.append({
                    "content": doc,
                    "title": meta.get("title", "اللائحة"),
                    "source": meta.get("source", ""),
                    "score": score,
                })
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks

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
        import torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        for turn in history[-10:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.llm.device)

        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> int:
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name

        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext in {".txt", ".md"}:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported: {ext}")

        title = Path(file_path).stem
        chunker = TextChunker(chunk_size=700, overlap=80)
        chunks = chunker.split(text, {"source": name, "title": title})
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
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t.strip())
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

    def _detect_intent(self, question: str) -> str:
        q = question.lower()
        for intent, keywords in INTENT_MAP.items():
            if any(kw.lower() in q for kw in keywords):
                return intent
        return "general"

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.vector_store.count() if self.vector_store else 0,
            "embedding_model": "intfloat/multilingual-e5-large",
            "llm_model": self.model_name,
            "use_4bit": self.use_4bit,
            "use_cpu": self.use_cpu,
        }

    def reset_knowledge_base(self):
        self._chroma.delete_collection("uniq_bylaws")
        self.vector_store = self._chroma.get_or_create_collection(
            name="uniq_bylaws", metadata={"hnsw:space": "cosine"}
        )
