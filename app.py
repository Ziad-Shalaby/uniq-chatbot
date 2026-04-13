"""
UniQ AI Chatbot — Streamlit App
=================================
Local Arabic RAG chatbot for كلية الحاسبات وعلوم البيانات - Alexandria University
Run: streamlit run app.py
"""

import streamlit as st
import time
import os
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniQ | المساعد الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Arabic RTL CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    * { font-family: 'Cairo', sans-serif !important; }

    /* RTL layout */
    .stChatMessage, .stChatInput, .element-container {
        direction: rtl;
        text-align: right;
    }

    /* Main header */
    .uniq-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(13,71,161,0.3);
    }
    .uniq-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .uniq-header p  { font-size: 0.95rem; margin: 6px 0 0; opacity: 0.85; }

    /* Chat bubbles */
    .user-bubble {
        background: #e3f2fd;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        direction: rtl;
        border-right: 4px solid #1565c0;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-bubble {
        background: #f8f9fa;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        direction: rtl;
        border-right: 4px solid #43a047;
        max-width: 88%;
    }
    .bot-bubble-fallback {
        background: #fff8e1;
        border-right: 4px solid #ff8f00;
    }

    /* Source badge */
    .source-badge {
        background: #e8eaf6;
        color: #3949ab;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.72rem;
        margin: 3px 2px;
        display: inline-block;
    }

    /* Suggested actions */
    .action-btn {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.82rem;
        color: #1565c0;
        cursor: pointer;
        margin: 3px;
        display: inline-block;
    }

    /* Sidebar */
    .sidebar-card {
        background: #f0f4f8;
        border-radius: 12px;
        padding: 12px 15px;
        margin: 8px 0;
        direction: rtl;
    }
    .sidebar-card h4 { color: #1a237e; margin: 0 0 6px; font-size: 0.9rem; }
    .sidebar-card p  { color: #546e7a; margin: 0; font-size: 0.82rem; }

    /* Status indicator */
    .status-ok   { color: #43a047; font-weight: 600; }
    .status-warn { color: #ff8f00; font-weight: 600; }
    .status-err  { color: #e53935; font-weight: 600; }

    /* Typing indicator */
    .typing-dots::after {
        content: '...';
        animation: dots 1.2s steps(4, end) infinite;
    }
    @keyframes dots {
        0%, 20%  { content: ''; }
        40%      { content: '.'; }
        60%      { content: '..'; }
        80%, 100%{ content: '...'; }
    }
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = {}


# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str, use_4bit: bool):
    """Load the RAG pipeline once and cache it."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_engine import RAGEngine
    engine = RAGEngine(model_name=model_name, use_4bit=use_4bit)
    engine.initialize()
    return engine


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")

    st.markdown("### 🤖 النموذج")
    st.info("يعمل البوت بواسطة **Groq API** (مجاني) — لا يحتاج GPU.", icon="✅")

    st.divider()

    # Load model button
    if not st.session_state.pipeline_ready:
        if st.button("🚀 تحميل النموذج وبدء الاستخدام", type="primary", use_container_width=True):
            with st.spinner("جاري تحميل النموذج... قد يستغرق ذلك بضع دقائق في أول مرة"):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent))
                    os.environ["USE_CPU"] = "1" if use_cpu else "0"

                    engine = load_pipeline("claude-haiku-4-5-20251001", False)
                    st.session_state.pipeline = engine
                    st.session_state.pipeline_ready = True

                    stats = engine.get_stats()
                    st.session_state.kb_stats = stats
                    st.success("✅ النموذج جاهز!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ خطأ في تحميل النموذج:\n{e}")
    else:
        st.markdown('<p class="status-ok">● النموذج جاهز ✓</p>', unsafe_allow_html=True)
        if st.button("🔄 إعادة تحميل النموذج", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.pipeline_ready = False
            st.session_state.pipeline = None
            st.rerun()

    st.divider()

    # Knowledge base stats
    st.markdown("### 📚 قاعدة المعرفة")
    if st.session_state.kb_stats:
        stats = st.session_state.kb_stats
        st.markdown(f"""
        <div class="sidebar-card">
            <h4>📄 اللائحة الداخلية</h4>
            <p>عدد الأجزاء المفهرسة: <b>{stats.get('total_chunks', 0)}</b></p>
            <p>نموذج التضمين: <b>{stats.get('embedding_model', '').split('/')[-1]}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # Add document
    st.markdown("#### إضافة وثائق جديدة")
    uploaded = st.file_uploader(
        "ارفع ملف PDF أو TXT",
        type=["pdf", "txt", "md"],
        help="أضف مستندات إلى قاعدة المعرفة"
    )
    if uploaded and st.session_state.pipeline_ready:
        if st.button("📥 إضافة إلى قاعدة المعرفة", use_container_width=True):
            save_path = f"/tmp/{uploaded.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("جاري إضافة المستند..."):
                n = st.session_state.pipeline.ingest_file(save_path)
                st.success(f"✅ تمت إضافة {n} جزء من الوثيقة")
                st.session_state.kb_stats = st.session_state.pipeline.get_stats()
                st.rerun()

    st.divider()

    # Clear chat
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Quick questions
    st.markdown("### 💡 أسئلة سريعة")
    quick_questions = [
        "كم ساعة أحتاج للتخرج؟",
        "ما هي البرامج الدراسية؟",
        "ما الفرق بين الحذف والانسحاب؟",
        "متى أُعتبر طالباً متعثراً؟",
        "شروط مرتبة الشرف؟",
        "كيف أحسب المعدل التراكمي؟",
    ]
    for q in quick_questions:
        if st.button(q, key=f"q_{q}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state._pending_question = q
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="uniq-header">
    <h1>🎓 UniQ — المساعد الذكي</h1>
    <p>كلية الحاسبات وعلوم البيانات · جامعة الإسكندرية<br>
    اسألني عن اللائحة الأكاديمية، المقررات، المعدلات، والإجراءات</p>
</div>
""", unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div class="bot-bubble">
        <b>👋 أهلاً بك في UniQ!</b><br><br>
        أنا المساعد الذكي الخاص بكلية الحاسبات وعلوم البيانات، جامعة الإسكندرية.<br><br>
        يمكنني مساعدتك في:
        <ul style="text-align:right; direction:rtl;">
            <li>📋 اللائحة الداخلية والأحكام الأكاديمية</li>
            <li>📊 حساب المعدلات وفهم نظام التقييم</li>
            <li>📚 الخطط الدراسية والمقررات</li>
            <li>📝 إجراءات التسجيل والحذف والانسحاب</li>
            <li>🎓 متطلبات التخرج ومرتبة الشرف</li>
        </ul>
        <b>ابدأ بسؤالك! 😊</b>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        fallback_class = "bot-bubble-fallback" if msg.get("fallback") else ""
        icon = "⚠️" if msg.get("fallback") else "🤖"
        st.markdown(f'<div class="bot-bubble {fallback_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        # Show sources if available
        if msg.get("sources"):
            with st.expander("📎 المصادر المستخدمة"):
                for src in msg["sources"]:
                    score_pct = int(src.get("score", 0) * 100)
                    st.markdown(f"""
                    <span class="source-badge">📄 {src.get('title', 'مستند')} — {score_pct}% تطابق</span>
                    """, unsafe_allow_html=True)
                    st.caption(src.get("content_preview", "")[:200] + "...")

        # Suggested actions
        if msg.get("suggested_actions"):
            cols = st.columns(len(msg["suggested_actions"]))
            for i, action in enumerate(msg["suggested_actions"]):
                with cols[i]:
                    if st.button(action, key=f"action_{id(msg)}_{i}"):
                        st.session_state.messages.append({"role": "user", "content": action})
                        st.session_state._pending_question = action
                        st.rerun()


# ── Process pending question ──────────────────────────────────────────────────
def process_question(question: str):
    if not st.session_state.pipeline_ready:
        return {
            "answer": "⚠️ يرجى تحميل النموذج أولاً من القائمة الجانبية.",
            "sources": [], "intent": None,
            "suggested_actions": [], "fallback": True,
        }

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude current question
        if m["role"] in ("user", "assistant")
    ][-10:]

    result = st.session_state.pipeline.query(
        question=question,
        history=history,
    )
    return result


if hasattr(st.session_state, "_pending_question") and st.session_state._pending_question:
    question = st.session_state._pending_question
    st.session_state._pending_question = None

    with st.spinner("يونيك بوت يفكر... 🤔"):
        start = time.time()
        result = process_question(question)
        elapsed = int((time.time() - start) * 1000)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "intent": result.get("intent"),
        "suggested_actions": result.get("suggested_actions", []),
        "fallback": result.get("fallback", False),
        "latency_ms": elapsed,
    })
    st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "اكتب سؤالك هنا... (بالعربية أو الإنجليزية)",
    disabled=not st.session_state.pipeline_ready,
)

if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.session_state._pending_question = user_input.strip()
    st.rerun()

# Hint when model not loaded
if not st.session_state.pipeline_ready:
    st.info("👈 اضغط على **'تحميل النموذج'** من القائمة الجانبية للبدء")
