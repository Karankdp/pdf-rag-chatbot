"""
Chat with your PDFs — Enhanced Streamlit app
==============================================
RAG chatbot with a more interactive UI: live stats, example question chips,
source relevance bars, chat download, and custom styling.
"""

import uuid
from datetime import datetime

import streamlit as st
from google import genai
import chromadb
from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.6-flash"

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")

# --- Custom styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9ca3af;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border-radius: 10px;
        padding: 10px 15px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .example-chip button {
        border-radius: 20px !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📄 Chat with your PDFs</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload documents, then ask questions grounded in their content.</p>', unsafe_allow_html=True)


# --- API key ---
try:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    api_key = None
if not api_key:
    api_key = st.sidebar.text_input("🔑 Gemini API key", type="password")
if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

client = genai.Client(api_key=api_key)


# --- Pipeline functions ---
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if end < text_length:
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                end = start + break_point + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_documents(documents):
    all_chunks, chunk_metadata, chunk_ids = [], [], []
    for filename, text in documents.items():
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(chunk)
            chunk_metadata.append({"source": filename, "chunk_index": i})
            chunk_ids.append(str(uuid.uuid4()))
    return all_chunks, chunk_metadata, chunk_ids


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBED_MODEL, contents=text, config={"task_type": task_type},
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings


def retrieve(collection, query, top_k=TOP_K):
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks)
    prompt = f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make things up.
Cite the source filename when relevant.

Context:
{context}

Question: {query}

Answer:"""
    interaction = client.interactions.create(model=GEN_MODEL, input=prompt)
    return interaction.output_text


def distance_to_relevance_pct(distance):
    """Rough conversion of cosine distance to a 0-100% relevance display."""
    return max(0, min(100, round((1 - distance) * 100)))


# --- Session state ---
for key, default in [
    ("collection", None),
    ("messages", []),
    ("doc_names", []),
    ("chunk_count", 0),
    ("questions_asked", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()


# --- Sidebar ---
with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if st.button("⚡ Process PDFs", type="primary", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Extracting, chunking, and embedding..."):
            progress = st.progress(0, text="Extracting text...")
            documents = {f.name: extract_text_from_pdf(f) for f in uploaded_files}
            progress.progress(33, text="Chunking...")
            chunks, metadata, ids = chunk_documents(documents)
            progress.progress(66, text="Generating embeddings...")

            collection = st.session_state.chroma_client.get_or_create_collection(
                f"session_{uuid.uuid4().hex[:8]}"
            )
            embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadata)
            progress.progress(100, text="Done!")

            st.session_state.collection = collection
            st.session_state.doc_names = list(documents.keys())
            st.session_state.chunk_count = len(chunks)
            st.session_state.messages = []
            st.session_state.questions_asked = 0
        st.success(f"✅ Processed {len(uploaded_files)} file(s)")

    if st.session_state.doc_names:
        st.subheader("Loaded documents")
        for name in st.session_state.doc_names:
            st.markdown(f"📄 {name}")

    st.divider()

    if st.session_state.collection is not None:
        st.subheader("📊 Session stats")
        c1, c2 = st.columns(2)
        c1.metric("Chunks indexed", st.session_state.chunk_count)
        c2.metric("Questions asked", st.session_state.questions_asked)

    st.divider()

    if st.session_state.messages:
        chat_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download chat",
            data=chat_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            use_container_width=True,
        )
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# --- Main area ---
if st.session_state.collection is None:
    st.info("👈 Upload and process a PDF in the sidebar to start chatting.")
else:
    # Example question chips (only before first message, for a clean start)
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        example_qs = [
            "Summarize this document in 3 bullet points",
            "What are the key steps or sections?",
            "What's the main goal or purpose here?",
        ]
        cols = st.columns(len(example_qs))
        clicked_example = None
        for col, q in zip(cols, example_qs):
            with col:
                st.markdown('<div class="example-chip">', unsafe_allow_html=True)
                if st.button(q, key=f"ex_{q}", use_container_width=True):
                    clicked_example = q
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        clicked_example = None

    # Render chat history
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} sources used"):
                    for s in msg["sources"]:
                        rel = distance_to_relevance_pct(s["distance"])
                        st.caption(f"**{s['source']}** — relevance {rel}%")
                        st.progress(rel / 100)
                        st.caption(s["text"][:200] + "...")

    user_prompt = st.chat_input("Ask a question about your uploaded PDFs...")
    final_prompt = clicked_example or user_prompt

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(final_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching your documents..."):
                retrieved = retrieve(st.session_state.collection, final_prompt)
                answer = generate_answer(final_prompt, retrieved)
                st.markdown(answer)
                with st.expander(f"📚 {len(retrieved)} sources used"):
                    for s in retrieved:
                        rel = distance_to_relevance_pct(s["distance"])
                        st.caption(f"**{s['source']}** — relevance {rel}%")
                        st.progress(rel / 100)
                        st.caption(s["text"][:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved})
        st.session_state.questions_asked += 1
        st.rerun()
