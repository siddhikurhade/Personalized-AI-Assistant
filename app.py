import uuid
from pathlib import Path
import streamlit as st
from config import DOCUMENTS_DIR, MODEL_ID
from database import init_db, save_message, get_history, clear_history
from rag import index_saved_documents, retrieve
from llm import generate_answer

st.set_page_config(page_title="Personalized AI Assistant", page_icon="🧠", layout="wide")
init_db()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Personalized AI Assistant")
st.caption("Open-source LLM · RAG over your documents · Optional LoRA/QLoRA experiment")
st.info(f"Configured model: `{MODEL_ID}`. First run may download several GB of model weights.")

with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Save uploads", use_container_width=True):
        saved = 0
        for upload in uploads or []:
            safe_name = Path(upload.name).name.replace("..", "_")
            (DOCUMENTS_DIR / safe_name).write_bytes(upload.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s). Click index when ready.")
    if st.button("Index uploaded documents", use_container_width=True):
        try:
            with st.spinner("Extracting text and building FAISS index..."):
                count = index_saved_documents()
            st.success(f"Indexed {count} chunks.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
    use_rag = st.toggle("Use document RAG", value=True)
    if st.button("Clear this chat", use_container_width=True):
        clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']} · chunk {src['chunk']} · score {src['score']:.3f}**")
                    st.write(src["text"])

question = st.chat_input("Ask a question or ask about your documents...")
if question:
    history = get_history(st.session_state.session_id, limit=12)
    st.session_state.messages.append({"role": "user", "content": question})
    save_message(st.session_state.session_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)
    sources = []
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if use_rag:
                    sources = retrieve(question)
                answer = generate_answer(question, retrieved=sources, history=history)
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved sources"):
                    for src in sources:
                        st.markdown(f"**{src['source']} · chunk {src['chunk']} · score {src['score']:.3f}**")
                        st.write(src["text"])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        save_message(st.session_state.session_id, "assistant", answer)
    except Exception as exc:
        message = f"An error occurred: {exc}\n\nCheck the terminal and confirm that your computer can load the selected model."
        st.error(message)
        st.session_state.messages.append({"role": "assistant", "content": message})
