from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from config import DOCUMENTS_DIR, MODEL_ID
from rag import index_saved_documents, retrieve
from llm import generate_answer

app = FastAPI(title="Personalized AI Assistant API", version="1.0.0")

class ChatRequest(BaseModel):
    question: str
    use_rag: bool = True

@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID}

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        name = Path(file.filename or "upload.txt").name
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {name}")
        content = await file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large: {name}")
        target = DOCUMENTS_DIR / name.replace("..", "_")
        target.write_bytes(content)
        saved.append(target.name)
    try:
        chunks = index_saved_documents()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"saved_files": saved, "indexed_chunks": chunks}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    sources = retrieve(request.question) if request.use_rag else []
    answer = generate_answer(request.question, retrieved=sources)
    return {"answer": answer, "sources": [
        {"file": s["source"], "chunk": s["chunk"], "score": s["score"], "text": s["text"]}
        for s in sources
    ]}
