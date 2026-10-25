import json
from pathlib import Path
from typing import Iterable
import faiss
from sentence_transformers import SentenceTransformer
from config import CHUNKS_PATH, INDEX_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, DOCUMENTS_DIR

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    overlap = min(overlap, max(0, chunk_size - 1))
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def index_files(file_paths: Iterable[Path]):
    records = []
    for path in file_paths:
        for i, chunk in enumerate(split_text(extract_text(path))):
            records.append({"text": chunk, "source": path.name, "chunk": i + 1})
    if not records:
        raise ValueError("No text found. Scanned PDFs may need OCR first.")
    vectors = get_embedder().encode([r["text"] for r in records],
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)

def index_saved_documents():
    paths = sorted(p for p in DOCUMENTS_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"})
    return index_files(paths)

def retrieve(query: str, top_k: int = TOP_K):
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return []
    index = faiss.read_index(str(INDEX_PATH))
    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if index.ntotal == 0 or not records:
        return []
    qvec = get_embedder().encode([query], convert_to_numpy=True,
        normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qvec, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if 0 <= idx < len(records):
            item = dict(records[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results
