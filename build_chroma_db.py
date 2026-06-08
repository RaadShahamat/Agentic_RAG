import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)
# =========================
# LOAD CHUNKS
# =========================
CHUNKS_PATH = Path("parsed_output/_10-K-2025-As-Filed_chunks.json")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loaded:", len(chunks)) 


# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

dim = 1024  # Qwen embedding size


# =========================
# FAISS INDEX
# =========================
index = faiss.IndexFlatIP(dim)   # cosine-like similarity (after normalization)

metadata_store = []


# =========================
# BUILD VECTORS
# =========================
embeddings = []

for i, c in enumerate(chunks):

    text = c.get("markdown", "")
    if not text.strip():
        continue

    emb = model.encode(text)

    embeddings.append(emb)

    metadata_store.append({
        "id": c["id"],
        "text": text,
        "page": c.get("grounding", {}).get("page"),
        "type": c.get("type"),
        "bbox": c.get("grounding", {}).get("box"),
    })

    if i % 50 == 0:
        print("processed", i)


# =========================
# TO NUMPY
# =========================
embeddings = np.array(embeddings).astype("float32")


# IMPORTANT: normalize for cosine similarity
faiss.normalize_L2(embeddings)


# =========================
# ADD TO INDEX
# =========================
index.add(embeddings)

print("FAISS index size:", index.ntotal)

print("Saving FAISS index...")
faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

print("Saving metadata...")
with open(FAISS_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, ensure_ascii=False, indent=2)

print("Done.")