import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from helper import extract_chunk_image
from pathlib import Path

# =========================
# LOAD INDEX + METADATA
# =========================
INDEX_PATH = "faiss_store/index.faiss"
METADATA_PATH = "faiss_store/metadata.json"
PDF_PATH = Path("_10-K-2025-As-Filed.pdf")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


# =========================
# RAG QUERY FUNCTION
# =========================
def rag_query(question, top_k=5, threshold=0.25, show_images=True):

    print(f"\n🔎 Query: {question}\n")
    print("=" * 80)

    # 1. Embed query
    q_emb = model.encode(question, normalize_embeddings=True)
    q_emb = np.array([q_emb], dtype=np.float32)

    # 2. Search FAISS
    scores, indices = index.search(q_emb, top_k)

    results = []
    found_any = False

    # 3. Process results
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):

        if idx == -1:
            continue

        meta = metadata_store[idx]

        similarity = float(score)

        if similarity < threshold:
            continue

        found_any = True

        print(f"\n📌 Result {rank+1} | similarity={similarity:.3f}")
        print(f"ID   : {meta['id']}")
        print(f"Page : {meta.get('page')}")
        print(f"Type : {meta.get('type')}")
        print("-" * 60)
        print(meta.get("text", "")[:500])
        print("-" * 60)

        # 4. Visual grounding
        if show_images and meta.get("bbox"):

            print("🖼 Extracting evidence...")

            extract_chunk_image(
                pdf_path=PDF_PATH,
                page_num=meta["page"],
                bbox=meta["bbox"],
                output_path=f"evidence_rank_{rank+1}.png"
            )

        results.append(meta)

    if not found_any:
        print("❌ No strong matches found.")

    return results