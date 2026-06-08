import json
from pathlib import Path
import fitz
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display


# =========================
# LOAD CHUNKS (RAG FORMAT)
# =========================
CHUNKS_PATH = Path("parsed_output/_10-K-2025-As-Filed_chunks.json")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# =========================
# BASIC INSPECTION
# =========================
print("\nSample chunk:")
print(json.dumps(chunks[0], indent=2)[:1000])


# =========================
# CHUNK TYPE DISTRIBUTION
# =========================
chunk_type_counts = {}

for c in chunks:
    t = c.get("type")   # correct field from your sample
    chunk_type_counts[t] = chunk_type_counts.get(t, 0) + 1

print("\nChunk type distribution:")
print(chunk_type_counts)


# =========================
# VISUALIZATION FUNCTION (FIXED)
# =========================
def visualize_page_chunks(pdf_path, chunks, page_num=5, zoom=2.0):

    doc = fitz.open(pdf_path)
    page = doc[page_num]  # <-- FIX: zero-based indexing

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img = Image.fromarray(
        np.frombuffer(pix.samples, dtype=np.uint8)
        .reshape(pix.height, pix.width, pix.n)
    )

    draw = ImageDraw.Draw(img)

    page_chunks = [
        c for c in chunks
        if c.get("grounding", {}).get("page") == page_num
    ]

    print(f"Found {len(page_chunks)} chunks on page {page_num}")

    has_bbox = any(c.get("grounding", {}).get("box") for c in page_chunks)

    if not has_bbox:
        print("⚠️ No bounding boxes found (unexpected if using raw ADE output)")

    for i, c in enumerate(page_chunks):

        bbox = c.get("grounding", {}).get("box")

        if bbox:
            x1 = bbox["left"] * pix.width
            y1 = bbox["top"] * pix.height
            x2 = bbox["right"] * pix.width
            y2 = bbox["bottom"] * pix.height

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            label = f"{c.get('id')} | {c.get('type')}"
            draw.text((x1, max(0, y1 - 15)), label, fill="blue")

    output_path = f"page_{page_num}_chunks.png"
    img.save(output_path)

    print(f"Saved visualization to: {output_path}")

    return img


# =========================
# RUN VISUALIZATION
# =========================
document_path = Path("_10-K-2025-As-Filed.pdf")

img = visualize_page_chunks(
    document_path,
    chunks,
    page_num=3   # IMPORTANT FIX
)

display(img)


# =========================
# FILTER FOR RAG
# =========================
text_chunks = [
    c for c in chunks
    if c.get("type") in ["text", "paragraph", "list", "table"]
]

print(f"\nText-like chunks: {len(text_chunks)}")