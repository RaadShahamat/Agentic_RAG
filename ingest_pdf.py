import os
import json
from pathlib import Path
from getpass import getpass

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from landingai_ade import LandingAIADE
from landingai_ade.types import ParseResponse


# =========================
# API KEY SETUP
# =========================
if not os.environ.get("VISION_AGENT_API_KEY"):
    os.environ["VISION_AGENT_API_KEY"] = getpass("Enter your Landing AI API key: ")

client = LandingAIADE(apikey=os.environ.get("VISION_AGENT_API_KEY"))
print("Landing AI ADE client initialized.")


# =========================
# INPUT PDF
# =========================
document_path = Path("_10-K-2025-As-Filed.pdf")
assert document_path.exists(), "PDF file not found"


# =========================
# PARSE PDF (ONE TIME COST)
# =========================
print("Calling Landing AI ADE Parse API...")

parse_result: ParseResponse = client.parse(
    document=document_path,
    model="dpt-2-latest"
)

print("Parsing completed.")


# =========================
# OUTPUT DIRECTORY
# =========================
output_dir = Path("parsed_output")
output_dir.mkdir(exist_ok=True)


# =========================
# SAVE MARKDOWN
# =========================
markdown_path = output_dir / f"{document_path.stem}_parsed.md"
markdown_path.write_text(parse_result.markdown, encoding="utf-8")


# =========================
# SAVE FULL JSON RESPONSE
# =========================
json_path = output_dir / f"{document_path.stem}_parse_response.json"
json_path.write_text(
    json.dumps(parse_result.model_dump(), indent=2, ensure_ascii=False),
    encoding="utf-8"
)


# =========================
# SAVE CHUNKS ONLY (MOST IMPORTANT FOR RAG)
# =========================
chunks_path = output_dir / f"{document_path.stem}_chunks.json"

chunks_data = [chunk.model_dump() for chunk in parse_result.chunks]

chunks_path.write_text(
    json.dumps(chunks_data, indent=2, ensure_ascii=False),
    encoding="utf-8"
)


# =========================
# BASIC STATS
# =========================
print("\n=== DONE ===")
print("Markdown:", markdown_path)
print("Full JSON:", json_path)
print("Chunks:", chunks_path)

chunk_type_counts = {}
for c in parse_result.chunks:
    chunk_type_counts[c.type] = chunk_type_counts.get(c.type, 0) + 1

print("\nChunk type distribution:")
print(chunk_type_counts)