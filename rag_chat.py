import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv  # 1. Import the loader
from retriever import rag_query
from helper import extract_chunk_image


load_dotenv()
# =========================
# CONFIG
# =========================
PDF_PATH = Path("_10-K-2025-As-Filed.pdf")
API_KEY = os.getenv("OPENROUTER_API_KEY")  # set in environment

if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in environment variables")


# Free / strong open model via OpenRouter
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# =========================
# SYSTEM PROMPT (IMPORTANT)
# =========================
SYSTEM_PROMPT = """
You are a strict document QA assistant.

RULES:
1. Answer ONLY from the given context.
2. Do NOT use outside knowledge.
3. If answer is missing, say: "Not found in document."
4. Be concise and factual.
5. Format output in clean markdown.
6. Always prefer exact phrases from context.
"""


# =========================
# CALL LLM (OPENROUTER)
# =========================
def generate_answer(question, context):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}

Answer in markdown:
"""
            }
        ],
        "temperature": 0.0
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(response.text)
        raise Exception("LLM request failed")

    return response.json()["choices"][0]["message"]["content"]


# =========================
# BUILD CONTEXT
# =========================
def build_context(chunks):

    context = ""

    for c in chunks:
        context += f"""
PAGE: {c.get('page')}
TYPE: {c.get('type')}

TEXT:
{c.get('text')}

-------------------------
"""

    return context


# =========================
# VISUAL GROUNDING (BEST CHUNK)
# =========================
def show_grounding(chunks):

    if not chunks:
        return

    best = chunks[0]  # highest score from FAISS

    bbox = best.get("bbox")
    page = best.get("page")

    if bbox and page is not None:
        try:
            extract_chunk_image(
                pdf_path=PDF_PATH,
                page_num=page,
                bbox=bbox,
                output_path="evidence.png"
            )
            print("\n📌 Evidence saved: evidence.png")
        except Exception as e:
            print("Grounding failed:", e)


# =========================
# MAIN LOOP
# =========================
def main():

    print("\n🚀 RAG Chat Started (type 'exit' to quit)\n")

    while True:

        question = input("Ask: ")

        if question.lower() == "exit":
            break

        # 1. Retrieve
        chunks = rag_query(question, top_k=5)

        # 2. Build context
        context = build_context(chunks)

        # 3. Generate answer
        answer = generate_answer(question, context)

        # 4. Output
        print("\n" + "="*80)
        print("ANSWER")
        print("="*80)
        print(answer)

        # 5. Visual grounding
        show_grounding(chunks)


if __name__ == "__main__":
    main()