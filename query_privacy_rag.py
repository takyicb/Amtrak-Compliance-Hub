"""
Query local Privacy-Law RAG (Chroma + OpenAI).
Author: Takyi Boamah
"""

import os, sys, textwrap, json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────────

DB_PATH = "./chroma_privacy_law_db"
COLLECTION_NAME = "privacy_law_collection"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # or "gpt-3.5-turbo" "gpt-4o-mini"
TOP_K = 5
MAX_CONTEXT_CHARS = 8500


# ── Init ────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

chroma = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef  
)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def query_chunks(question: str, k: int = TOP_K):
    if collection.count() == 0:
        print("ERROR: Collection is empty. Run ingest_privacy_laws.py first.")
        sys.exit(1)
    try:
        res = collection.query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas", "distances"],  # no "ids" here
        )
    except Exception as e:
        # Common pitfall: dimension mismatch
        if "dimension" in str(e).lower():
            print("ERROR: Embedding dimension mismatch.\n"
                  f"- Ingest used: {EMBED_MODEL} (1536-dim if *-small)\n"
                  "- Make sure this script uses the SAME EMBED_MODEL and COLLECTION_NAME/DB_PATH.\n"
                  "Tip: set EMBED_MODEL in .env to match ingestion.")
        raise
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0] or [{} for _ in docs]
    dists = res.get("distances", [[]])[0] or [0.0 for _ in docs]
    rows = sorted(zip(docs, metas, dists), key=lambda r: r[2])
    return rows

def build_context(rows):
    out, total = ["Context Chunks:"], 0
    for i, (doc, meta, dist) in enumerate(rows, 1):
        src = meta.get("source", "Unknown Source")
        sec = meta.get("section", "")
        jdx = meta.get("jurisdiction", "")
        label = f"[{i}] {src}" + (f" — {sec}" if sec else "") + (f" — {jdx}" if jdx else "")
        block = f"{label}\n{doc.strip()}\n"
        if total + len(block) > MAX_CONTEXT_CHARS: break
        out.append(block); total += len(block)
    return "\n".join(out)

SYSTEM_PROMPT = (
    "You are a meticulous legal RAG assistant for U.S. privacy statutes.\n"
    'Answer ONLY using the provided Context Chunks. If not found, say "Not found in supplied context." '
    "Never invent section numbers, thresholds, or remedies.\n"
    "Return at most 100 words."
)

def ask_llm(question: str, context: str):
    prompt = f"{context}\n\nQuestion:\n{question}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=280,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

def print_support(rows):
    print("\n=== CONTEXT USED ===")
    for i, (doc, meta, dist) in enumerate(rows, 1):
        src = meta.get("source", "")
        sec = meta.get("section", "")
        jdx = meta.get("jurisdiction", "")
        snippet = textwrap.shorten(" ".join(doc.split()), width=200, placeholder="…")
        print(f"[{i}] {src}" + (f" — {sec}" if sec else "") + (f" — {jdx}" if jdx else "")
              + f"  (distance={dist:.4f})")
        print("    " + snippet)

# ── Run ──────────────────────────────────────────────────────────────────────── 
if __name__ == "__main__":
    try:
        question = input("> ").strip()
        if not question:
            print("Please enter a question.")
            sys.exit(0)
        rows = query_chunks(question, TOP_K)
        context = build_context(rows)
        answer = ask_llm(question, context)
        print("\n=== ANSWER ===")
        print(answer)
        print_support(rows)
    except KeyboardInterrupt:
        print("\nCancelled.")