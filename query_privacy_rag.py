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
def query_chunks(question, k=TOP_K):
    if collection.count() == 0:
        print("⚠️  Collection is empty. Run ingest_privacy_laws.py first.")
        sys.exit(1)

    res = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    rows = sorted(zip(docs, metas, dists), key=lambda r: r[2])
    return rows

def build_context(rows):
    out, total = ["Context Chunks:"], 0
    for i, (doc, meta, dist) in enumerate(rows, 1):
        src = meta.get("source_file", "Unknown Source")
        sec = meta.get("section_id", "")
        law = meta.get("law", "")
        jdx = meta.get("jurisdiction", "")
        sd = meta.get("law_signed_date_provided", "")
        ed = meta.get("law_effective_date_provided", "")
        label = f"[{i}] {law} — {sec} ({jdx})"
        if sd or ed:
            label += f" [Signed: {sd or '?'} | Effective: {ed or '?'}]"
        block = f"{label}\n{doc.strip()}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)

def ask_llm(question, context):
    system_prompt = (
        "You are a legal RAG assistant for U.S. privacy laws. "
        "Answer ONLY using the provided context. "
        "If not found, say 'Not found in supplied context.' "
        "Keep your answer under 120 words, concise but accurate."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=350,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nQuestion:\n{question}"},
        ],
    )
    return resp.choices[0].message.content.strip()

def print_support(rows):
    print("\n=== CONTEXT USED ===")
    for i, (doc, meta, dist) in enumerate(rows, 1):
        law = meta.get("law", "")
        sec = meta.get("section_id", "")
        jdx = meta.get("jurisdiction", "")
        sd = meta.get("law_signed_date", "")
        ed = meta.get("law_effective_date", "")
        snippet = textwrap.shorten(" ".join(doc.split()), width=180, placeholder="…")
        print(f"[{i}] {law} — {sec} ({jdx}) [Signed {sd or '?'} | Effective {ed or '?'}] (distance={dist:.4f})")
        print("    " + snippet)


# ── Run ──────────────────────────────────────────────────────────────────────── 
if __name__ == "__main__":
    try:
  
        question = input("\nEnter your question:\n> ").strip()
        if not question:
            print("No question entered.")
            sys.exit(0)

        rows = query_chunks(question, TOP_K)
        context = build_context(rows)

        # show which law(s) are most relevant
        top_meta = rows[0][1]
        top_law = top_meta.get("law", "")
        sd = top_meta.get("law_signed_date", "")
        ed = top_meta.get("law_effective_date", "")

        answer = ask_llm(question, context)

        print("\n=== ANSWER ===")
        if top_law:
            print(f"📘 {top_law} — Signed {sd or '?'} | Effective {ed or '?'}\n")
        print(answer)
        print_support(rows)

    except KeyboardInterrupt:
        print("\nCancelled.")