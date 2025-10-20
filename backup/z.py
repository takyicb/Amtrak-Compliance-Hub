"""
Query local Privacy-Law RAG (Chroma + OpenAI).
Author: Takyi Boamah
"""

import os, argparse, textwrap, json, sys
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────────
DB_PATH = "./chroma_privacy_law_db"
COLLECTION_NAME = "privacy_law_collection"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = 8500  # keep under model limits comfortably
TOP_K = 6


# ── Init ────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"
client = OpenAI(api_key=OPENAI_API_KEY)

chroma = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def build_context(chunks):
    """
    Build a numbered context block with lightweight labels for clean citations.
    """
    lines, total = ["Context Chunks:"], 0
    for i, c in enumerate(chunks, start=1):
        src = c["meta"].get("source", "Unknown Source")
        sec = c["meta"].get("section")
        jdx = c["meta"].get("jurisdiction")
        label = f"[{i}] Source: {src}"
        if sec: label += f" — {sec}"
        if jdx: label += f" — {jdx}"
        block = f"{label}\n{c['text'].strip()}\n"
        if total + len(block) > MAX_CONTEXT_CHARS: break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)

def query_chunks(question, k=TOP_K):
    res = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # ← removed "ids"
    )
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids   = res["ids"][0]  # still present even if not listed in include

    rows = sorted(zip(docs, metas, ids, dists), key=lambda r: r[3])
    chunks = [{"text": d, "meta": m, "id": i, "distance": dist} for d, m, i, dist in rows]
    return chunks

SYSTEM_PROMPT = """You are a meticulous legal RAG assistant for U.S. privacy statutes.
Answer ONLY using the provided Context Chunks. If the answer is not fully supported by the chunks, say "Not found in supplied context."
Never invent section numbers, definitions, dates, thresholds, or remedies.

Output format (JSON):
{
  "answer": "<concise plain-English answer, <=120 words, no hedging>",
  "citations": [
    {"doc": "<short source name>", "section_or_part": "<as shown in chunk>"}
  ],
  "direct_quotes": ["<at most one short quote (<=25 words) if essential>"],
  "notes": "<optional 1-line nuance or limitation>",
  "confidence": "low|medium|high"
}

Rules:
- Prefer the narrowest directly-on-point clause (definitions → rights → duties → exemptions → enforcement).
- Include the exact section label or heading when available (e.g., "§ 12D-104" or "§ 59.1-577").
- If multiple jurisdictions differ, state the jurisdiction explicitly.
- If a requirement is conditional (thresholds, scope, exemptions), state the condition tersely.
- Do NOT provide legal advice; summarize statutory text only.
- Return only the JSON object; no extra prose.
"""

def ask_llm(question, context, model=DEFAULT_MODEL):
    user_prompt = f"{context}\n\nQuestion:\n{question}\n\nReturn only the JSON object."
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        top_p=0.9,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content

def pretty_support(chunks):
    print("\n--- Retrieved Chunks (top-k) ---")
    for i, c in enumerate(chunks, start=1):
        src = c["meta"].get("source", "Unknown Source")
        sec = c["meta"].get("section", "")
        jdx = c["meta"].get("jurisdiction", "")
        dist = c["distance"]
        header = f"[{i}] {src}"
        if sec: header += f" — {sec}"
        if jdx: header += f" — {jdx}"
        print(header + f"  (distance={dist:.4f})")
        snippet = textwrap.shorten(" ".join(c["text"].split()), width=240, placeholder="…")
        print("  " + snippet)

# ── CLI ────────────────────────────────────────────────────────────────────────   

def main():
    parser = argparse.ArgumentParser(description="Query local Privacy-Law RAG")
    parser.add_argument("--q", "--question", dest="question", required=True, help="Your question")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k chunks")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI chat model")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    chunks = query_chunks(args.question, k=args.k)
    if not chunks:
        print("No chunks retrieved. Is the collection empty?")
        sys.exit(1)

    context = build_context(chunks)
    raw = ask_llm(args.question, context, model=args.model)

    # Try to pretty-print JSON; if it fails, show raw
    print("\n=== Answer (JSON) ===")
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(raw)

    if args.show_chunks:
        pretty_support(chunks)

if __name__ == "__main__":
    main()