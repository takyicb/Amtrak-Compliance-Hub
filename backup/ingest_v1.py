"""
Ingest U.S. Privacy Law texts into a local ChromaDB for RAG use.
Author: Takyi Boamah
"""

import os, re, time, hashlib
from itertools import islice
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions


# ── Config ──────────────────────────────────────────────────────────────────────
DATA_DIR = "./Data"  
DB_PATH = "./chroma_privacy_law_db"
COLLECTION_NAME = "privacy_law_collection"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE = 64


# ── Init ────────────────────────────────────────────────────────────────────────
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
assert openai_key, "Missing OPENAI_API_KEY in .env file!"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name=EMBED_MODEL
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)


# ── Initial test ───────────────────────────────────────────────────────────────
# resp = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the United States?",
#         },
#     ],
# )
# print(resp.choices[0].message)



# ── Helpers ─────────────────────────────────────────────────────────────────────
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def normalize(text: str) -> str:
    """Clean up whitespace and newlines."""
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_jurisdiction(filename: str) -> str:
    f = filename.lower()
    if "california" in f: return "California"
    if "delaware" in f: return "Delaware"
    if "illinois" in f: return "Illinois"
    if "virginia" in f: return "Virginia"
    return "Unknown"

SECTION_PATTERN = re.compile(
    r'(?=^(?:\s*§+\s*\S+|'                       # § 12D-104 …
    r'\s*(?:Sec\.|Section)\s*\d+[^\n]*|'         # Sec. 15 … / Section 15 …
    r'\s*\(740 ILCS[^\n]*|'                      # (740 ILCS 14/15) …
    r'\s*Chapter\s+\w+[^\n]*))',                 # Chapter 53 …
    r'^(?:Article|ART\.)\s*[IVXLC]+\s*[^\n]*|'   # Article IV … / ART. IV …
    re.MULTILINE
)

HEADING_PATTERN = re.compile(
    r'^\s*(§+\s*\S+|'                            # § 12D-104 …
    r'(?:Sec\.|Section)\s*\d+[^\n]*|'            # Sec. 15 … / Section 15 …
    r'\(740 ILCS[^\n]*|'                         # (740 ILCS 14/15) …
    r'Chapter\s+\w+[^\n]*)(?::)?\s*',            # Chapter 53 …
    re.IGNORECASE
)

def split_into_sections(text: str):
    """Split a law text into (heading, body) pairs by section markers."""
    parts = SECTION_PATTERN.split(text)
    blocks, buf = [], ""
    for p in parts:
        if not p:
            continue
        if HEADING_PATTERN.match(p) and buf:
            blocks.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        blocks.append(buf)

    results = []
    for b in blocks:
        first_line = b.split("\n", 1)[0]
        m = HEADING_PATTERN.match(first_line)
        heading = m.group(0).strip() if m else ""
        body = b[len(heading):].strip() if heading and b.startswith(heading) else b.strip()
        heading = re.sub(r"\s*[:\-–]\s*$", "", heading)
        results.append((heading, body))
    return results or [("", text)]

def chunk_section(heading, body, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Break section body into overlapping chunks."""
    chunks, start = [], 0
    n = len(body)
    while start < n:
        end = min(start + size, n)
        chunk = body[start:end].strip()
        if chunk:
            chunks.append((heading, chunk))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

# ── Ingest ──────────────────────────────────────────────────────────────────────
def build_records(filename, text):
    jurisdiction = detect_jurisdiction(filename)
    sections = split_into_sections(text)
    records = []

    for s_idx, (heading, body) in enumerate(sections, start=1):
        for c_idx, (h, chunk) in enumerate(chunk_section(heading, body), start=1):
            rid = f"{filename}::sec{s_idx:04d}::chunk{c_idx:03d}::{sha1(chunk)}"
            meta = {
                "source": filename,
                "jurisdiction": jurisdiction,
                "section": h or "Unlabeled section",
                "sec_index": s_idx,
                "chunk_index": c_idx
            }
            records.append({"id": rid, "text": chunk, "meta": meta})
    return records

def ingest():
    print(f"=== Loading .txt laws from {DATA_DIR} ===")
    all_records = []

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = normalize(f.read())
        recs = build_records(filename, raw)
        all_records.extend(recs)
        print(f"Loaded {filename}: {len(recs)} chunks")

    print(f"\nTotal chunks to upsert: {len(all_records)}")

    for batch in batched(all_records, BATCH_SIZE):
        ids = [r["id"] for r in batch]
        docs = [r["text"] for r in batch]
        metas = [r["meta"] for r in batch]
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        time.sleep(0.05)

    print(f"\n✅ Successfully ingested {collection.count()} chunks into {COLLECTION_NAME}")


# ── Run ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest()