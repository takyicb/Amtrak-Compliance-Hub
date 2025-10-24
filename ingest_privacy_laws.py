"""
Ingest U.S. Privacy Law texts into a local ChromaDB for RAG use.
Author: Takyi Boamah
"""

import os, re, time, hashlib
from typing import List, Dict, Tuple, Iterable
from itertools import islice
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_DIR = "./Data"
DB_PATH = "./chroma_privacy_law_db"
COLLECTION_NAME = "privacy_law_collection"
EMBED_MODEL = "text-embedding-3-small"

# Chunking heuristics (words as a crude token proxy)
TARGET_TOKENS = 500
MAX_TOKENS = 700
MIN_TOKENS = 180
OVERLAP_TOKENS = 100

# Upsert behavior
BATCH_SIZE = 64
SLEEP_BETWEEN_BATCHES = 0.05
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"  # set DRY_RUN=1 to preview without writing

# ========== OFFICIAL LAW DATES (enacted + effective) ==========
LAW_DATES_RAW = {
    "California Consumer Privacy Act (as amended)": {
        "signed":   "6/28/2018",
        "effective":"1/1/2020",
    },
    "Virginia Consumer Data Protection Act": {
        "signed":   "3/2/2021",
        "effective":"1/1/2023",
    },
    "Delaware Personal Data Privacy Act (2023)": {
        "signed":   "9/11/2023",
        "effective":"1/1/2025",
    },
    "Illinois Biometric Information Privacy Act (BIPA)": {
        "signed":   "10/3/2008",
        "effective":"10/3/2008",
    },
    "New Jersey SB 332 (2025)": {
        "signed":   "1/16/2024",
        "effective":"1/15/2025",
    },
    "New Hampshire SB 255 (2025)": {
        "signed":   "3/6/2024",
        "effective":"1/1/2025",
    },
}

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

# ── Helpers ─────────────────────────────────────────────────────────────────────
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def approx_tokens(s: str) -> int:
    # crude word-count proxy for tokens
    return max(1, len(s.split()))

def batched(it: Iterable, n: int):
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk

def _to_iso(date_str: str) -> str:
    """Accept 'M/D/YYYY', 'MM/DD/YYYY', 'M-D-YYYY', etc. Return 'YYYY-MM-DD' or ''."""
    if not date_str:
        return ""
    candidates = [date_str.strip(),
                  date_str.strip().replace("-", "/"),
                  date_str.strip().replace(".", "/")]
    for ds in candidates:
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(ds, fmt).date().isoformat()
            except ValueError:
                continue
    return ""

def get_law_dates_iso(law_name: str) -> dict:
    d = LAW_DATES_RAW.get(law_name, {})
    return {
        "law_signed_date":    _to_iso(d.get("signed", "")),
        "law_effective_date": _to_iso(d.get("effective", "")),
        "law_signed_date_provided":    d.get("signed", ""),
        "law_effective_date_provided": d.get("effective", ""),
    }

# ========== JURISDICTION + SECTION DETECTION ==========
def detect_jurisdiction(filename: str, text: str) -> str:
    f = filename.lower()
    if "california" in f or re.search(r"\b1798\.\d+", text): return "California"
    if "delaware" in f or "12d-" in text.lower(): return "Delaware"
    if "virginia" in f or re.search(r"\b59\.1-\d+", text): return "Virginia"
    if "new jersey" in f or "c.56:8-166" in text.lower(): return "New Jersey"
    if "illinois" in f or "ilcs" in text.lower(): return "Illinois"
    if "new hampshire" in f: return "New Hampshire"
    return "Unknown"

def detect_law_name(filename: str) -> str:
    f = filename.lower()
    if "delaware" in f: return "Delaware Personal Data Privacy Act (2023)"
    if "virginia" in f: return "Virginia Consumer Data Protection Act"
    if "california" in f: return "California Consumer Privacy Act (as amended)"
    if "new jersey" in f: return "New Jersey SB 332 (2025)"
    if "illinois" in f: return "Illinois Biometric Information Privacy Act (BIPA)"
    if "new hampshire" in f: return "New Hampshire SB 255 (2025)"
    return os.path.splitext(filename)[0]

SECTION_PATTERNS = {
    "Delaware":      re.compile(r'^(§\s*12D-\d+[A-Za-z\-\.]*\.)', re.MULTILINE),
    "California":    re.compile(r'^(1798\.\d+[A-Za-z\-\.]*\.)', re.MULTILINE),
    "Virginia":      re.compile(r'^(§\s*59\.1-\d+[A-Za-z\-\.]*\.)', re.MULTILINE),
    "New Jersey":    re.compile(r'^(C\.56:8-166\.\d+[A-Za-z\-\.]*)', re.MULTILINE),
    # Illinois: allow either "(740 ILCS ...)" line or "Section N."
    "Illinois":      re.compile(r'^(\(740 ILCS[^)]+\)|Section\s+\d+[\.:])', re.MULTILINE),
    # New Hampshire: "Section N." or "§ N."
    "New Hampshire": re.compile(r'^(Section\s+\d+[\.:]|§\s*\d+[\.:])', re.MULTILINE),
    # Fallback
    "Unknown":       re.compile(r'^(§\s*\S+|Section\s+\d+[\.:]|Chapter\s+\w+[^:\n]*)', re.MULTILINE),
}

HEADING_CLEAN = re.compile(r'\s*[:\-–]\s*$')

def split_by_sections(text: str, jurisdiction: str) -> List[Tuple[str, str]]:
    pat = SECTION_PATTERNS.get(jurisdiction, SECTION_PATTERNS["Unknown"])
    parts = pat.split(text)
    blocks, buf = [], ""
    for p in parts:
        if not p:
            continue
        if pat.match(p) and buf:
            blocks.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        blocks.append(buf)

    results = []
    for b in blocks:
        first = b.split("\n", 1)[0]
        m = pat.match(first)
        heading = m.group(0).strip() if m else ""
        body = b[len(heading):].strip() if heading and b.startswith(heading) else b.strip()
        heading = HEADING_CLEAN.sub("", heading)
        results.append((heading, body))
    if all(h == "" for h, _ in results):
        return [("Unlabeled section", text.strip())]
    return results

# Subsection splitters
SUB_AZ = re.compile(r'(?m)^\([a-z]\)\s')      # (a)
SUB_NUM = re.compile(r'(?m)^\(\d+\)\s')       # (1)

def split_subsections(heading: str, body: str) -> List[Tuple[str, str, List[str]]]:
    def _split(body: str, pat: re.Pattern):
        parts = pat.split(body)
        blocks, buf = [], ""
        for p in parts:
            if not p: continue
            if pat.match(p) and buf:
                blocks.append(buf); buf = p
            else:
                buf += p
        if buf: blocks.append(buf)
        out = []
        for bl in blocks:
            m1 = re.match(r'^\(([a-z])\)\s', bl)
            m2 = re.match(r'^\((\d+)\)\s', bl)
            marker = f"({m1.group(1)})" if m1 else (f"({m2.group(1)})" if m2 else None)
            out.append((marker, bl))
        return out

    az_blocks = _split(body, SUB_AZ)
    if len(az_blocks) <= 1:
        return [(heading, body.strip(), [heading])]

    results = []
    for az_marker, az_text in az_blocks:
        num_blocks = _split(az_text, SUB_NUM)
        if len(num_blocks) <= 1:
            eff = f"{heading} {az_marker}" if az_marker else heading
            results.append((eff.strip(), az_text.strip(), [heading] + ([az_marker] if az_marker else [])))
        else:
            for num_marker, num_text in num_blocks:
                eff = " ".join(filter(None, [heading, az_marker, num_marker]))
                results.append((eff.strip(), num_text.strip(),
                                [heading] + ([az_marker] if az_marker else []) + ([num_marker] if num_marker else [])))
    return results

SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z(])')  # sentence-ish splitter

def sized_chunks(eff_heading: str, text: str, hierarchy: List[str]) -> List[Tuple[str, str, List[str]]]:
    if approx_tokens(text) <= MAX_TOKENS:
        return [(eff_heading, text, hierarchy)]

    sentences = SENT_SPLIT.split(text)
    chunks, buf, buf_tokens = [], [], 0
    for sent in sentences:
        t = approx_tokens(sent)
        if buf_tokens + t > MAX_TOKENS and buf:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                chunks.append(chunk_text)
            overlap_words = " ".join(chunk_text.split()[-OVERLAP_TOKENS:]) if chunk_text else ""
            buf, buf_tokens = ([overlap_words] if overlap_words else []), approx_tokens(overlap_words) if overlap_words else 0
        buf.append(sent); buf_tokens += t
    if buf:
        chunks.append(" ".join(buf).strip())

    out = []
    for ch in chunks:
        if approx_tokens(ch) > MAX_TOKENS:
            words = ch.split()
            start = 0
            while start < len(words):
                end = min(start + MAX_TOKENS, len(words))
                piece = " ".join(words[start:end]).strip()
                if piece:
                    out.append((eff_heading, piece, hierarchy))
                if end == len(words): break
                start = max(end - OVERLAP_TOKENS, start + 1)
        else:
            if ch.strip():
                out.append((eff_heading, ch, hierarchy))
    return out

def parse_section_id_and_title(heading: str) -> Tuple[str, str]:
    if not heading:
        return "", ""
    m = re.match(r'^(§\s*[0-9A-Za-z\.\-:]+\.?)\s*(.*)$', heading)
    if m:
        return m.group(1).rstrip("."), m.group(2).strip()
    m = re.match(r'^([0-9]+\.[0-9A-Za-z\-]+\.?)\s*(.*)$', heading)  # California 1798.x
    if m:
        return m.group(1).rstrip("."), m.group(2).strip()
    m = re.match(r'^(C\.[0-9:.\-]+)\s*(.*)$', heading)              # New Jersey C.56:8-166.x
    if m:
        return m.group(1), m.group(2).strip()
    m = re.match(r'^\((740 ILCS[^)]+)\)\s*(.*)$', heading)          # Illinois (740 ILCS …)
    if m:
        return m.group(1), m.group(2).strip()
    return heading.strip(), ""

# ========== RECORD BUILD ==========
def build_records(filename: str, text: str) -> List[Dict]:
    law = detect_law_name(filename)
    jurisdiction = detect_jurisdiction(filename, text)
    law_dates = get_law_dates_iso(law)
    sections = split_by_sections(text, jurisdiction)

    records = []
    for s_idx, (section_heading, section_body) in enumerate(sections, start=1):
        sec_id, sec_title = parse_section_id_and_title(section_heading)
        subs = split_subsections(section_heading, section_body)
        for sub_heading, sub_text, hierarchy in subs:
            sized = sized_chunks(sub_heading, sub_text, hierarchy)
            for c_idx, (eff_head, chunk_text, hier) in enumerate(sized, start=1):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                if approx_tokens(chunk_text) < MIN_TOKENS and len(sized) > 1:
                    continue
                content_hash = sha1(chunk_text)
                rec_id = f"{filename}::sec{s_idx:04d}::chunk{c_idx:03d}::{content_hash[:10]}"
                meta = {
                    "source_file": filename,
                    "law": law,
                    "jurisdiction": jurisdiction,
                    "section_id": sec_id or section_heading or "Unlabeled section",
                    "section_title": sec_title,
                    "effective_heading": eff_head,
                    "hierarchy_path": hier,
                    "text_sha1": content_hash,
                    "sec_index": s_idx,
                    "chunk_index": c_idx,
                    # Dates for the statute
                    **law_dates,
                    # Ingestion timestamp (UTC ISO-8601 with Z)
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                records.append({"id": rec_id, "text": chunk_text, "meta": meta})
    return records

# ========== DEDUPE / UPSERT ==========
def get_existing_hashes(hashes: List[str]) -> set:
    existing = set()
    if not hashes:
        return existing
    for batch_hashes in batched(hashes, 100):
        try:
            res = collection.get(
                where={"text_sha1": {"$in": batch_hashes}},
                include=["metadatas"],
                # no limit -> avoid truncation
            )
        except Exception as e:
            print("Warning: collection.get failed while checking existing hashes:", e)
            continue
        for m in (res.get("metadatas") or []):
            if isinstance(m, dict) and "text_sha1" in m:
                existing.add(m["text_sha1"])
    return existing

def upsert_records(records: List[Dict]):
    if not records:
        print("No new records to upsert.")
        return
    print(f"Upserting {len(records)} chunks into '{COLLECTION_NAME}'...")
    for i, group in enumerate(batched(records, BATCH_SIZE), start=1):
        ids = [r["id"] for r in group]
        docs = [r["text"] for r in group]
        metas = [r["meta"] for r in group]
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        if i % 10 == 0:
            print(f"  … batch {i}")
        time.sleep(SLEEP_BETWEEN_BATCHES)
    print("Done.")

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print(f"Reading .txt laws from: {DATA_DIR}")
    all_records: List[Dict] = []

    files = [fn for fn in sorted(os.listdir(DATA_DIR)) if fn.lower().endswith(".txt")]
    if not files:
        print("No .txt files found. Nothing to ingest.")
    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = normalize_text(raw)
        lawname = detect_law_name(fn)
        print(f"— Processing {fn} ({lawname})")
        recs = build_records(fn, text)
        print(f"   sections/subsections/chunks: {len(recs)}")
        all_records.extend(recs)

    # Dedupe by content-hash across entire run
    all_hashes = [r["meta"]["text_sha1"] for r in all_records]
    existing = get_existing_hashes(all_hashes)
    new_records = [r for r in all_records if r["meta"]["text_sha1"] not in existing]

    # Summary
    print(f"\nTotal chunks built: {len(all_records)}")
    print(f"Already in DB (by hash): {len(existing)}")
    print(f"New chunks to upsert: {len(new_records)}")

    # Actually write
    if DRY_RUN:
        print("DRY_RUN=1 — skipping upsert. Preview only.")
    else:
        if new_records:
            upsert_records(new_records)
        else:
            print("Nothing new to insert.")

        # Optional: show total count after ingest
        try:
            print("Collection count now:", collection.count())
        except Exception as e:
            print("Could not fetch collection count:", e)
