import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# === CONFIG ===
DB_PATH = "./chroma_privacy_law_db"
COLLECTION_NAME = "privacy_law_collection"
EMBED_MODEL = "text-embedding-3-small"

# === INIT ===
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

# === CHECK & SAMPLE ===
print(f"DB_PATH: {DB_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print("Count:", collection.count())

res = collection.get(limit=3, include=["metadatas", "documents"])
metas = res.get("metadatas", [])
docs = res.get("documents", [])

for m, d in zip(metas, docs):
    law = m.get("law", "(unknown law)")
    section = m.get("section_id", m.get("section", "(no section)"))
    signed = m.get("law_signed_date", m.get("law_signed_date_provided", ""))
    effective = m.get("law_effective_date", m.get("law_effective_date_provided", ""))
    print(f"{law} — {section} | signed: {signed} | effective: {effective}")
    print(d[:180].replace("\n", " "), "…")
