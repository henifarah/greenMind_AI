import os
import json
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import uuid
from config import *

def read_file(filepath, source):
    ext = filepath.split(".")[-1].lower()

    if ext == "pdf":
        reader = PdfReader(filepath)
        text = ""
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text += f"[Page {i+1}]\n{t}\n"
        return text

    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(filepath)
        return df.to_string(index=False)

    elif ext == "csv":
        df = pd.read_csv(filepath, encoding="latin-1")
        return df.head(500).to_string(index=False)

    elif ext == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data)

    return ""

# ── Découper en chunks ────────────────────────────────────
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

# ── Pipeline principal ────────────────────────────────────
def run():
    # Charger le modèle d'embedding
    print("Chargement BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connexion Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Créer la collection si elle n'existe pas
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{QDRANT_COLLECTION}' créée")

    # Lire tous les fichiers
    all_chunks = []

    for source in SOURCES:
        source_path = os.path.join(DATA_PATH, source)
        if not os.path.exists(source_path):
            continue

        files = os.listdir(source_path)
        print(f"\n{source.upper()} : {len(files)} fichiers")

        for filename in files:
            if filename.startswith("."):
                continue

            filepath = os.path.join(source_path, filename)
            text = read_file(filepath, source)

            if not text or len(text) < 100:
                continue

            chunks = split_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text":     chunk,
                    "source":   source,
                    "filename": filename,
                    "chunk_id": i
                })

            print(f"  ✅ {filename} → {len(chunks)} chunks")

    print(f"\nTotal : {len(all_chunks)} chunks")

    # Encoder et stocker par batch
    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        vectors = model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload={
                    "text":     batch[j]["text"],
                    "source":   batch[j]["source"],
                    "filename": batch[j]["filename"],
                    "chunk_id": batch[j]["chunk_id"]
                }
            )
            for j in range(len(batch))
        ]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

    print(f"\n✅ {len(all_chunks)} chunks stockés dans Qdrant")

if __name__ == "__main__":
    run()