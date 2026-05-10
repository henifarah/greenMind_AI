# ingest.py
from loguru import logger
from config import settings
from ingest.readers import read_source
from ingest.chunker import chunk_all_documents
from ingest.embedder import embed_chunks
from ingest.indexer import index_chunks, get_collection_stats

def run_ingestion(sources=None):
    """
    Lance le pipeline complet d'ingestion.
    sources : liste de sources à traiter (défaut: toutes)
    """
    if sources is None:
        sources = settings.SOURCES

    logger.info("=" * 60)
    logger.info("GreenRAG — Pipeline d'ingestion")
    logger.info(f"Sources : {sources}")
    logger.info("=" * 60)

    all_chunks = []

    # ── Étape 1 + 2 : Lecture + Chunking ─────────────────
    for source in sources:
        logger.info(f"Traitement source : {source.upper()}")
        docs = read_source(source, settings.DATA_RAW_PATH)
        chunks = chunk_all_documents(docs)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks à encoder : {len(all_chunks)}")

    # ── Étape 3 : Embedding ───────────────────────────────
    logger.info("Encodage BGE-M3...")
    embedded = embed_chunks(all_chunks)

    # ── Étape 4 : Indexation ──────────────────────────────
    logger.info("Indexation dans Qdrant...")
    indexed = index_chunks(embedded)

    # ── Résumé final ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RÉSUMÉ FINAL")
    get_collection_stats()
    logger.success("✅ Pipeline terminé avec succès !")
    logger.info("=" * 60)

if __name__ == "__main__":
    # ← On commence par IRENA seulement
    run_ingestion(sources=["irena"])