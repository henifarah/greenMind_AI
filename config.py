import os
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env
# (clés API, secrets — jamais committés sur GitHub)
load_dotenv()

class Settings:

    # ── LLM ──────────────────────────────────────────────
    OPENAI_API_KEY: str = ""           # pas besoin avec Ollama
    LLM_MODEL: str = "llama3.2:3b"         # modèle Ollama
    LLM_MODEL_COMPLEX: str = "gemma2"  # pour plus tard
    LLM_TEMPERATURE: float = 0.1
    OLLAMA_URL: str = "http://localhost:11434"  # URL Ollama local

    # ── Embeddings ───────────────────────────────────────
    # BGE-M3 : meilleur modèle open-source multilingue FR+EN
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Dimension des vecteurs produits par BGE-M3
    # Doit correspondre exactement à la collection Qdrant
    EMBEDDING_DIM: int = 1024

    # Nombre de chunks encodés en même temps
    # 32 = bon équilibre mémoire/vitesse sur CPU
    EMBEDDING_BATCH_SIZE: int = 32

    # ── Qdrant ───────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"

    # Nom de la collection — contient tous les chunks
    QDRANT_COLLECTION: str = "greenrag_docs"

    # ── Chunking ─────────────────────────────────────────
    # Taille maximale d'un chunk en tokens
    # 512 = limite optimale de BGE-M3
    CHUNK_SIZE: int = 512

    # Overlap entre chunks consécutifs
    # 64 tokens = ~2-3 phrases pour garder la continuité
    CHUNK_OVERLAP: int = 64

    # Taille minimale d'un chunk pour éviter les fragments
    # Un chunk de 10 tokens n'a aucune valeur sémantique
    CHUNK_MIN_SIZE: int = 50

    # ── Retrieval ────────────────────────────────────────
    # Nombre de chunks récupérés avant le re-ranking
    TOP_K_RETRIEVAL: int = 20

    # Nombre de chunks gardés après le re-ranking
    # Top 20 → re-ranker → Top 5 envoyés au LLM
    TOP_K_FINAL: int = 5

    # Score minimum de similarité pour accepter un chunk
    # En dessous de 0.65 = chunk probablement non pertinent
    MIN_CONFIDENCE: float = 0.65

    # ── Sources de données ───────────────────────────────
    # Dossier racine des données brutes
    DATA_RAW_PATH: Path = Path("data/raw")

    # Sources actives — correspond aux dossiers dans data/raw/
    SOURCES: list = ["irena", "iea", "owid"]

    # Extensions de fichiers supportées par source
    SUPPORTED_EXTENSIONS: dict = {
        "pdf":  [".pdf"],
        "excel": [".xlsx", ".xls"],
        "csv":  [".csv"],
        "json": [".json"]
    }

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/greenrag.log"

# Instance globale — importée partout dans le projet
# Usage : from config import settings
settings = Settings()