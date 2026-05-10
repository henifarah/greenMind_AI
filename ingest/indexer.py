from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from loguru import logger
from tqdm import tqdm
import uuid
from config import settings


# ── Connexion Qdrant ──────────────────────────────────────
def get_client() -> QdrantClient:
    """
    Retourne un client connecté à Qdrant.
    Qdrant tourne dans Docker sur localhost:6333.
    """
    return QdrantClient(url=settings.QDRANT_URL)


# ── Créer la collection ───────────────────────────────────
def create_collection(client: QdrantClient) -> None:
    """
    Crée la collection Qdrant si elle n'existe pas encore.

    Une collection Qdrant = une table dans une base SQL.
    Elle stocke des vecteurs + métadonnées (payload).

    Paramètres importants :
    - size=1024     : dimension des vecteurs BGE-M3
    - distance=COSINE : mesure de similarité entre vecteurs
      COSINE = mesure l'angle entre vecteurs
      → 1.0 = identiques | 0.0 = sans rapport | -1.0 = opposés
    """
    # Vérifier si la collection existe déjà
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION in existing:
        count = client.count(settings.QDRANT_COLLECTION).count
        logger.info(
            f"Collection '{settings.QDRANT_COLLECTION}' existe déjà "
            f"({count} points)"
        )
        return

    # Créer la collection
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIM,    # 1024 dimensions BGE-M3
            distance=Distance.COSINE,       # similarité cosine
        )
    )
    logger.success(f"✅ Collection '{settings.QDRANT_COLLECTION}' créée")


# ── Indexer les chunks ────────────────────────────────────
def index_chunks(chunks: list[dict]) -> int:
    """
    Stocke les chunks encodés dans Qdrant.

    Chaque chunk devient un "Point" Qdrant :
    - id      : identifiant unique (UUID)
    - vector  : vecteur BGE-M3 1024D
    - payload : toutes les métadonnées (texte, source, fichier...)

    Retourne le nombre de chunks indexés avec succès.
    """
    client = get_client()
    create_collection(client)

    total = len(chunks)
    indexed = 0
    batch_size = 100  # Qdrant accepte jusqu'à 100 points par batch

    logger.info(f"Indexation de {total} chunks dans Qdrant...")

    with tqdm(total=total, desc="Qdrant indexing", unit="chunks") as pbar:

        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]

            # Construire les points Qdrant
            points = []
            for chunk in batch:

                # Vérifier que le chunk a un vecteur valide
                vector = chunk.get("vector")
                if not vector or len(vector) != settings.EMBEDDING_DIM:
                    logger.warning(f"Chunk sans vecteur ignoré : {chunk.get('filename')}")
                    continue

                # Le payload contient tout sauf le vecteur
                # C'est ce qu'on récupère lors d'une recherche
                payload = {
                    "text":        chunk.get("text", ""),
                    "source":      chunk.get("source", "unknown"),
                    "filename":    chunk.get("filename", "unknown"),
                    "file_type":   chunk.get("file_type", "unknown"),
                    "chunk_id":    chunk.get("chunk_id", 0),
                    "chunk_total": chunk.get("chunk_total", 0),
                    "chunk_size":  chunk.get("chunk_size", 0),
                }

                # Ajouter les métadonnées optionnelles si présentes
                for key in ["pages", "sheets", "rows", "columns"]:
                    if key in chunk:
                        payload[key] = chunk[key]

                # Créer le point Qdrant
                point = PointStruct(
                    id=str(uuid.uuid4()),  # ID unique pour chaque chunk
                    vector=vector,          # vecteur BGE-M3
                    payload=payload         # métadonnées
                )
                points.append(point)

            # Envoyer le batch à Qdrant
            if points:
                try:
                    client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points
                    )
                    indexed += len(points)
                except Exception as e:
                    logger.error(f"❌ Erreur indexation batch {i//batch_size} : {e}")

            pbar.update(len(batch))

    logger.success(f"✅ Indexation terminée : {indexed}/{total} chunks stockés")
    return indexed


# ── Vérification ─────────────────────────────────────────
def get_collection_stats() -> dict:
    """
    Retourne les statistiques de la collection Qdrant.
    Utilisé pour vérifier que tout est bien indexé.
    """
    client = get_client()

    try:
        count = client.count(settings.QDRANT_COLLECTION).count
        info = client.get_collection(settings.QDRANT_COLLECTION)

        stats = {
            "total_points": count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }

        logger.info(f"Collection '{settings.QDRANT_COLLECTION}' :")
        logger.info(f"  Points total  : {stats['total_points']}")
        logger.info(f"  Taille vecteur: {stats['vector_size']}D")
        logger.info(f"  Distance      : {stats['distance']}")

        return stats

    except Exception as e:
        logger.error(f"❌ Erreur stats : {e}")
        return {}


# ── Recherche test ────────────────────────────────────────
def search_test(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Fait une recherche test dans Qdrant.
    Utilisé pour vérifier que la recherche fonctionne.
    """
    client = get_client()

    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True  # retourner les métadonnées avec les résultats
    )

    return [
        {
            "score":    r.score,
            "text":     r.payload.get("text", "")[:200],
            "source":   r.payload.get("source", ""),
            "filename": r.payload.get("filename", ""),
        }
        for r in results
    ]