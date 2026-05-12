from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm
from config import settings


# ── Chargement du modèle ──────────────────────────────────
# On charge BGE-M3 une seule fois au démarrage du module.
# Pourquoi une variable globale ?
# → Charger BGE-M3 prend ~30 secondes la première fois.
# → Si on le rechargeait à chaque appel, ce serait 30s × 496 batches.
# → Une fois chargé en mémoire, les appels suivants sont instantanés.
_model = None

def get_model() -> SentenceTransformer:
    """
    Retourne le modèle BGE-M3, le charge si nécessaire.
    Pattern "lazy loading" — on ne charge que quand on en a besoin.
    """
    global _model
    if _model is None:
        logger.info(f"Chargement de BGE-M3 ({settings.EMBEDDING_MODEL})...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.success("✅ BGE-M3 chargé en mémoire")
    return _model


# ── Encodage d'un batch ───────────────────────────────────
def encode_batch(texts: list[str]) -> list[list[float]]:
    """
    Encode un batch de textes en vecteurs 1024D.

    Reçoit  : liste de textes (max 32 recommandé)
    Retourne: liste de vecteurs (chacun = liste de 1024 floats)

    Pourquoi normalize_embeddings=True ?
    → Normalise les vecteurs entre -1 et +1
    → Rend la similarité cosine plus précise
    → Standard pour la recherche sémantique
    """
    model = get_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # normalisation pour cosine similarity
        show_progress_bar=False,    # on gère notre propre barre de progression
    )

    # Convertir numpy array → liste Python (compatible JSON et Qdrant)
    return vectors.tolist()


# ── Encodage de tous les chunks ───────────────────────────
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Encode tous les chunks et ajoute leur vecteur.

    Reçoit  : liste de chunks de chunker.py
    Retourne: mêmes chunks + champ "vector" ajouté

    Chaque chunk devient :
    {
        "text": "Le coût LCOE solaire en Tunisie...",
        "source": "irena",
        "filename": "IRENA_costs_2024.pdf",
        "chunk_id": 3,
        ...
        "vector": [0.23, -0.87, 0.45, ...]  ← NOUVEAU (1024 valeurs)
    }
    """
    total = len(chunks)
    logger.info(f"Encodage de {total} chunks avec BGE-M3...")
    logger.info(f"Batch size : {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"Batches nécessaires : {total // settings.EMBEDDING_BATCH_SIZE + 1}")

    embedded_chunks = []

    # Barre de progression pour suivre l'avancement
    with tqdm(
        total=total,
        desc="BGE-M3 encoding",
        unit="chunks"
    ) as pbar:

        # Traiter les chunks par batch
        for i in range(0, total, settings.EMBEDDING_BATCH_SIZE):

            # Extraire le batch courant
            batch = chunks[i : i + settings.EMBEDDING_BATCH_SIZE]

            # Extraire uniquement les textes pour l'encodage
            # BGE-M3 n'a besoin que du texte, pas des métadonnées
            texts = [chunk["text"] for chunk in batch]

            # Encoder le batch → liste de vecteurs 1024D
            try:
                vectors = encode_batch(texts)

                # Ajouter le vecteur à chaque chunk du batch
                for chunk, vector in zip(batch, vectors):
                    chunk_with_vector = {
                        **chunk,           # toutes les métadonnées existantes
                        "vector": vector   # + le vecteur BGE-M3
                    }
                    embedded_chunks.append(chunk_with_vector)

            except Exception as e:
                logger.error(f"❌ Erreur encodage batch {i//settings.EMBEDDING_BATCH_SIZE} : {e}")
                # En cas d'erreur sur un batch → on continue avec le suivant
                # Les chunks de ce batch sont perdus mais le pipeline continue
                continue

            # Mettre à jour la barre de progression
            pbar.update(len(batch))

    # Statistiques finales
    success_rate = len(embedded_chunks) / total * 100
    logger.success(
        f"✅ Encodage terminé : {len(embedded_chunks)}/{total} chunks "
        f"({success_rate:.1f}% succès)"
    )

    return embedded_chunks


# ── Vérification d'un vecteur ─────────────────────────────
def verify_embedding(chunk: dict) -> bool:
    """
    Vérifie qu'un chunk a bien un vecteur valide.
    Utilisé par indexer.py avant de stocker dans Qdrant.
    """
    vector = chunk.get("vector")

    if vector is None:
        return False

    if len(vector) != settings.EMBEDDING_DIM:
        logger.warning(
            f"Dimension incorrecte : {len(vector)} "
            f"(attendu {settings.EMBEDDING_DIM})"
        )
        return False

    return True
def release_model():
    """Libère BGE-M3 de la RAM."""
    global _model
    _model = None
    import gc
    gc.collect()
    logger.info("✅ BGE-M3 libéré de la RAM")