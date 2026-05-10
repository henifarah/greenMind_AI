from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loguru import logger
from tools.base_tool import BaseTool
from config import settings


class EnergyRetriever(BaseTool):
    """
    MCP Server 1 — energy-retriever-server

    Combine deux méthodes de recherche :
    1. Vectorielle (cosine BGE-M3) → comprend le sens
    2. BM25 (mots-clés)            → trouve les termes exacts
    3. Fusion RRF                  → combine les deux
    """

    def __init__(self):
        # Connexion Qdrant
        self.client = QdrantClient(url=settings.QDRANT_URL)

        # Modèle d'embedding — même que l'ingestion
        # IMPORTANT : même modèle = même espace vectoriel
        logger.info("Chargement BGE-M3 pour EnergyRetriever...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.success("✅ EnergyRetriever initialisé")

        # Cache BM25 — construit à la première recherche
        self._bm25 = None
        self._bm25_chunks = None

    # ── Contrat BaseTool ──────────────────────────────────
    def get_name(self) -> str:
        return "energy-retriever-server"

    def get_description(self) -> str:
        return (
            "Recherche sémantique hybride dans la base de "
            "connaissances GreenMind sur les énergies renouvelables. "
            "Combine la recherche vectorielle BGE-M3 et BM25 "
            "pour des résultats précis et pertinents."
        )

    # ── Outil 1 : Recherche vectorielle pure ─────────────
    def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        source_filter: str = None
    ) -> list[dict]:
        """
        Recherche vectorielle pure dans Qdrant.

        Encode la question avec BGE-M3 et cherche
        les chunks dont le vecteur est le plus proche.

        source_filter : filtrer par source ("irena", "iea", "owid")
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Encoder la question
        query_vector = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # Construire le filtre optionnel par source
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter)
                )]
            )

        # Rechercher dans Qdrant
        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        # Formater les résultats
        return [
            {
                "text":     r.payload.get("text", ""),
                "source":   r.payload.get("source", ""),
                "filename": r.payload.get("filename", ""),
                "score":    round(r.score, 4),
                "method":   "vector",
                "chunk_id": r.payload.get("chunk_id", 0),
            }
            for r in results
        ]

    # ── Construction index BM25 ───────────────────────────
    def _build_bm25_index(self) -> None:
        """
        Construit l'index BM25 depuis tous les chunks Qdrant.

        BM25 a besoin de tous les documents pour calculer
        les fréquences de mots (TF-IDF amélioré).

        On le construit UNE SEULE FOIS et on le met en cache.
        """
        if self._bm25 is not None:
            return  # déjà construit

        logger.info("Construction index BM25...")

        # Récupérer tous les chunks depuis Qdrant
        all_chunks = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False  # pas besoin des vecteurs pour BM25
            )
            all_chunks.extend(results)
            if offset is None:
                break

        # Tokeniser les textes pour BM25
        # BM25 travaille sur des listes de mots
        tokenized = [
            r.payload.get("text", "").lower().split()
            for r in all_chunks
        ]

        self._bm25 = BM25Okapi(tokenized)
        self._bm25_chunks = all_chunks

        logger.success(f"✅ Index BM25 construit : {len(all_chunks)} documents")

    # ── Outil 2 : Recherche hybride ───────────────────────
    def hybrid_search(
        self,
        query: str,
        top_k: int = None
    ) -> list[dict]:
        """
        Recherche hybride : BM25 + vectorielle → fusion RRF.

        RRF (Reciprocal Rank Fusion) :
        Pour chaque chunk, calcule un score combiné :
        score_RRF = 1/(rang_vector + k) + 1/(rang_bm25 + k)

        Un chunk bien classé dans les DEUX méthodes
        obtient un score RRF élevé → remonte en tête.

        k=60 est la constante standard de RRF.
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # ── Résultats vectoriels ──────────────────────────
        vector_results = self.retrieve_context(query, top_k=top_k)

        # Créer un dictionnaire rang → chunk pour vector
        vector_ranks = {
            r["filename"] + str(r["chunk_id"]): rank
            for rank, r in enumerate(vector_results)
        }

        # ── Résultats BM25 ────────────────────────────────
        self._build_bm25_index()

        query_tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(query_tokens)

        # Top k résultats BM25
        import numpy as np
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

        bm25_results = []
        for rank, idx in enumerate(top_bm25_indices):
            chunk = self._bm25_chunks[idx]
            bm25_results.append({
                "text":     chunk.payload.get("text", ""),
                "source":   chunk.payload.get("source", ""),
                "filename": chunk.payload.get("filename", ""),
                "score":    float(bm25_scores[idx]),
                "method":   "bm25",
                "chunk_id": chunk.payload.get("chunk_id", 0),
                "bm25_rank": rank
            })

        bm25_ranks = {
            r["filename"] + str(r["chunk_id"]): rank
            for rank, r in enumerate(bm25_results)
        }

        # ── Fusion RRF ────────────────────────────────────
        k = 60  # constante standard RRF
        all_chunks_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        rrf_scores = {}
        for chunk_id in all_chunks_ids:
            v_rank = vector_ranks.get(chunk_id, top_k)
            b_rank = bm25_ranks.get(chunk_id, top_k)
            rrf_scores[chunk_id] = 1/(v_rank + k) + 1/(b_rank + k)

        # Trier par score RRF décroissant
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:top_k]

        # Assembler les résultats finaux
        all_results = {
            r["filename"] + str(r["chunk_id"]): r
            for r in vector_results + bm25_results
        }

        final_results = []
        for chunk_id in sorted_ids:
            if chunk_id in all_results:
                result = all_results[chunk_id].copy()
                result["score"] = round(rrf_scores[chunk_id], 6)
                result["method"] = "hybrid_rrf"
                final_results.append(result)

        logger.info(
            f"Hybrid search '{query[:50]}...' "
            f"→ {len(final_results)} résultats"
        )

        return final_results

    # ── Méthode search() du contrat BaseTool ─────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Point d'entrée standard MCP.
        Utilise la recherche hybride par défaut.
        """
        return self.hybrid_search(query, top_k=top_k)


# ── Instance globale ──────────────────────────────────────
# Créée une seule fois — BGE-M3 chargé une seule fois
energy_retriever = EnergyRetriever()