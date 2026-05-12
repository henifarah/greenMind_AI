# model/rag_model.py
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from loguru import logger
from config import settings
from tools.energy_retriever import energy_retriever
from tools.lcoe_calculator import lcoe_calculator
from context.builder import context_builder
import gc


class RAGModel:
    """
    MCP Client — Couche Modèle.
    Orchestre les outils MCP et génère la réponse via Mistral/Ollama.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            base_url=settings.OLLAMA_URL,
        )
        self.history = []
        logger.success(f"✅ RAGModel initialisé avec {settings.LLM_MODEL}")

    def _needs_lcoe(self, question: str) -> bool:
        """Détecte si la question nécessite un calcul LCOE."""
        keywords = [
            "lcoe", "coût", "cost", "prix", "price",
            "kwh", "$/kwh", "tarif", "combien", "how much"
        ]
        return any(kw in question.lower() for kw in keywords)

    def answer(self, question: str) -> dict:
        logger.info(f"Question : {question[:80]}...")

        # ── Étape 1 : Recherche hybride ───────────────────
        logger.info("Étape 1 : Recherche hybride...")
        chunks = energy_retriever.hybrid_search(
            question,
            top_k=settings.TOP_K_RETRIEVAL
        )
        logger.info(f"→ {len(chunks)} chunks trouvés")

        # ── Étape 2 : Calcul LCOE si nécessaire ──────────
        lcoe_result = None
        if self._needs_lcoe(question):
            logger.info("Étape 2 : Calcul LCOE...")
            lcoe_results = lcoe_calculator.search(question)
            lcoe_result = lcoe_results[0] if lcoe_results else None

        # ── Libérer BGE-M3 avant Mistral ─────────────────
        logger.info("Libération BGE-M3 de la RAM...")
        from ingest.embedder import release_model
        release_model()
        gc.collect()
        logger.info("✅ RAM libérée")

        # ── Étape 3 : Construction du contexte ────────────
        logger.info("Étape 3 : Construction du contexte...")
        messages = context_builder.build_prompt(
            question=question,
            chunks=chunks[:settings.TOP_K_FINAL],
            history=self.history,
            lcoe_result=lcoe_result
        )

        # ── Étape 4 : Génération Mistral ──────────────────
        logger.info("Étape 4 : Génération Mistral...")
        try:
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            response = self.llm.invoke(lc_messages)
            answer_text = response.content

        except Exception as e:
            logger.error(f"Erreur Mistral : {e}")
            answer_text = f"Erreur : {e}"

        # ── Étape 5 : Historique ──────────────────────────
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer_text})

        sources = list(set([
            f"{c['source'].upper()} — {c['filename']}"
            for c in chunks[:settings.TOP_K_FINAL]
        ]))

        logger.success("✅ Réponse générée")

        return {
            "answer":  answer_text,
            "sources": sources,
            "chunks":  chunks[:settings.TOP_K_FINAL],
            "lcoe":    lcoe_result
        }


# Instance globale
rag_model = RAGModel()