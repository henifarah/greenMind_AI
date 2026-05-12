from loguru import logger
from config import settings


class ContextBuilder:
    """
    MCP Host — Couche Contexte.

    Assemble tout ce que le LLM a besoin de savoir
    pour générer une réponse précise et sourcée.
    """

    def __init__(self, max_tokens: int = 3000):
        # Limite de tokens pour le contexte
        # GPT-4o-mini accepte jusqu'à 128k tokens
        # mais on limite à 3000 pour économiser les coûts
        self.max_tokens = max_tokens

    def build_system_prompt(self) -> str:
        """
        Prompt système — définit le comportement du LLM.
        C'est la "personnalité" de GreenMind.
        """
        return """Tu es GreenMind, un assistant expert en énergies renouvelables.

Tes règles absolues :
1. Tu réponds UNIQUEMENT avec les informations du contexte fourni
2. Tu cites TOUJOURS tes sources (fichier + page si disponible)
3. Tu donnes des chiffres précis avec leurs unités ($/kWh, MW, GWh...)
4. Si l'information n'est pas dans le contexte → tu le dis clairement
5. Tu réponds en français sauf si la question est en anglais
6. Tu structure ta réponse : chiffre clé → explication → source

Tu NE dois PAS :
- Inventer des chiffres non présents dans le contexte
- Répondre sans citer la source
- Mélanger des informations de différentes années sans le préciser"""

    def build_context_from_chunks(
        self,
        chunks: list[dict],
        max_chars: int = 4000
    ) -> str:
        """
        Convertit les chunks en texte de contexte structuré.

        Trie par score de pertinence et limite la taille
        pour ne pas dépasser la fenêtre de contexte du LLM.
        """
        if not chunks:
            return "Aucun document pertinent trouvé."

        # Trier par score décroissant
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        context_parts = []
        total_chars = 0

        for chunk in sorted_chunks:
            text     = chunk.get("text", "")
            source   = chunk.get("source", "").upper()
            filename = chunk.get("filename", "")
            score    = chunk.get("score", 0)

            # Formater le chunk avec sa source
            chunk_text = (
                f"[Source: {source} — {filename}]\n"
                f"{text}\n"
            )

            # Vérifier qu'on ne dépasse pas la limite
            if total_chars + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        logger.debug(
            f"Contexte assemblé : {len(context_parts)} chunks, "
            f"{total_chars} caractères"
        )

        return "\n---\n".join(context_parts)

    def build_lcoe_context(self, lcoe_result: dict) -> str:
        """
        Formate le résultat LCOE en texte pour le LLM.
        """
        if not lcoe_result or "error" in lcoe_result:
            return ""

        if isinstance(lcoe_result, list):
            lcoe_result = lcoe_result[0] if lcoe_result else {}

        lines = ["\n[Calcul LCOE GreenMind]"]
        for key, value in lcoe_result.items():
            if key not in ["technology_key", "note"]:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def build_prompt(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict] = None,
        lcoe_result: dict = None
    ) -> list[dict]:
        """
        Construit le prompt complet pour le LLM.

        Format LangChain/OpenAI :
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},  # historique
            {"role": "assistant", "content": "..."},  # historique
            {"role": "user",      "content": "..."}   # question actuelle
        ]

        Pourquoi ce format ?
        → OpenAI et LangChain attendent ce format standard
        → "system" = instructions permanentes
        → "user/assistant" = conversation historique
        → Dernier "user" = question actuelle avec contexte
        """
        messages = []

        # 1. Prompt système
        messages.append({
            "role": "system",
            "content": self.build_system_prompt()
        })

        # 2. Historique de conversation (si disponible)
        if history:
            for turn in history[-4:]:  # garder les 4 derniers échanges
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })

        # 3. Contexte documentaire
        doc_context = self.build_context_from_chunks(chunks)

        # 4. Contexte LCOE (si disponible)
        lcoe_context = ""
        if lcoe_result:
            lcoe_context = self.build_lcoe_context(lcoe_result)

        # 5. Message utilisateur avec contexte
        user_message = f"""CONTEXTE DOCUMENTAIRE :
{doc_context}
{lcoe_context}

QUESTION : {question}

Réponds de façon précise et sourcée en utilisant uniquement 
les informations du contexte ci-dessus."""

        messages.append({
            "role": "user",
            "content": user_message
        })

        logger.debug(f"Prompt construit : {len(messages)} messages")
        return messages


# Instance globale
context_builder = ContextBuilder()