# retriever.py
# MCP Couche Outils — Recherche Qdrant + Calcul LCOE IRENA

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL


class GreenMindRetriever:

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.model  = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ GreenMindRetriever initialisé")

    # ── MCP Server 1 : Recherche sémantique ──────────────
    def search(self, question, top_k=5):
        # Étape 1 : encoder la question en vecteur 1024D
        vector = self.model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        # Étape 2 : chercher les chunks similaires dans Qdrant
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )

        # Étape 3 : retourner format standardisé MCP
        return [
            {
                "text":     r.payload["text"],
                "source":   r.payload["source"],
                "filename": r.payload["filename"],
                "score":    round(r.score, 3)
            }
            for r in results
        ]

    # ── MCP Server 2 : Calcul LCOE IRENA 2024 ────────────
    def calculate_lcoe(self, technology="solar_tunisia"):
        """
        Calcule le LCOE (Levelized Cost of Energy) en $/kWh.
        Formule : LCOE = coûts actualisés / énergie actualisée
        Source   : IRENA Renewable Power Generation Costs 2024
        """

        # Données officielles IRENA 2024 par technologie
        data = {
            "solar_tunisia": {
                "nom":      "Solaire PV Tunisie",
                "capex":    680,    # $/kW  — coût installation
                "opex":     14,     # $/kW/an — coût entretien
                "cf":       0.22,   # 22% — heures soleil Tunisie
                "lifetime": 25,     # ans — durée de vie panneau
                "rate":     0.08,   # 8% — taux actualisation MENA
                "source":   "IRENA Renewable Energy Outlook MENA 2024"
            },
            "wind_onshore": {
                "nom":      "Éolien Terrestre",
                "capex":    1274,
                "opex":     39,
                "cf":       0.30,
                "lifetime": 25,
                "rate":     0.07,
                "source":   "IRENA Renewable Power Generation Costs 2024"
            },
            "solar_global": {
                "nom":      "Solaire PV Mondial",
                "capex":    740,
                "opex":     17,
                "cf":       0.18,
                "lifetime": 25,
                "rate":     0.07,
                "source":   "IRENA Renewable Power Generation Costs 2024"
            }
        }

        # Sélectionner la technologie (solar_tunisia par défaut)
        d = data.get(technology, data["solar_tunisia"])

        # Production annuelle par kW installé (kWh/kW/an)
        # 8760 = nombre d'heures dans une année (365 × 24)
        annual_energy = d["cf"] * 8760

        # Initialisation des accumulateurs
        total_cost   = d["capex"]  # on commence avec l'investissement
        total_energy = 0

        # Actualisation sur toute la durée de vie
        for t in range(1, d["lifetime"] + 1):
            # Plus on est loin dans le futur, moins ça vaut aujourd'hui
            discount      = (1 + d["rate"]) ** t
            total_cost   += d["opex"] / discount
            total_energy += annual_energy / discount

        # LCOE = coût total actualisé / énergie totale actualisée
        lcoe = total_cost / total_energy

        return {
            "technologie": d["nom"],
            "lcoe":        round(lcoe, 4),
            "lcoe_cents":  round(lcoe * 100, 2),
            "unite":       "$/kWh",
            "source":      d["source"]
        }