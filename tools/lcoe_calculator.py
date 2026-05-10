from loguru import logger
from tools.base_tool import BaseTool


# ── Données de référence IRENA 2024 ──────────────────────
# Source : IRENA Renewable Power Generation Costs 2024
# Ces valeurs sont les médianes mondiales officielles

TECHNOLOGY_DATA = {
    "solar_pv": {
        "name": "Solaire Photovoltaïque",
        "capex": 740,        # $/kW — médiane mondiale IRENA 2024
        "opex": 17,          # $/kW/an
        "capacity_factor": 0.18,  # 18% — médiane mondiale
        "lifetime": 25,      # années
        "discount_rate": 0.07,    # 7% standard
        "lcoe_irena": 0.049, # $/kWh — valeur IRENA 2024
        "unit": "$/kWh",
        "source": "IRENA Renewable Power Generation Costs 2024"
    },
    "wind_onshore": {
        "name": "Éolien Terrestre",
        "capex": 1274,
        "opex": 39,
        "capacity_factor": 0.30,
        "lifetime": 25,
        "discount_rate": 0.07,
        "lcoe_irena": 0.033,
        "unit": "$/kWh",
        "source": "IRENA Renewable Power Generation Costs 2024"
    },
    "wind_offshore": {
        "name": "Éolien Offshore",
        "capex": 3461,
        "opex": 112,
        "capacity_factor": 0.40,
        "lifetime": 25,
        "discount_rate": 0.07,
        "lcoe_irena": 0.081,
        "unit": "$/kWh",
        "source": "IRENA Renewable Power Generation Costs 2024"
    },
    "hydro": {
        "name": "Hydraulique",
        "capex": 1704,
        "opex": 45,
        "capacity_factor": 0.44,
        "lifetime": 40,
        "discount_rate": 0.07,
        "lcoe_irena": 0.026,
        "unit": "$/kWh",
        "source": "IRENA Renewable Power Generation Costs 2024"
    },
    "geothermal": {
        "name": "Géothermique",
        "capex": 3916,
        "opex": 154,
        "capacity_factor": 0.80,
        "lifetime": 25,
        "discount_rate": 0.07,
        "lcoe_irena": 0.068,
        "unit": "$/kWh",
        "source": "IRENA Renewable Power Generation Costs 2024"
    },
    "solar_pv_tunisia": {
        "name": "Solaire PV Tunisie",
        "capex": 680,        # $/kW — données MENA IRENA
        "opex": 14,
        "capacity_factor": 0.22,  # Tunisie = fort ensoleillement
        "lifetime": 25,
        "discount_rate": 0.08,    # taux MENA légèrement plus élevé
        "lcoe_irena": 0.041,      # $/kWh — IRENA MENA 2024
        "unit": "$/kWh",
        "source": "IRENA Renewable Energy Outlook MENA 2024"
    }
}


class LCOECalculator(BaseTool):
    """
    MCP Server 2 — lcoe-calculator-server

    Calcule le coût actualisé de l'énergie (LCOE)
    pour différentes technologies renouvelables.
    """

    def get_name(self) -> str:
        return "lcoe-calculator-server"

    def get_description(self) -> str:
        return (
            "Calcule le coût actualisé de l'énergie (LCOE) "
            "pour les technologies renouvelables. "
            "Compare les coûts entre technologies et régions. "
            "Source : IRENA Renewable Power Generation Costs 2024."
        )

    # ── Outil 1 : Calculer le LCOE ────────────────────────
    def calculate_lcoe(
        self,
        technology: str,
        capex: float = None,
        opex: float = None,
        capacity_factor: float = None,
        lifetime: int = None,
        discount_rate: float = None
    ) -> dict:
        """
        Calcule le LCOE d'une technologie.

        Si les paramètres sont fournis → calcul personnalisé
        Sinon → utilise les valeurs IRENA 2024 par défaut

        Formule LCOE :
        LCOE = (CAPEX + Σ OPEX/(1+r)^t) / Σ E/(1+r)^t

        Où E = CAPEX × capacity_factor × 8760h/an
        """
        # Vérifier que la technologie existe
        if technology not in TECHNOLOGY_DATA:
            available = list(TECHNOLOGY_DATA.keys())
            return {
                "error": f"Technologie inconnue : {technology}",
                "available": available
            }

        # Récupérer les données de référence
        ref = TECHNOLOGY_DATA[technology]

        # Utiliser les paramètres fournis ou les valeurs par défaut
        capex           = capex           or ref["capex"]
        opex            = opex            or ref["opex"]
        capacity_factor = capacity_factor or ref["capacity_factor"]
        lifetime        = lifetime        or ref["lifetime"]
        discount_rate   = discount_rate   or ref["discount_rate"]

        # ── Calcul LCOE ───────────────────────────────────
        # Production annuelle par kW installé (kWh/kW/an)
        annual_energy = capacity_factor * 8760  # 8760h = 1 an

        # Actualisation sur la durée de vie
        total_cost   = capex   # coût initial
        total_energy = 0

        for year in range(1, lifetime + 1):
            discount = (1 + discount_rate) ** year
            total_cost   += opex / discount
            total_energy += annual_energy / discount

        # LCOE en $/kWh
        lcoe_calculated = total_cost / total_energy

        result = {
            "technology":       ref["name"],
            "technology_key":   technology,

            # Paramètres utilisés
            "capex_per_kw":        f"${capex}/kW",
            "opex_per_kw_year":    f"${opex}/kW/an",
            "capacity_factor":     f"{capacity_factor*100:.1f}%",
            "lifetime":            f"{lifetime} ans",
            "discount_rate":       f"{discount_rate*100:.1f}%",

            # Résultats
            "lcoe_calculated":     f"${lcoe_calculated:.4f}/kWh",
            "lcoe_irena_2024":     f"${ref['lcoe_irena']:.3f}/kWh",
            "annual_production":   f"{annual_energy:.0f} kWh/kW/an",

            # Source
            "source": ref["source"],
            "note": "Calcul basé sur la formule LCOE standard IEA/IRENA"
        }

        logger.info(
            f"LCOE calculé : {ref['name']} → "
            f"${lcoe_calculated:.4f}/kWh "
            f"(IRENA : ${ref['lcoe_irena']:.3f}/kWh)"
        )

        return result

    # ── Outil 2 : Comparer les technologies ──────────────
    def compare_technologies(
        self,
        technologies: list[str] = None,
        region: str = "global"
    ) -> dict:
        """
        Compare le LCOE de plusieurs technologies.
        Retourne un classement du moins cher au plus cher.
        """
        if technologies is None:
            technologies = list(TECHNOLOGY_DATA.keys())

        results = []
        for tech in technologies:
            if tech in TECHNOLOGY_DATA:
                ref = TECHNOLOGY_DATA[tech]
                results.append({
                    "technology":  ref["name"],
                    "key":         tech,
                    "lcoe":        ref["lcoe_irena"],
                    "lcoe_str":    f"${ref['lcoe_irena']:.3f}/kWh",
                    "capex":       f"${ref['capex']}/kW",
                    "source":      ref["source"]
                })

        # Trier du moins cher au plus cher
        results.sort(key=lambda x: x["lcoe"])

        # Ajouter le rang
        for i, r in enumerate(results, 1):
            r["rank"] = i

        comparison = {
            "region":      region,
            "year":        "2024",
            "source":      "IRENA Renewable Power Generation Costs 2024",
            "ranking":     results,
            "cheapest":    results[0]["technology"] if results else None,
            "most_expensive": results[-1]["technology"] if results else None,
        }

        logger.info(
            f"Comparaison {len(results)} technologies → "
            f"moins cher : {comparison['cheapest']}"
        )

        return comparison

    # ── Méthode search() du contrat BaseTool ─────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Détecte la technologie dans la question
        et retourne son LCOE.
        """
        query_lower = query.lower()

        # Détecter la technologie mentionnée
        tech_map = {
            "solaire": "solar_pv",
            "solar":   "solar_pv",
            "pv":      "solar_pv",
            "éolien":  "wind_onshore",
            "wind":    "wind_onshore",
            "offshore":"wind_offshore",
            "hydro":   "hydro",
            "géotherm":"geothermal",
            "tunisie": "solar_pv_tunisia",
            "tunisia": "solar_pv_tunisia",
            "mena":    "solar_pv_tunisia",
        }

        detected_tech = "solar_pv"  # défaut
        for keyword, tech in tech_map.items():
            if keyword in query_lower:
                detected_tech = tech
                break

        result = self.calculate_lcoe(detected_tech)
        return [result]


# ── Instance globale ──────────────────────────────────────
lcoe_calculator = LCOECalculator()