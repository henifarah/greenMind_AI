from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Classe abstraite — définit le contrat MCP.
    Tout outil qui hérite de BaseTool DOIT implémenter
    les méthodes search() et get_name().

    Pourquoi une classe abstraite ?
    → Garantit que tous les outils ont la même interface
    → Le modèle peut appeler n'importe quel outil
      de la même façon sans savoir comment il fonctionne
    → C'est le principe MCP : séparation claire
    """

    @abstractmethod
    def get_name(self) -> str:
        """Retourne le nom de l'outil."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Retourne la description de l'outil."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Recherche les documents pertinents pour une requête.

        Reçoit  : query (question en langage naturel)
        Retourne: liste de dicts avec au minimum :
            - text     : texte du chunk
            - source   : source du document
            - filename : nom du fichier
            - score    : score de pertinence (0-1)
        """
        pass