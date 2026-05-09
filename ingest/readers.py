from pathlib import Path
from typing import Optional
import pandas as pd
import json
from pypdf import PdfReader
from loguru import logger


# ── Format de sortie standard ─────────────────────────────
# Chaque reader retourne toujours ce dictionnaire.
# C'est le "contrat" entre readers.py et chunker.py.
def make_document(
    text: str,
    source: str,
    filename: str,
    file_type: str,
    extra: dict = {}
) -> dict:
    """
    Crée un document standardisé.
    Peu importe le type de fichier en entrée,
    la sortie est toujours identique — principe MCP.
    """
    return {
        "text": text.strip(),        # texte nettoyé
        "source": source,            # ex: "irena", "iea", "owid"
        "filename": filename,        # nom du fichier
        "file_type": file_type,      # "pdf", "excel", "csv", "json"
        **extra                      # métadonnées supplémentaires
    }


# ── Lecteur PDF ───────────────────────────────────────────
def read_pdf(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un PDF page par page et extrait le texte.

    Pourquoi page par page ?
    → Permet de garder le numéro de page dans les métadonnées
    → Si le PDF fait 200 pages, on sait exactement où est
      chaque information pour citer la source précisément.
    """
    try:
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)

        # Extraire le texte de chaque page
        all_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            # Ignorer les pages vides (couvertures, pages blanches)
            if page_text and len(page_text.strip()) > 50:
                # Ajouter un marqueur de page pour la traçabilité
                all_text.append(f"[Page {page_num}]\n{page_text}")

        # Joindre toutes les pages avec séparateur
        full_text = "\n\n".join(all_text)

        if not full_text.strip():
            logger.warning(f"PDF vide ou non lisible : {file_path.name}")
            return None

        logger.success(f"✅ PDF lu : {file_path.name} ({total_pages} pages)")

        return make_document(
            text=full_text,
            source=source,
            filename=file_path.name,
            file_type="pdf",
            extra={"pages": total_pages}
        )

    except Exception as e:
        logger.error(f"❌ Erreur PDF {file_path.name} : {e}")
        return None


# ── Lecteur Excel ─────────────────────────────────────────
def read_excel(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un fichier Excel et convertit les tableaux en texte.

    Pourquoi convertir en texte ?
    → BGE-M3 encode du texte, pas des tableaux Excel.
    → On convertit chaque feuille en texte structuré lisible.

    Exemple de conversion :
    | Pays    | Capacité | Année |     →    "Pays: Tunisia,
    | Tunisia | 542 MW   | 2023  |           Capacité: 542 MW,
                                             Année: 2023"
    """
    try:
        # Lire toutes les feuilles du fichier Excel
        excel_file = pd.ExcelFile(str(file_path))
        all_text = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Ignorer les feuilles vides
            if df.empty:
                continue

            # Nettoyer : supprimer colonnes et lignes entièrement vides
            df = df.dropna(how="all").dropna(axis=1, how="all")

            if df.empty:
                continue

            # Convertir le DataFrame en texte structuré
            # Chaque ligne du tableau devient une phrase
            sheet_text = f"[Feuille: {sheet_name}]\n"
            sheet_text += df.to_string(index=False, na_rep="N/A")
            all_text.append(sheet_text)

        if not all_text:
            logger.warning(f"Excel vide : {file_path.name}")
            return None

        full_text = "\n\n".join(all_text)
        sheets_count = len(all_text)

        logger.success(f"✅ Excel lu : {file_path.name} ({sheets_count} feuilles)")

        return make_document(
            text=full_text,
            source=source,
            filename=file_path.name,
            file_type="excel",
            extra={"sheets": sheets_count}
        )

    except Exception as e:
        logger.error(f"❌ Erreur Excel {file_path.name} : {e}")
        return None


# ── Lecteur CSV ───────────────────────────────────────────
def read_csv(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un CSV et le convertit en texte structuré.

    Pour les CSV OWID qui ont des colonnes comme :
    Entity, Year, Solar capacity (GW)
    Tunisia, 2023, 0.542

    On produit un texte comme :
    "Entity: Tunisia, Year: 2023, Solar capacity: 0.542 GW"

    Ce format est bien plus lisible par BGE-M3
    qu'un CSV brut avec des virgules.
    """
    try:
        # Essayer différents encodages (les CSV peuvent varier)
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"❌ Encodage impossible : {file_path.name}")
            return None

        # Ignorer les CSV vides
        if df.empty:
            logger.warning(f"CSV vide : {file_path.name}")
            return None

        # Nettoyer
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Convertir en texte lisible
        # head(500) → on limite à 500 lignes pour éviter les CSV géants
        rows_count = len(df)
        df_sample = df.head(500)

        # Format : "Col1: val1, Col2: val2, ..."
        lines = []
        for _, row in df_sample.iterrows():
            line = ", ".join([
                f"{col}: {val}"
                for col, val in row.items()
                if pd.notna(val)
            ])
            if line:
                lines.append(line)

        full_text = f"[Dataset: {file_path.stem}]\n" + "\n".join(lines)

        logger.success(f"✅ CSV lu : {file_path.name} ({rows_count} lignes)")

        return make_document(
            text=full_text,
            source=source,
            filename=file_path.name,
            file_type="csv",
            extra={"rows": rows_count}
        )

    except Exception as e:
        logger.error(f"❌ Erreur CSV {file_path.name} : {e}")
        return None


# ── Lecteur JSON ──────────────────────────────────────────
def read_json(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un fichier JSON (métadonnées OWID).

    Les fichiers .metadata.json OWID contiennent des infos
    précieuses : description du dataset, source originale,
    unités de mesure, couverture temporelle.

    Ces métadonnées enrichissent le RAG — quand on répond
    à une question, on peut citer la source exacte et
    les unités utilisées.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convertir le JSON en texte lisible
        # On extrait les champs les plus utiles pour le RAG
        text_parts = []

        # Parcourir récursivement le JSON
        def extract_text(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_text(value, f"{prefix}{key}: ")
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, prefix)
            elif obj is not None:
                text_parts.append(f"{prefix}{obj}")

        extract_text(data)
        full_text = "\n".join(text_parts)

        if len(full_text) < 50:
            return None

        logger.success(f"✅ JSON lu : {file_path.name}")

        return make_document(
            text=full_text,
            source=source,
            filename=file_path.name,
            file_type="json"
        )

    except Exception as e:
        logger.error(f"❌ Erreur JSON {file_path.name} : {e}")
        return None


# ── Fonction principale ───────────────────────────────────
def read_file(file_path: Path, source: str) -> Optional[dict]:
    """
    Point d'entrée unique du module readers.
    Détecte automatiquement le type de fichier
    et appelle le bon lecteur.

    Usage :
        doc = read_file(Path("data/raw/irena/costs.pdf"), "irena")
        print(doc["text"])  # texte extrait
        print(doc["source"])  # "irena"
    """
    suffix = file_path.suffix.lower()

    # Router vers le bon lecteur selon l'extension
    if suffix == ".pdf":
        return read_pdf(file_path, source)

    elif suffix in [".xlsx", ".xls"]:
        return read_excel(file_path, source)

    elif suffix == ".csv":
        return read_csv(file_path, source)

    elif suffix == ".json":
        return read_json(file_path, source)

    else:
        # Extension non supportée — on log et on ignore
        logger.warning(f"⏭️ Extension non supportée : {suffix} ({file_path.name})")
        return None


# ── Lecture d'une source complète ────────────────────────
def read_source(source_name: str, data_path: Path) -> list[dict]:
    """
    Lit TOUS les fichiers d'une source (ex: tous les fichiers IRENA).

    Retourne une liste de documents — un par fichier réussi.
    Les fichiers qui échouent sont ignorés (logged mais pas bloquants).
    """
    source_path = data_path / source_name

    if not source_path.exists():
        logger.error(f"Dossier introuvable : {source_path}")
        return []

    documents = []

    # Parcourir tous les fichiers du dossier
    all_files = list(source_path.iterdir())
    readable_files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in
        [".pdf", ".xlsx", ".xls", ".csv", ".json"]
        and not f.name.startswith(".")  # ignorer .gitkeep
    ]

    logger.info(f"Source {source_name.upper()} : {len(readable_files)} fichiers à lire")

    for file_path in readable_files:
        doc = read_file(file_path, source_name)
        if doc and len(doc["text"]) > 100:  # ignorer les docs quasi-vides
            documents.append(doc)

    logger.info(f"Source {source_name.upper()} : {len(documents)} documents extraits")
    return documents