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


def read_csv(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un CSV et le convertit en texte structuré intelligent.
    
    Gère deux formats :
    1. Format standard  : colonnes normales (OWID, IEA)
    2. Format IRENA     : 1 seule colonne avec long nom,
                          vraies colonnes dans les données
    """
    try:
        # ── Étape 1 : Lire le fichier ─────────────────────
        df = None
        used_encoding = "utf-8"

        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            logger.error(f"❌ Encodage impossible : {file_path.name}")
            return None

        # ── Étape 2 : Détecter et corriger format IRENA ───
        # Format IRENA : 1 colonne avec un très long nom
        # qui contient en réalité plusieurs colonnes séparées
        # par des virgules dans le header original.
        # Solution : re-lire en sautant la première ligne.
        if len(df.columns) == 1 and len(str(df.columns[0])) > 50:
            logger.debug(
                f"Format IRENA détecté : {file_path.name} "
                f"— re-lecture avec skiprows=1"
            )
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=used_encoding,
                    skiprows=1,   # sauter le faux header
                    header=0,     # 1ère ligne = vrais noms de colonnes
                )
                # Nettoyer les espaces dans les noms de colonnes
                df.columns = [str(c).strip() for c in df.columns]
                logger.debug(
                    f"Colonnes détectées : {df.columns.tolist()}"
                )
            except Exception as e:
                logger.warning(
                    f"Re-lecture IRENA échouée pour {file_path.name} : {e}"
                )

        # ── Étape 3 : Nettoyer ────────────────────────────
        if df.empty:
            logger.warning(f"CSV vide : {file_path.name}")
            return None

        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            logger.warning(f"CSV vide après nettoyage : {file_path.name}")
            return None

        rows_count = len(df)

        # ── Étape 4 : Convertir en texte structuré ────────
        # Format : header une seule fois + données compactes
        # Exemple :
        # [Dataset: R-ELECCAP]
        # Colonnes: Region | Technology | Year | Capacity
        # World | Solar | 2023 | 1418000
        # Africa | Solar | 2023 | 12500

        col_names = " | ".join(str(c) for c in df.columns)
        header = f"[Dataset: {file_path.stem}]\nColonnes: {col_names}\n"

        # Limiter à 500 lignes pour éviter les CSV géants
        df_sample = df.head(500)

        lines = []
        for _, row in df_sample.iterrows():
            line = " | ".join([
                str(val).strip() if pd.notna(val) else "N/A"
                for val in row.values
            ])
            # Ignorer les lignes vides
            if line.replace("|", "").replace("N/A", "").strip():
                lines.append(line)

        full_text = header + "\n".join(lines)

        logger.success(
            f"✅ CSV lu : {file_path.name} "
            f"({rows_count} lignes, {len(df.columns)} colonnes)"
        )

        return make_document(
            text=full_text,
            source=source,
            filename=file_path.name,
            file_type="csv",
            extra={"rows": rows_count, "columns": len(df.columns)}
        )

    except Exception as e:
        logger.error(f"❌ Erreur CSV {file_path.name} : {e}")
        return None
# ── Lecteur JSON ──────────────────────────────────────────
def read_json(file_path: Path, source: str) -> Optional[dict]:
    """
    Lit un fichier JSON (métadonnées OWID).
    Contient : description, source, unités, couverture temporelle.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_parts = []

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


# ── Routeur principal ─────────────────────────────────────
def read_file(file_path: Path, source: str) -> Optional[dict]:
    """
    Point d'entrée unique — détecte le type et appelle
    le bon lecteur automatiquement.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(file_path, source)
    elif suffix in [".xlsx", ".xls"]:
        return read_excel(file_path, source)
    elif suffix == ".csv":
        return read_csv(file_path, source)
    elif suffix == ".json":
        return read_json(file_path, source)
    else:
        logger.warning(f"⏭️ Extension non supportée : {suffix} ({file_path.name})")
        return None


# ── Lecture d'une source complète ────────────────────────
def read_source(source_name: str, data_path: Path) -> list[dict]:
    """
    Lit TOUS les fichiers d'une source (ex: tous les fichiers IRENA).
    Retourne une liste de documents — un par fichier réussi.
    """
    source_path = data_path / source_name

    if not source_path.exists():
        logger.error(f"Dossier introuvable : {source_path}")
        return []

    documents = []

    all_files = list(source_path.iterdir())
    readable_files = [
        f for f in all_files
        if f.is_file()
        and f.suffix.lower() in [".pdf", ".xlsx", ".xls", ".csv", ".json"]
        and not f.name.startswith(".")
    ]

    logger.info(f"Source {source_name.upper()} : {len(readable_files)} fichiers à lire")

    for file_path in readable_files:
        doc = read_file(file_path, source_name)
        if doc and len(doc["text"]) > 100:
            documents.append(doc)

    logger.info(f"Source {source_name.upper()} : {len(documents)} documents extraits")
    return documents