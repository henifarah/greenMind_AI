from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from config import settings


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    Crée le splitter avec les paramètres de config.py.

    RecursiveCharacterTextSplitter essaie de couper le texte
    aux frontières naturelles dans cet ordre :
    1. Entre deux paragraphes (\n\n) ← idéal
    2. Entre deux lignes (\n)
    3. Fin de phrase (". ")
    4. Entre deux mots (" ")
    5. Caractère par caractère ← dernier recours

    Pourquoi "Recursive" ?
    → Il essaie le premier séparateur.
    → Si le chunk est encore trop grand, il essaie le suivant.
    → Récursivement jusqu'à trouver la bonne taille.
    """
    return RecursiveCharacterTextSplitter(
        # Taille maximale d'un chunk en caractères
        # On utilise des caractères ici (pas des tokens)
        # car le comptage exact des tokens est lent.
        # 512 tokens ≈ 1500-2000 caractères en français/anglais
        chunk_size=1800,

        # Overlap en caractères entre chunks consécutifs
        # 64 tokens ≈ 200 caractères
        chunk_overlap=200,

        # Séparateurs dans l'ordre de priorité
        separators=["\n\n", "\n", ". ", " ", ""],

        # Ne pas couper au milieu des mots
        length_function=len,
    )


def chunk_document(document: dict) -> list[dict]:
    """
    Découpe un document en chunks.

    Reçoit : un document du format make_document() de readers.py
    Retourne : liste de chunks, chacun avec :
        - text      : le texte du chunk
        - source    : hérité du document parent (ex: "irena")
        - filename  : hérité du document parent
        - chunk_id  : numéro du chunk dans ce document
        - chunk_total : nombre total de chunks dans ce document
        - + toutes les autres métadonnées du document parent

    Pourquoi hériter des métadonnées ?
    → Quand Qdrant retourne ce chunk, on sait exactement
      d'où il vient pour citer la source précisément.
    """
    text = document.get("text", "")

    # Ignorer les documents trop courts
    # Un document de moins de 50 caractères n'a aucune valeur
    if len(text) < settings.CHUNK_MIN_SIZE:
        logger.warning(
            f"Document trop court ignoré : {document.get('filename')} "
            f"({len(text)} caractères)"
        )
        return []

    # Créer le splitter
    splitter = create_splitter()

    # Découper le texte en chunks
    raw_chunks = splitter.split_text(text)

    # Construire la liste de chunks avec métadonnées complètes
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):

        # Ignorer les chunks trop courts (fragments sans sens)
        if len(chunk_text.strip()) < settings.CHUNK_MIN_SIZE:
            continue

        # Construire le chunk avec toutes ses métadonnées
        chunk = {
            # Le texte du chunk — c'est ce que BGE-M3 va encoder
            "text": chunk_text.strip(),

            # Métadonnées héritées du document parent
            "source":    document.get("source", "unknown"),
            "filename":  document.get("filename", "unknown"),
            "file_type": document.get("file_type", "unknown"),

            # Métadonnées de position dans le document
            # Crucial pour citer "page X" dans les réponses
            "chunk_id":    i,
            "chunk_total": len(raw_chunks),

            # Taille du chunk en caractères
            "chunk_size": len(chunk_text),
        }

        # Hériter les métadonnées supplémentaires du document
        # ex: "pages" pour les PDFs, "rows" pour les CSV
        for key in ["pages", "sheets", "rows"]:
            if key in document:
                chunk[key] = document[key]

        chunks.append(chunk)

    logger.debug(
        f"📄 {document.get('filename')} → "
        f"{len(chunks)} chunks "
        f"(~{len(text)//len(chunks) if chunks else 0} car/chunk)"
    )

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Découpe une liste de documents en chunks.

    C'est la fonction principale appelée par ingest.py.
    Elle traite tous les documents d'une source et retourne
    tous leurs chunks dans une seule liste plate.

    Exemple :
        15 documents IRENA
        → document 1 : 847 chunks
        → document 2 : 234 chunks
        → ...
        → Total : ~8000 chunks pour IRENA seul
    """
    all_chunks = []
    total_docs = len(documents)

    logger.info(f"Chunking de {total_docs} documents...")

    for i, document in enumerate(documents, start=1):
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

        # Log de progression tous les 10 documents
        if i % 10 == 0 or i == total_docs:
            logger.info(
                f"Progression : {i}/{total_docs} documents "
                f"→ {len(all_chunks)} chunks au total"
            )

    logger.success(
        f"✅ Chunking terminé : {total_docs} documents "
        f"→ {len(all_chunks)} chunks"
    )

    return all_chunks