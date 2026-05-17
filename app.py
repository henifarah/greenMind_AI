# app.py
# Interface GreenMind AI — RAG Énergies Renouvelables

import os
import time
import json
from datetime import datetime

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

from retriever import GreenMindRetriever
from config import LLM_MODEL, OLLAMA_URL, QDRANT_URL, QDRANT_COLLECTION

# ── Configuration page ────────────────────────────────────
st.set_page_config(
    page_title="GreenMind AI - RAG Énergies Renouvelables",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #60ad5e 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .chat-user {
        background-color: #e8f5e9;
        border-radius: 1rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 5px solid #2e7d32;
    }
    .chat-assistant {
        background-color: #f1f8e9;
        border-radius: 1rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 5px solid #ffb74d;
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #c8e6c9;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Gestion conversations JSON ────────────────────────────
def load_conversations():
    if os.path.exists("conversations.json"):
        with open("conversations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_conversations(conversations):
    with open("conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def get_current_conversation():
    convs = st.session_state.get("conversations", {})
    current_id = st.session_state.get("current_conversation_id", "default")
    if current_id not in convs:
        convs[current_id] = {
            "name": "Nouvelle conversation",
            "messages": [],
            "created": datetime.now().isoformat()
        }
        st.session_state.conversations = convs
        save_conversations(convs)
    return convs[current_id]


# ── Initialisation ────────────────────────────────────────
@st.cache_resource
def load_retriever():
    return GreenMindRetriever()

@st.cache_resource
def load_llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL)

@st.cache_resource
def get_chunks_count():
    client = QdrantClient(url=QDRANT_URL)
    return client.count(QDRANT_COLLECTION).count

# Initialiser session state
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()
    if not st.session_state.conversations:
        st.session_state.conversations = {
            "default": {
                "name": "Conversation principale",
                "messages": [],
                "created": datetime.now().isoformat()
            }
        }
        save_conversations(st.session_state.conversations)

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = "default"

retriever = load_retriever()
llm       = load_llm()


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/solar-panel.png", width=80)
    st.markdown("## 🌍 GreenMind AI")
    st.markdown("*Assistant RAG sur les énergies renouvelables*")
    st.markdown("---")

    # Conversations
    st.markdown("### 💬 Conversations")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_conv_name = st.text_input(
            "Nom nouvelle conv.",
            placeholder="Ex: Étude solaire",
            key="new_conv_name"
        )
    with col2:
        if st.button("➕ Créer", use_container_width=True):
            if new_conv_name.strip():
                conv_id = str(int(time.time()))
                st.session_state.conversations[conv_id] = {
                    "name": new_conv_name.strip(),
                    "messages": [],
                    "created": datetime.now().isoformat()
                }
                save_conversations(st.session_state.conversations)
                st.session_state.current_conversation_id = conv_id
                st.rerun()
            else:
                st.error("Nom requis")

    # Liste conversations
    for conv_id, conv_data in list(st.session_state.conversations.items()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"📁 {conv_data['name']}", key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.current_conversation_id = conv_id
                st.rerun()
        with col2:
            if conv_id != "default" and st.button("🗑️", key=f"del_{conv_id}"):
                del st.session_state.conversations[conv_id]
                if st.session_state.current_conversation_id == conv_id:
                    st.session_state.current_conversation_id = "default"
                save_conversations(st.session_state.conversations)
                st.rerun()

    st.markdown("---")

    # Statistiques
    st.markdown("### 📊 Statistiques")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks indexés", f"{get_chunks_count():,}")
    with col2:
        st.metric("Conversations", len(st.session_state.conversations))

    st.markdown("---")

    # Paramètres RAG
    st.markdown("### 🔧 Paramètres RAG")
    top_k = st.slider("Documents à consulter", 3, 10, 5)

    st.markdown("---")

    # À propos
    st.markdown("### ℹ️ À propos")
    st.caption("Sources : IRENA · IEA · OWID")
    st.caption("Embedding : BGE-M3 1024D")
    st.caption("LLM : Mistral via Ollama")
    st.caption("Architecture : RAG + MCP")


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌱 GreenMind AI</h1>
    <p>Système RAG intelligent pour les énergies renouvelables</p>
</div>
""", unsafe_allow_html=True)


# ── Affichage conversation ────────────────────────────────
current_conv = get_current_conversation()
messages     = current_conv["messages"]

chat_container = st.container()
with chat_container:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><strong>👤 Vous :</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-assistant"><strong>🤖 GreenMind :</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )


# ── Traitement question ───────────────────────────────────
def process_question(question):
    if not question:
        return

    # Ajouter question dans la conversation
    current_conv = get_current_conversation()
    current_conv["messages"].append({"role": "user", "content": question})
    save_conversations(st.session_state.conversations)

    # Afficher question
    with chat_container:
        st.markdown(
            f'<div class="chat-user"><strong>👤 Vous :</strong><br>{question}</div>',
            unsafe_allow_html=True
        )

    # Pipeline RAG
    with st.status("🔍 GreenMind analyse...", expanded=False) as status:

        # Étape 1 : Recherche sémantique
        chunks = retriever.search(question, top_k=top_k)
        status.update(label="📄 Documents trouvés", state="running")
        time.sleep(0.3)

        # Étape 2 : Calcul LCOE si question financière
        lcoe      = None
        lcoe_info = ""
        mots_lcoe = ["coût", "lcoe", "prix", "kwh", "cout", "cost", "price"]

        if any(mot in question.lower() for mot in mots_lcoe):
            lcoe = retriever.calculate_lcoe("solar_tunisia")
            lcoe_info = f"""
[Calcul LCOE — {lcoe['source']}]
Technologie : {lcoe['technologie']}
LCOE exact  : {lcoe['lcoe']} $/kWh = {lcoe['lcoe_cents']} centimes/kWh
"""
            status.update(label="💹 LCOE calculé", state="running")
            time.sleep(0.3)

        # Étape 3 : Construire le contexte
        context = "\n\n".join([
            f"[{c['source'].upper()} — {c['filename']}]\n{c['text']}"
            for c in chunks
        ])

        # Étape 4 : Prompt structuré
        prompt = f"""Tu es GreenMind, expert en énergies renouvelables.

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les informations du contexte ci-dessous
2. Donne les chiffres EXACTS sans conversion ni arrondi
3. $0.0403/kWh = 4.03 centimes — NE PAS dire 40 centimes
4. Cite toujours la source exacte
5. NE FAIS AUCUN CALCUL supplémentaire

CONTEXTE :
{context}
{lcoe_info}

QUESTION : {question}

FORMAT :
Chiffre clé : [valeur exacte avec unité]
Explication : [1-2 phrases simples]
Source : [nom fichier ou rapport]"""

        # Étape 5 : Génération LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        answer   = response.content
        status.update(label="✅ Réponse générée", state="complete")

    # Afficher réponse
    with chat_container:
        st.markdown(
            f'<div class="chat-assistant"><strong>🤖 GreenMind :</strong><br>{answer}</div>',
            unsafe_allow_html=True
        )

    # Afficher sources
    with st.expander("📚 Sources utilisées", expanded=False):
        for c in chunks:
            score = f"{c['score']*100:.1f}%"
            st.markdown(f"""
<div class="source-card">
    <strong>📁 {c['source'].upper()}</strong> — {c['filename']}<br>
    <span style="color:#2e7d32;">🔗 Pertinence : {score}</span><br>
    <span style="font-size:0.75rem;">📄 {c['text'][:200]}...</span>
</div>
""", unsafe_allow_html=True)

    # Afficher LCOE si calculé
    if lcoe:
        with st.expander("💰 Calcul LCOE Tunisie", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("LCOE calculé",    f"${lcoe['lcoe']}/kWh")
            col2.metric("En centimes",     f"{lcoe['lcoe_cents']} cts")
            col3.metric("Capacity Factor", "22% (Tunisie)")
            st.caption(f"Source : {lcoe['source']}")

    # Sauvegarder réponse
    current_conv["messages"].append({"role": "assistant", "content": answer})
    save_conversations(st.session_state.conversations)

    st.rerun()


# ── Input ─────────────────────────────────────────────────
question = st.chat_input("💬 Posez votre question sur les énergies renouvelables...")
if question:
    process_question(question)


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2rem; padding:1rem;
            background-color:#e8f5e9; border-radius:1rem;">
    🌞 Powered by RAG | Ollama | Qdrant | BGE-M3 | IRENA · IEA · OWID
</div>
""", unsafe_allow_html=True)