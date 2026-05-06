import streamlit as st
import os
from dotenv import load_dotenv
from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent

# ── Config page ─────────────────────────────
st.set_page_config(
    page_title="MediAssist",
    page_icon="⚕️",
    layout="wide"
)

# ── Style simple ────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1f2937;
}

[data-testid="stSidebar"] {
    background-color: #f9fafb;
}

button {
    background-color: #0D9DA8 !important;
    color: white !important;
    border-radius: 6px !important;
}

[data-testid="stChatInputContainer"] {
    background-color: #f3f4f6 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Env ─────────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("Clé API manquante (.env)")
    st.stop()

# ── Init RAG ────────────────────────────────
@st.cache_resource
def init_rag():
    modele = charger_modele()
    index, chunks = charger_index()
    retriever = RetrieverAgent(modele, index, chunks, avec_reformulation=True)
    generator = GeneratorAgent()
    return retriever, generator

retriever, generator = init_rag()

# ── Session state ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ─────────────────────────────────
with st.sidebar:
    st.title("⚕️ MediAssist")
    st.caption("Assistant médicaments")

    st.divider()

    st.metric("Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))

    st.divider()

    if st.button("🗑️ Nouvelle session"):
        st.session_state.messages = []
        st.rerun()

    st.caption("⚠️ Ne remplace pas un avis médical")

# ── Header ──────────────────────────────────
st.title("⚕️ Assistant Médicaments")
st.caption("Pose tes questions sur les médicaments simplement")

# ── Chat history ────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input user ──────────────────────────────
if prompt := st.chat_input("Ex : effets secondaires du paracétamol ?"):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche..."):

            historique = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            resultat = retriever.run(prompt, historique=historique)

            if not resultat["confiance_ok"]:
                reponse = "Je n'ai pas trouvé d'information pertinente dans la base."
            else:
                chunks = resultat["resultats"]
                reponse = generator.run(prompt, chunks, historique=historique)

            st.markdown(reponse)

    # sauvegarde
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reponse})

    st.rerun()