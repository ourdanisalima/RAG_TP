import sys
import io
import os
import argparse
from dotenv import load_dotenv

# UTF-8 safe (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY manquante dans le fichier .env")
    sys.exit(1)

from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent


# ─────────────────────────────────────────────
def log(title, msg):
    print(f"\n[{title}] {msg}")


def print_bar():
    print("-" * 60)


# ─────────────────────────────────────────────
def afficher_sources(chunks):
    print("\n📚 SOURCES UTILISÉES\n")

    for i, c in enumerate(chunks, 1):
        nom = c["metadata"].get("denomination", "Inconnu")
        type_c = c["metadata"].get("type_chunk", "?")
        score = c["score"]

        print(f"{i}. {nom} [{type_c}]")
        print(f"   score: {score:.1%}")


# ─────────────────────────────────────────────
def main(avec_reformulation=True):

    print("=" * 60)
    print("⚕️ ASSISTANT MÉDICAMENTS RAG (BDPM + GROQ)")
    print("=" * 60)

    print("\nArchitecture : Retriever → Generator → Réponse")

    # ── Chargement ───────────────────────────
    log("SETUP", "Chargement du modèle et de la base...")

    try:
        modele = charger_modele()
        index, chunks = charger_index()
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        sys.exit(1)

    retriever = RetrieverAgent(modele, index, chunks, avec_reformulation)
    generator = GeneratorAgent()

    mode = "reformulation ON" if avec_reformulation else "standard"
    log("OK", f"Système prêt ({mode})")

    print("\nCommandes : 'quit' | 'sources'")
    print_bar()

    history = []
    last_chunks = []

    # ── LOOP ────────────────────────────────
    while True:

        try:
            question = input("\n❓ Question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Bye")
            break

        if question.lower() == "sources":
            if last_chunks:
                afficher_sources(last_chunks)
            continue

        # ── Retriever ───────────────────────
        log("R1", "Recherche des documents...")

        res = retriever.run(question, historique=history)

        if res["reformulee"]:
            print(f"↪ reformulation : {res['question_recherche']}")

        print(f"📊 score max : {res['meilleur_score']:.1%}")

        if not res["confiance_ok"]:
            print("\n⚠️ Aucune information fiable trouvée.")
            print("Consultez un professionnel de santé.")
            continue

        chunks = res["resultats"]
        last_chunks = chunks

        print(f"📄 {len(chunks)} chunks récupérés")

        # ── Generator ───────────────────────
        log("R2", "Génération de la réponse...")

        answer = generator.run(question, chunks, historique=history)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        print_bar()
        print(answer)
        print_bar()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bonus", action="store_true")
    args = parser.parse_args()

    main(avec_reformulation=not args.no_bonus)