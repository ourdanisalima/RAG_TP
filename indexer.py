import sys
import io
import time
from src.loader import charger_medicaments
from src.chunker import chunker_medicaments
from src.embedder import charger_modele, creer_index, sauvegarder_index

# UTF-8 safe output (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def log(step, message):
    print(f"\n[{step}] {message}")


def main():
    start = time.time()

    print("=" * 60)
    print("   ⚕️ INDEXATION - Assistant Médicaments RAG")
    print("=" * 60)

    try:
        # ─────────────────────────────
        log("1/4", "Chargement des données BDPM...")
        t0 = time.time()
        medicaments = charger_medicaments()
        print(f"   ✔ {len(medicaments)} médicaments chargés ({time.time()-t0:.1f}s)")

        # ─────────────────────────────
        log("2/4", "Découpage en chunks...")
        t0 = time.time()
        chunks = chunker_medicaments(medicaments, taille_max=500, overlap=50)
        print(f"   ✔ {len(chunks)} chunks créés ({time.time()-t0:.1f}s)")

        # ─────────────────────────────
        log("3/4", "Chargement du modèle d'embedding...")
        t0 = time.time()
        modele = charger_modele()
        print(f"   ✔ Modèle chargé ({time.time()-t0:.1f}s)")

        # ─────────────────────────────
        log("4/4", "Création de l'index FAISS...")
        t0 = time.time()
        index = creer_index(chunks, modele)
        sauvegarder_index(index, chunks)
        print(f"   ✔ Index sauvegardé ({time.time()-t0:.1f}s)")

        # ─────────────────────────────
        total = time.time() - start
        print("\n" + "=" * 60)
        print(f"   Indexation terminée en {total:.1f}s")
        print("=" * 60)

    except Exception as e:
        print("\n❌ ERREUR pendant l'indexation :")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()