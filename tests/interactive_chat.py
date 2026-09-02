import os
import sys

# Ajout du dossier parent au path Python pour importer depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import TelecomRAGEngine
from src.llm_orchestrator import LLMOrchestrator


def start_chat():
    print("==================================================")
    print(" CHAT INTERACTIF D'ASSISTANCE RÉSEAU 5G (RAG + LLM)")
    print(" Tapez 'exit' ou 'quit' pour quitter.")
    print("==================================================")

    rag = TelecomRAGEngine()
    orchestrator = LLMOrchestrator(model_name="llama3")

    # Indexation de la base de documentation
    rag.index_documents([
        {
            "source": "3GPP TS 38.300",
            "content": "En cas d'augmentation du BLER au-delà de 10% lors d'une forte mobilité, adapter le schéma de modulation et codage (MCS)."
        },
        {
            "source": "3GPP TS 38.214",
            "content": "Pour résoudre la saturation du débit sous brouillage, appliquer un ajustement dynamique de la puissance de transmission en Uplink."
        }
    ])

    while True:
        try:
            query = input("\n[Vous] : ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                print("Fermeture du chat interactif.")
                break

            # Recherche des documents pertinents
            rag_docs = rag.query(query, top_k=2)

            # KPIs fictifs pour la requête
            kpis = {"rsrp": "Non spécifié", "bler": "Non spécifié", "throughput": "Non spécifié"}
            
            prompt = orchestrator.construct_prompt(kpis, [], rag_docs)
            response = orchestrator.generate_remediation(prompt)

            print(f"\n[IA 5G] :\n{response}")

        except KeyboardInterrupt:
            print("\nFermeture du chat interactif.")
            break


if __name__ == "__main__":
    start_chat()