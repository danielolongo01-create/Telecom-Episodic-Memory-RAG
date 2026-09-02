import os
import sys
import numpy as np

# Ajout du dossier parent au path Python pour importer depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.episodic_memory import EpisodicMemory
from src.rag_engine import TelecomRAGEngine
from src.llm_orchestrator import LLMOrchestrator


def run_full_evaluation():
    print("==================================================")
    print(" LANCE ET ÉVALUE LE MODÈLE HYBRIDE (TESTS SIMULÉS)")
    print("==================================================")

    feature_dim = 128
    memory = EpisodicMemory(feature_dim=feature_dim)
    rag = TelecomRAGEngine()
    orchestrator = LLMOrchestrator(model_name="llama3")

    # Indexation de la base RAG 3GPP
    rag.index_documents([
        {
            "source": "3GPP TS 38.300 Sec 5.2",
            "content": "Si le BLER dépasse 10%, basculer vers un schéma MCS plus conservateur."
        },
        {
            "source": "3GPP TS 38.214 Sec 6.1",
            "content": "Si le RSRP baisse sous -105 dBm, ajuster le Power Control P_PUSCH."
        }
    ])

    # Scénarios de métriques pour simulation
    test_scenarios = [
        {"rsrp": -85.0, "bler": 0.02, "throughput": 45.0, "status": "Normal"},
        {"rsrp": -112.0, "bler": 0.15, "throughput": 5.0, "status": "Anomalie RSRP/BLER"},
        {"rsrp": -90.0, "bler": 0.12, "throughput": 12.0, "status": "Anomalie BLER"}
    ]

    for idx, metrics in enumerate(test_scenarios, 1):
        print(f"\n--- [Test #{idx}] Scénario : {metrics['status']} ---")
        print(f"RSRP: {metrics['rsrp']} dBm | BLER: {metrics['bler']*100}% | Throughput: {metrics['throughput']} Mbps")

        # Construction du vecteur d'état
        raw_features = [metrics['rsrp'] / -140.0, metrics['bler'], metrics['throughput'] / 100.0]
        vector = np.zeros(feature_dim, dtype='float32')
        vector[:len(raw_features)] = raw_features

        # Recherche d'épisodes similaires
        similar_episodes = memory.search_similar(vector, top_k=2)

        if metrics['bler'] > 0.10 or metrics['rsrp'] < -105:
            print(" Anomalie détectée ! Génération de remédiation...")
            rag_docs = rag.query(f"BLER {metrics['bler']} RSRP {metrics['rsrp']}")
            prompt = orchestrator.construct_prompt(metrics, similar_episodes, rag_docs)
            remediation = orchestrator.generate_remediation(prompt)

            print(" Recommandation :")
            print(remediation)

            # Enregistrement en mémoire épisodique
            memory.add_episode(vector, {
                "id": idx,
                "resolution_action": remediation,
                "kpis": metrics
            })
        else:
            print(" Réseau dans les limites normales.")


if __name__ == "__main__":
    run_full_evaluation()