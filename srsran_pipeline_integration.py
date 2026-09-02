import os
import sys
import time

# Ajout du dossier src au path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from episodic_memory import EpisodicMemory
from rag_engine import TelecomRAGEngine
from llm_orchestrator import LLMOrchestrator
from telemetry_listener import TelemetryListener

def main():
    print("=== PIPELINE IA 5G TEMPS RÉEL (srsRAN + Episodic Memory + RAG) ===")
    
    feature_dim = 128
    memory = EpisodicMemory(feature_dim=feature_dim)
    rag = TelecomRAGEngine()
    orchestrator = LLMOrchestrator(model_name="llama3")
    
    # Chargement de la base de connaissances 3GPP dans le RAG
    rag.index_documents([
        {"source": "3GPP TS 38.300 Sec 5.2", "content": "Si le BLER dépasse 10%, ajuster le schéma de codage MCS et basculer en mode conservateur."},
        {"source": "3GPP TS 38.214 Sec 6.1", "content": "En cas de chute de RSRP sous -105 dBm, augmenter le contrôle de puissance P_PUSCH ou déclencher une procédure de handover."}
    ])

    def process_telemetry(metrics, vector):
        print(f"\n[KPIs reçus] RSRP: {metrics['rsrp']} dBm | BLER: {metrics['bler']}% | Throughput: {metrics['throughput']} Mbps")
        
        # 1. Recherche dans la mémoire épisodique
        similar_episodes = memory.search_similar(vector, top_k=2)
        
        # 2. Détection d'anomalie réseau
        if metrics['bler'] > 0.10 or metrics['rsrp'] < -105:
            print("ANOMALIE DÉTECTÉE - Déclenchement de l'analyse IA...")
            rag_docs = rag.query(f"BLER {metrics['bler']} RSRP {metrics['rsrp']}")
            
            prompt = orchestrator.construct_prompt(metrics, similar_episodes, rag_docs)
            remediation = orchestrator.generate_remediation(prompt)
            
            print("\n [RECOMMANDATION LLM] :")
            print(remediation)
            
            # Stockage de l'événement et de la résolution dans FAISS
            memory.add_episode(vector, {
                "id": len(memory.episodes) + 1,
                "resolution_action": remediation,
                "kpis": metrics
            })
        else:
            print("État du réseau : Normal")

    listener = TelemetryListener(filepath="/tmp/enb_metrics.csv", feature_dim=feature_dim)
    try:
        listener.listen(callback=process_telemetry)
    except KeyboardInterrupt:
        print("\nArrêt du pipeline.")
        listener.stop()

if __name__ == "__main__":
    main()