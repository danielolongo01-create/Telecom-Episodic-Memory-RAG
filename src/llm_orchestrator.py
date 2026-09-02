from typing import List, Dict
import ollama

class LLMOrchestrator:
    """
    Orchestrateur LLM utilisant Meta Llama 3 en local via Ollama.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def construct_prompt(self, current_kpis: Dict, top_episodes: List, rag_docs: List) -> str:
        """
        Formate le prompt d'ancrage strict pour limiter les hallucinations.
        """
        episodes_text = ""
        for i, (score, ep) in enumerate(top_episodes, 1):
            episodes_text += f"\n- Épisode passé #{ep.get('id', i)} (Similarité: {score*100:.1f}%): " \
                            f"Action appliquée: {ep.get('resolution_action', 'N/A')}"

        rag_text = ""
        for doc in rag_docs:
            rag_text += f"\n- [{doc.get('source', '3GPP')}]: {doc.get('content', '')}"

        prompt = f"""Tu es un expert ingénieur réseau 5G. Analyse les données suivantes :

[CONTEXTE SITUATORIEL RÉSEAU ACTUEL]
KPIs mesurés : RSRP = {current_kpis.get('rsrp')} dBm, BLER = {current_kpis.get('bler')}%, Throughput = {current_kpis.get('throughput')} Mbps

[MÉMOIRE ÉPISODIQUE - ÉVÉNEMENTS HISTORIQUES SIMILAIRES]
{episodes_text if episodes_text else "Aucun historique pertinent trouvé."}

[RECOMMANDATIONS NORMES 3GPP / RAG]
{rag_text if rag_text else "Aucune documentation spécifique trouvée."}

Sur la base de ces informations, fournis une recommandation de remédiation technique concrète et concise pour résoudre le problème réseau.
"""
        return prompt

    def generate_remediation(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"[LLM Error] Impossible de générer la réponse via Ollama : {e}"