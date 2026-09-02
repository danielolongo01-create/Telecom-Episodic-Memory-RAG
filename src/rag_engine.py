from typing import List, Dict

class TelecomRAGEngine:
    def __init__(self):
        self.documents = []

    def index_documents(self, docs: List[Dict]):
        self.documents.extend(docs)

    def query(self, query_str: str, top_k: int = 2) -> List[Dict]:
        results = []
        query_words = set(query_str.lower().split())
        for doc in self.documents:
            content = doc.get("content", "").lower()
            if any(word in content for word in query_words):
                results.append(doc)
            if len(results) >= top_k:
                break
        return results if results else self.documents[:top_k]