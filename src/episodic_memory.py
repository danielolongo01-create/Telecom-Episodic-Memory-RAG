import faiss
import numpy as np

class EpisodicMemory:
    def __init__(self, feature_dim=128, alpha=0.7):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.index = faiss.IndexFlatL2(feature_dim)
        self.episodes = []

    def add_episode(self, vector, metadata):
        vector = np.array([vector], dtype='float32')
        self.index.add(vector)
        self.episodes.append(metadata)

    def search_similar(self, vector, top_k=2):
        if self.index.ntotal == 0:
            return []
        vector = np.array([vector], dtype='float32')
        distances, indices = self.index.search(vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.episodes):
                # Conversion distance L2 -> score de similarité basique
                score = float(1.0 / (1.0 + dist))
                results.append((score, self.episodes[idx]))
        return results