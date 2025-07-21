import pandas as pd
import faiss
import numpy as np
import json
from semantic_encoder import SemanticEncoder  # Adjust path if needed
from constants import TOP_K


class MentalHealthVectorStore:
    def __init__(
        self, encoder: SemanticEncoder, index_path: str = "mental_health.index"
    ):
        self.encoder = encoder
        self.index_path = index_path
        self.index = None
        self.texts = []

    def load_datasets(self, reddit_path: str, counsel_path: str) -> list[str]:
        # Load Reddit Mental Health dataset
        reddit_df = pd.read_csv(reddit_path)
        reddit_texts = reddit_df["post_text"].dropna().tolist()

        # Load CounselChat dataset
        with open(counsel_path, "r", encoding="utf-8") as f:
            counsel_data = json.load(f)

        # Extract both questions and answers
        counsel_texts = []
        for entry in counsel_data:
            question = entry.get("questionText", "").strip()
            answers = entry.get("answers", [])
            for ans in answers:
                answer = ans.get("answerText", "").strip()
                if answer:
                    # Combine question and answer for better context
                    combined = f"Q: {question}\nA: {answer}"
                    counsel_texts.append(combined)

        self.texts = reddit_texts + counsel_texts
        return self.texts

    def build_index(self, quantize: bool = False):
        vectors = []
        for text in self.texts:
            result = self.encoder.encode(
                {"text": text, "dataType": "passage", "quantize": quantize}
            )
            if "vector" in result:
                vectors.append(result["vector"])
        vectors_np = np.array(vectors).astype("float32")

        self.index = faiss.IndexFlatIP(vectors_np.shape[1])
        self.index.add(vectors_np)
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".txt", "w", encoding="utf-8") as f:
            json.dump(self.texts, f)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".txt", "r", encoding="utf-8") as f:
            self.texts = json.load(f)

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        vector = self.encoder.encode(
            {"text": query, "dataType": "query", "quantize": False}
        )["vector"]
        vector_np = np.array([vector]).astype("float32")
        scores, indices = self.index.search(vector_np, top_k)
        return [self.texts[i] for i in indices[0] if i != -1]
