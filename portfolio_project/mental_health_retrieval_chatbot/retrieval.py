import pandas as pd
import faiss
import numpy as np
import json
from .semantic_encoder import SemanticEncoder
from .constants import TOP_K, SAVED_VECTOR_STORE


class MentalHealthVectorStore:
    def __init__(self, encoder: SemanticEncoder, index_path: str = SAVED_VECTOR_STORE):
        self.encoder = encoder
        self.index_path = index_path
        self.index = None
        self.texts = []

    def load_datasets(self, mh_conv_path: str, counsel_path: str) -> list[str]:
        # Load Mental Health Conversational Data (JSON)
        with open(mh_conv_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        conv_texts = []
        for entry in conv_data:
            print(entry)
            if not isinstance(entry, dict):
                continue
            patterns = entry.get("patterns", [])
            responses = entry.get("responses", [])
            for pattern in patterns:
                for response in responses:
                    combined = f"USER: {pattern.strip()}\nBOT: {response.strip()}"
                    conv_texts.append(combined)

        # Load CounselChat CSV
        counsel_df = pd.read_csv(counsel_path)
        counsel_df = counsel_df[["questionText", "answerText"]].dropna()

        counsel_texts = []
        for _, row in counsel_df.iterrows():
            question = str(row["questionText"]).strip()
            answer = str(row["answerText"]).strip()
            if answer:
                combined = f"Q: {question}\nA: {answer}"
                counsel_texts.append(combined)

        self.texts = conv_texts + counsel_texts
        return self.texts

    def build_index(self):
        vectors = []
        for text in self.texts:
            result = self.encoder.encode({"text": text, "dataType": "passage"})
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
        vector = self.encoder.encode({"text": query, "dataType": "query"})["vector"]
        vector_np = np.array([vector]).astype("float32")
        scores, indices = self.index.search(vector_np, top_k)
        return [self.texts[i] for i in indices[0] if i != -1]
