import pandas as pd
import faiss
import numpy as np
import json
from .semantic_encoder import SemanticEncoder
from .constants import TOP_K, SAVED_VECTOR_STORE
import re
import html
from langdetect import detect
from googletrans import Translator


class MentalHealthVectorStore:
    def __init__(self, encoder: SemanticEncoder, index_path: str = SAVED_VECTOR_STORE):
        self.encoder = encoder
        self.index_path = index_path
        self.index = None
        self.texts = []
        self.translator = Translator()

    @staticmethod
    def clean_html(raw_html: str) -> str:
        cleaned = re.sub(
            r"<(script|style).*?>.*?</\1>",
            "",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"<(br|BR)\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"</?(p|P|div|DIV|li|LI)[^>]*>", "\n", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\n+", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def is_spanish(text: str) -> bool:
        try:
            return detect(text) == "es"
        except Exception:
            return False

    def translate_if_spanish(self, text: str) -> str:
        if self.is_spanish(text):
            translated = self.translator.translate(text, src="es", dest="en")
            return translated.text
        else:
            return text

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
                    combined = f"Q: {pattern.strip()}\nA: {response.strip()}"
                    conv_texts.append(combined)

        # Load CounselChat CSV
        counsel_df = pd.read_csv(counsel_path)
        counsel_df = counsel_df[["questionText", "answerText"]].dropna()

        counsel_texts = []
        for _, row in counsel_df.iterrows():
            question = self.translate_if_spanish(
                self.clean_html(raw_html=str(row["questionText"]).strip())
            )
            answer = self.translate_if_spanish(
                self.clean_html(raw_html=str(row["answerText"]).strip())
            )

            # Split into advice sentences / chunks
            advice_sentences = [
                s.strip() for s in answer.split(".") if len(s.strip()) > 20
            ]

            for advice in advice_sentences:
                combined = f"Q: {question}\nA: {advice}"
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
