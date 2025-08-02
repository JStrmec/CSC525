import os
from time import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .constants import DEFAULT_SVS_MODEL, CHAR_TRUNCATION

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Vector:
    vector: list[float]


class SemanticEncoder:
    def __init__(self):
        # Initializing model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SVS_MODEL)
        self.model = AutoModel.from_pretrained(DEFAULT_SVS_MODEL)
        self.model.eval()
        print("Model initialization finished!")
        self.query_prefix = "query:"
        self.passage_prefix = ""

    def encode(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            timings_dict = {}
            _start_time = time()

            # Extracting text from input
            text = input_dict.get("text", None)
            validated_text = self.validate_and_truncate_text(text)
            if isinstance(validated_text, Dict):
                return validated_text
            text = validated_text

            # Check DataType if it is passage or query
            # and add prefix accordingly
            dataType = input_dict.get("dataType", None)
            if dataType == "passage":
                text = f"{self.passage_prefix} {text}".strip()
            elif dataType == "query":
                text = f"{self.query_prefix} {text}".strip()

            # Tokenization
            tokenized_texts = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            timings_dict["tokenizing_time"] = time()

            # Encoding
            with torch.inference_mode():
                # Forward pass of the model
                model_output = self.model(**tokenized_texts)
                embeddings = self.cls_pooling(model_output.last_hidden_state)
                timings_dict["encoding_time"] = time()

                # Normalizing embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                timings_dict["normalizing_time"] = time()

                # Converting into output format
                output_dict = {"vector": embeddings.squeeze().tolist()}

                self.log_response_timings(
                    action_name="encode_via_ray",
                    start_time=_start_time,
                    timings_dict=timings_dict,
                )
                return output_dict
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}

    def validate_and_truncate_text(self, text: Optional[str]) -> str | Dict[str, str]:
        if text is None:
            print("No text provided in the input dictionary.")
            return {"error": "No `text` input key provided"}
        elif text.isspace() or len(text) == 0:
            print("Empty text provided. Returning empty output.")
            return {"error": "No text provided"}
        elif len(text) > CHAR_TRUNCATION:
            print(f"Input text truncated to {CHAR_TRUNCATION} characters.")
            return text[:CHAR_TRUNCATION]
        return text

    @staticmethod
    def cls_pooling(encoded: torch.Tensor) -> torch.Tensor:
        return encoded[:, 0, :]

    @staticmethod
    def obj_to_bool(s: Any) -> bool:
        if isinstance(s, bool):
            return s
        elif isinstance(s, int):
            return s != 0
        elif isinstance(s, str):
            return str(s).strip().lower() == "true"
        else:
            return False

    def log_response_timings(
        self,
        action_name: str,
        start_time: float,
        timings_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        timings_str = (
            f"Time taken to {action_name} input: {(time() - start_time) * 1000:.1f}ms"
        )
        if timings_dict is not None:
            timings_dict_str = {}
            previous_time = start_time
            for k, v in timings_dict.items():
                timings_dict_str[k] = f"{(v - previous_time) * 1000:.1f}ms"
                previous_time = v
            timings_str += f" {timings_dict_str}"
        print(timings_str)
