from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from .constants import (
    TOP_K,
    MAX_INPUT_TOKENS,
    CLEANING_MAP,
    DEFAULT_CHAT_MODEL,
)
import torch
from .retrieval import MentalHealthVectorStore
from .prompt_builder import PromptBuilder
from .chat_history import ChatHistory
from typing import Any


class ChatbotResponder:
    def __init__(self, vector_store: MentalHealthVectorStore):
        torch.manual_seed(42)
        self.vector_store = vector_store
        self.prompt_builder = PromptBuilder()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

    def generate_response(self, user_input: str, history: ChatHistory) -> str:
        context = "\n".join(self.vector_store.search(user_input, top_k=TOP_K))
        print("Context retrieved:", context)
        prompt = self.prompt_builder.build(
            user_input=user_input, chat_history=history, context=context
        )

        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )

        # Generate tokens
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=150,
            min_length=20,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            repetition_penalty=1.2,
            temperature=0.7,
            top_k=50,
            top_p=0.85,
        )

        return self.format_response(
            output_ids=output_ids, input_ids=input_ids, context=context
        )

    def format_response(
        self, output_ids: torch.Tensor, input_ids: torch.Tensor, context: str
    ) -> str:
        # Slice off the prompt to get only the generated text
        gen_tokens = output_ids[:, input_ids.shape[-1] :]
        raw_response = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        if not raw_response:
            if "A:" in context:
                raw_response = context.split("A:")[-1].strip()
            else:
                raw_response = context.strip()

        for key, value in CLEANING_MAP.items():
            raw_response = raw_response.replace(key, value)

        clean_bot_response = raw_response.strip()
        print("Cleaned response:", clean_bot_response)
        return clean_bot_response


class FineTuned(ChatbotResponder):
    def __init__(self, vector_store: MentalHealthVectorStore):
        super().__init__(vector_store)
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CHAT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_CHAT_MODEL)

    def generate_response(self, user_input: str, history: ChatHistory) -> str:
        context = "\n".join(self.vector_store.search(user_input, top_k=TOP_K))
        print("Context retrieved:", context)
        prompt = self.prompt_builder.build(
            user_input=user_input, chat_history=history, context=context
        )

        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )
        if input_ids.shape[-1] > 1024:
            input_ids = input_ids[:, -1024:]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=150,
            min_length=20,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            repetition_penalty=1.2,
            temperature=0.3,
            top_k=50,
            top_p=0.75,
        )
        return self.format_response(
            output_ids=output_ids, input_ids=input_ids, context=context
        )

    def format_response(
        self, output: Any, input_ids: torch.Tensor, context: str
    ) -> str:
        gen_tokens = output[:, input_ids.shape[-1] :]
        raw_response = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        if not raw_response:
            if "A:" in context:
                raw_response = context.split("A:")[-1].strip()
            else:
                raw_response = context.strip()
        # Clean up the response
        for key, value in CLEANING_MAP.items():
            raw_response = raw_response.replace(key, value)
        clean_bot_response = raw_response.strip()
        print("Cleaned response:", clean_bot_response)
        return clean_bot_response
