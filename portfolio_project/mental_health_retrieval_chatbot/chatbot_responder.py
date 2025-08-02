from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from .constants import DEFAULT_CHAT_MODEL, DEFAULT_CHAT_PROMPT, TOP_K, MAX_INPUT_TOKENS
from .retrieval import MentalHealthVectorStore


class ChatbotResponder:
    def __init__(self, vector_store: MentalHealthVectorStore):
        self.vector_store = vector_store
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CHAT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_CHAT_MODEL)

    def generate_response(self, user_input: str, chat_history: str = "") -> str:
        context = "\n".join(self.vector_store.search(user_input, top_k=TOP_K))
        prompt = self.format_prompt(
            user_input=user_input, chat_history=chat_history, context=context
        )
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )
        # Truncate input_ids to the last 1024 tokens if longer to prevent crash
        if input_ids.shape[-1] > 1024:
            input_ids = input_ids[:, -1024:]
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=150,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=5,
            top_p=0.1,
        )
        return (
            self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            .split("BOT:")[-1]
            .strip()
        )

    @staticmethod
    def format_prompt(
        user_input: str, chat_history: str = "", context: str = ""
    ) -> str:
        return DEFAULT_CHAT_PROMPT.format(
            user_input=user_input, chat_history=chat_history, context=context
        ).strip()


class LlamaChatbotResponder(ChatbotResponder):
    def __init__(self, vector_store: MentalHealthVectorStore):
        super().__init__(vector_store)
        self.model_id = "akjindal53244/Llama-3.1-Storm-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate_response(self, user_input: str, chat_history: str = "") -> str:
        context = "\n".join(self.vector_store.search(user_input, top_k=TOP_K))
        chat_history = ""
        context = ""
        prompt = DEFAULT_CHAT_PROMPT.format(
            user_input=user_input, chat_history=chat_history, context=context
        ).strip()

        messages = [
            {
                "role": "system",
                "content": "You are a mental health specialist chatbot meant to answer questions about mental health and wellness using context and user conversation history to answer respectfully, and considerately.",
            },
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.01,
            top_k=100,
            top_p=0.95,
        )
        return outputs[0]["generated_text"].strip()
