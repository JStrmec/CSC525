from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import DEFAULT_CHAT_MODEL, DEFAULT_CHAT_PROMPT
from . import MentalHealthVectorStore


class ChatbotResponder:
    def __init__(self, vector_store: MentalHealthVectorStore):
        self.vector_store = vector_store
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CHAT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_CHAT_MODEL)

    def generate_response(self, user_input: str, chat_history: str = "") -> str:
        context = "\n".join(self.vector_store.search(user_input, top_k=5))
        prompt = DEFAULT_CHAT_PROMPT.format(
            user_input=user_input, chat_history=chat_history, context=context
        ).strip()
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        output_ids = self.model.generate(
            input_ids,
            max_length=512,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        return (
            self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            .split("CHATBOT:")[-1]
            .strip()
        )
