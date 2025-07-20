from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import DEFAULT_CHAT_MODEL

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CHAT_MODEL)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_CHAT_MODEL)

input_text = "I'm feeling really stressed out about everything lately."
input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

chat_history_ids = model.generate(
    input_ids,
    max_length=100,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

print(
    tokenizer.decode(
        chat_history_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True
    )
)
