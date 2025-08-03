import gradio as gr
from mental_health_retrieval_chatbot.chatbot_responder import ChatbotResponder
from mental_health_retrieval_chatbot.semantic_encoder import SemanticEncoder
from mental_health_retrieval_chatbot.retrieval import MentalHealthVectorStore
from mental_health_retrieval_chatbot.chat_history import ChatHistory

# Initialize once
encoder = SemanticEncoder()
store = MentalHealthVectorStore(encoder)
store.load_index()

responder = ChatbotResponder(store)
history = ChatHistory(max_turns=1, include_bot=True)


# Chat function
def chat(user_input: str, chat_history: list):
    # Generate response
    bot_response = responder.generate_response(user_input=user_input, history=history)
    history.add_turn(user_input, bot_response)

    chat_history = chat_history + [[user_input, bot_response]]

    return chat_history, ""


# Gradio UI
with gr.Blocks(title="Mental Health Chat Companion") as demo:
    gr.Markdown(
        "### 🧠 Mental Health Chat Companion\n_Type your concerns and get thoughtful, supportive replies._"
    )

    chatbot = gr.Chatbot(label="Companion Chat")
    msg = gr.Textbox(label="Your Message", placeholder="What's on your mind?")
    send_btn = gr.Button("Send")

    send_btn.click(fn=chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(fn=chat, inputs=[msg, chatbot], outputs=[chatbot, msg])

demo.launch()
