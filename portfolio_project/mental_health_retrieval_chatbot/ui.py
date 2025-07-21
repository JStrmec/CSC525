import gradio as gr
from mental_health_retrieval_chatbot.chatbot_responder import ChatbotResponder
from mental_health_retrieval_chatbot.semantic_encoder import SemanticEncoder
from mental_health_retrieval_chatbot.retrieval import MentalHealthVectorStore

# Initialize once
encoder = SemanticEncoder()
store = MentalHealthVectorStore(encoder)

# Load datasets and index (ensure the index has been built already to skip rebuilding)
store.load_index()

# Create chatbot responder
responder = ChatbotResponder(store)


# Chat function
def chat(user_input, history):
    if history is None:
        history = []

    # Combine previous chat into a string
    history_str = "\n".join([f"USER: {u}\nBOT: {b}" for u, b in history])

    # Generate chatbot response
    bot_response = responder.generate_response(user_input, chat_history=history_str)

    # Append to history
    history.append((user_input, bot_response))
    return history, history


# Gradio UI
with gr.Blocks(title="Mental Health Chat Companion") as demo:
    gr.Markdown(
        "### 🧠 Mental Health Chat Companion\n_Type your concerns and get thoughtful, supportive replies._"
    )

    chatbot = gr.Chatbot(label="Companion Chat")
    msg = gr.Textbox(label="Your Message", placeholder="What's on your mind?")
    state = gr.State([])

    send_btn = gr.Button("Send")

    send_btn.click(fn=chat, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(fn=chat, inputs=[msg, state], outputs=[chatbot, state])

demo.launch()
