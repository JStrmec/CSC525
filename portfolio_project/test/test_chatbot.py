import os
import unittest
from mental_health_retrieval_chatbot.semantic_encoder import SemanticEncoder
from mental_health_retrieval_chatbot.retrieval import MentalHealthVectorStore
from mental_health_retrieval_chatbot.chatbot_responder import ChatbotResponder
from mental_health_retrieval_chatbot.constants import (
    MENTAL_HEALTH_CONVO_PATH,
    COUNSEL_CHAT_PATH,
    SAVED_VECTOR_STORE,
)


class TestChatBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = SemanticEncoder()
        cls.index_path = SAVED_VECTOR_STORE
        cls.store = MentalHealthVectorStore(cls.encoder, index_path=cls.index_path)
        # Only build index if it doesn't already exist
        if not os.path.exists(cls.index_path) or not os.path.exists(
            cls.index_path + ".txt"
        ):
            print("[TEST SETUP] Building FAISS index...")
            cls.store.load_datasets(MENTAL_HEALTH_CONVO_PATH, COUNSEL_CHAT_PATH)
            cls.store.build_index()
        else:
            print("[TEST SETUP] Loading cached FAISS index.")
            cls.store.load_index()
        cls.responder = ChatbotResponder(cls.store)

    def test_prompt_formatting(self):
        responder = self.responder
        user_input = "How can I manage stress better?"
        chat_history = "USER: I've been feeling stressed lately.\nBOT: It's important to take breaks."
        context = "Stress management techniques include mindfulness, exercise, and talking to someone."

        formatted_prompt = responder.format_prompt(
            user_input=user_input, chat_history=chat_history, context=context
        )
        expected_prompt = (
            f"USER: {user_input}\n"
            f"{chat_history}\n"
            f"CONTEXT: {context}\n"
            "BOT:"
        )
        self.assertEqual(formatted_prompt, expected_prompt.strip())

    def test_chat_bot(self):
        responder = self.responder
        user_input = "I'm feeling overwhelmed and anxious lately."
        chat_history = "USER: I've been struggling at work.\nBOT: I'm sorry to hear that. Do you want to talk about it?"

        response = responder.generate_response(user_input, chat_history)
        self.assertIsNotNone(response)
        print("Response:", response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)

    @unittest.skip("Skipping test for empty user input as it may not be meaningful.")
    def test_empty_user_input(self):
        responder = self.responder
        response = responder.generate_response("", "")
        self.assertIsInstance(response, str)
        self.assertGreaterEqual(
            len(response.strip()), 0
        )  # May be empty, but shouldn't crash

    @unittest.skip("Skipping test for empty user input as it may not be meaningful.")
    def test_large_user_input(self):
        responder = self.responder
        long_input = "I'm anxious. " * 500  # Creates a long string > 1024 tokens
        response = responder.generate_response(long_input, "")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)

    def test_offensive_input(self):
        responder = self.responder
        offensive_input = (
            "I hate everyone. I want to punch a wall and scream. Life is stupid."
        )
        response = responder.generate_response(offensive_input, "")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)
        # Optionally check if bot avoids escalating language
        self.assertNotIn("stupid", response.lower())

    def test_multi_turn_conversation(self):
        responder = self.responder
        history = [("", "")]
        turns = [
            "I'm feeling really down lately.",
            "Nothing seems to help me anymore.",
            "Is there even a point to trying again?",
        ]

        for turn in turns:
            history_str = "\n".join([f"USER: {u}\nBOT: {b}" for u, b in history])
            bot_response = responder.generate_response(turn, history_str)
            history.append((turn, bot_response))

        final_response = history[-1][1]
        self.assertIsInstance(final_response, str)
        self.assertGreater(len(final_response.strip()), 0)
        print("Multi-turn Final Response:\n", final_response)
