import os
import unittest
from mental_health_retrieval_chatbot.semantic_encoder import SemanticEncoder
from mental_health_retrieval_chatbot.retrieval import MentalHealthVectorStore
from mental_health_retrieval_chatbot.chatbot_responder import ChatbotResponder
from mental_health_retrieval_chatbot.chat_history import ChatHistory
from mental_health_retrieval_chatbot.prompt_builder import PromptBuilder
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

    def test_prompt_builder(self):
        """Ensure prompt builder correctly inserts fields"""
        pb = PromptBuilder()
        history = "I feel lonely a lot."
        context = "Loneliness is common; talking to friends or journaling may help."
        user_input = "What should I do when I feel lonely?"

        prompt = pb.build(user_input, history, context)

        self.assertNotIn("user:", prompt)
        self.assertIn(user_input, prompt)
        self.assertIn(history, prompt)
        self.assertIn(context, prompt)
        print("PromptBuilder output:\n", prompt)

    def test_chat_history_user_only(self):
        """Ensure ChatHistory keeps only user turns when include_bot=False"""
        history = ChatHistory(max_turns=3, include_bot=False)
        history.add_turn("I feel anxious", "You might try deep breathing.")
        history.add_turn("I can’t sleep", "Try a calming routine.")
        history.add_turn("I get panic attacks", "Grounding techniques may help.")

        formatted = history.format()
        self.assertIn("I feel anxious", formatted)
        self.assertNotIn("bot:", formatted)  # should exclude bot
        print("Formatted history (user-only):\n", formatted)

    def test_chat_history_truncation(self):
        """Ensure ChatHistory truncates older turns"""
        history = ChatHistory(max_turns=2, include_bot=False)
        history.add_turn("Turn 1", "Resp 1")
        history.add_turn("Turn 2", "Resp 2")
        history.add_turn("Turn 3", "Resp 3")

        formatted = history.format()
        self.assertNotIn("Turn 1", formatted)  # dropped oldest
        self.assertIn("Turn 2", formatted)
        self.assertIn("Turn 3", formatted)

    def test_generate_response_basic(self):
        """Ensure responder generates a non-empty response"""
        history = ChatHistory(max_turns=3, include_bot=False)
        user_input = "I'm feeling overwhelmed and anxious lately."
        response = self.responder.generate_response(user_input, history)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)
        print("Response:", response)

    def test_generate_response_with_history(self):
        """Test a short multi-turn conversation"""
        history = ChatHistory(max_turns=3, include_bot=False)
        turns = [
            "I'm feeling really down lately.",
            "Nothing seems to help me anymore.",
            "Is there even a point to trying again?",
        ]

        for turn in turns:
            response = self.responder.generate_response(turn, history)
            history.add_turn(turn, response)

        formatted = history.format()
        self.assertTrue(formatted.count("USER:") <= 3)  # max_turns enforced
        print("Final response:", history.history[-1][1])

    def test_offensive_input(self):
        """Ensure bot avoids mirroring offensive input"""
        history = ChatHistory()
        offensive_input = (
            "I hate everyone. I want to punch a wall and scream. Life is stupid."
        )
        response = self.responder.generate_response(offensive_input, history)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)
        self.assertNotIn("stupid", response.lower())  # avoid escalation
        print("Offensive input response:", response)
