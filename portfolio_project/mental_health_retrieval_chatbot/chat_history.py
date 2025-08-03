from typing import List


class ChatHistory:
    def __init__(self, max_turns: int = 1, include_bot: bool = False):
        """
        max_turns: how many previous turns to keep
        include_bot: whether to include BOT responses in the formatted history
        """
        self.max_turns = max_turns
        self.include_bot = include_bot
        self.history: List[tuple[str, str]] = []  # (user, bot)

    def add_turn(self, user_input: str, bot_response: str):
        self.history.append((user_input, bot_response))
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def format(self) -> str:
        """Format history for the prompt"""
        if self.include_bot:
            return "\n".join([f"{u}\nbot: {b}" for u, b in self.history])
        else:
            # Only user turns
            return "\n".join([f"{u}" for u, _ in self.history])

    def reset(self):
        self.history = []
