from typing import Optional
from .chat_history import ChatHistory


class PromptBuilder:
    def __init__(self, template: str = None):
        self.template = (
            "I am English speaking only mental health chat companion meant to provide guidance on mental health topics through informing and providing support. Given CONTEXT and CHAT_HISTORY I respond to the users USER_INPUT in a thoughtful, considerate, kind, understanding, sensitive manner while providing suggestions and advice based on the CONTEXT."
            if template is None
            else template
        )

    def build(
        self,
        user_input: str,
        chat_history: Optional[str | ChatHistory] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt_parts = [self.template.strip()]

        prompt_parts.append(f"USER:{user_input}")

        if chat_history:
            prompt_parts.append(
                "HISTORY:" + chat_history.format()
                if isinstance(chat_history, ChatHistory)
                else chat_history.strip()
            )

        if context:
            prompt_parts.append("CONTEXT:" + context.strip())

        return "\n".join(prompt_parts).strip() + "\nRESPONSE:"
