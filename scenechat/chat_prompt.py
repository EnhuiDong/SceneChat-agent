"""Chat-native prompts that retain the legacy string interface and byte budget."""
from __future__ import annotations


class ChatPrompt(str):
    """One rendering for logging/budgets, separate messages for chat providers.

    No fabricated assistant turns: historical statements remain quoted user data.
    Appending a correction keeps the original system instructions and references.
    """

    def __new__(cls, messages):
        messages = tuple((str(role), str(content)) for role, content in messages if content)
        value = "\n\n".join(content for _, content in messages)
        instance = super().__new__(cls, value)
        instance.messages = messages
        return instance

    def __add__(self, correction):
        if not isinstance(correction, str):
            return NotImplemented
        return ChatPrompt((*self.messages, ("user", correction)))

    def __getnewargs__(self):
        return (self.messages,)
