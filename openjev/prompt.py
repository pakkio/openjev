"""The chat-format user message, shared by every scorer (MLX, torch, LoRA) so they all prompt alike.

The context goes in a user turn followed by the option list, and each option is scored as the
model's reply. Showing the options matters: on Gemma 4 E4B zero-shot it lifts claim verification
from 0.48 to 0.89 and BBC section tagging from 0.41 to 0.83, against the bare context alone.
"""
from __future__ import annotations

from typing import Sequence


def chat_message(context: str, options: Sequence[str] | None = None) -> str:
    if not options:
        return context
    return context + "\nOptions: " + "; ".join(options) + "\nAnswer with one option exactly."
