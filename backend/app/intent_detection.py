from __future__ import annotations

from typing import Literal, Protocol

from app.llm import LLMServiceError


Intent = Literal["registration", "tracking", "escalation", "general", "greeting"]


class IntentClassifier(Protocol):
    def classify_intent(
        self,
        message: str,
        *,
        session_mode: str | None = None,
        has_grievance_context: bool = False,
    ) -> str: ...


def detect_message_intent(
    message: str,
    classifier: IntentClassifier,
    *,
    session_mode: str | None = None,
    has_grievance_context: bool = False,
) -> Intent:
    intent = classifier.classify_intent(
        message,
        session_mode=session_mode,
        has_grievance_context=has_grievance_context,
    )
    if intent == "registration":
        return "registration"
    if intent == "tracking":
        return "tracking"
    if intent == "escalation":
        return "escalation"
    if intent == "general":
        return "general"
    if intent == "greeting":
        return "greeting"
    raise LLMServiceError("Gemini returned an invalid intent.")
