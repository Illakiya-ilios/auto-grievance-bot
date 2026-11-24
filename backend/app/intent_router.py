from __future__ import annotations

from typing import Literal


ChatRoute = Literal["greeting", "grievance", "tracking", "escalation", "document_question"]


def route_intent(intent: str) -> ChatRoute:
    if intent == "greeting":
        return "greeting"
    if intent == "registration":
        return "grievance"
    if intent == "tracking":
        return "tracking"
    if intent == "escalation":
        return "escalation"
    if intent == "general":
        return "document_question"
    raise ValueError(f"Unsupported intent: {intent}")
