import re

import pytest

from app import main


@pytest.fixture(autouse=True)
def use_test_llm_responses(monkeypatch):
    def classify_intent(message, *, session_mode=None, has_grievance_context=False):
        lowered = message.lower().strip()
        if re.fullmatch(r"(?:hi|hello|hey|good morning|good afternoon|good evening)[!. ]*", lowered):
            return "greeting"
        if "escalat" in lowered:
            return "escalation"
        if any(term in lowered for term in ("track", "status", "where is", "grv-")):
            return "tracking"
        if lowered.startswith(("who ", "what ", "when ", "where ", "why ", "how ", "can ", "could ")):
            return "general"
        if session_mode == "registration":
            return "registration"
        if any(term in lowered for term in (
            "grievance", "complaint", "not working", "broken", "charged", "paid",
            "delayed", "issue", "problem", "not received", "no response",
        )):
            return "registration"
        return "general"

    def generate_reply(message, intent, instruction, context=None, documents=None):
        context = context or {}
        lowered_instruction = instruction.lower()
        field = context.get("missing_field")
        if field == "description":
            return "Please describe what happened. I’ll ask follow-up questions about the issue."
        if field == "name":
            return "Thanks for explaining that. What name should I put on the grievance?"
        if field == "contact":
            return "What phone number or email can we use to follow up with you?"
        if field == "recipient":
            return "Please provide the person or department that should receive the grievance."
        if field:
            return f"To prepare your review, please provide {field.replace('_', ' ')}."
        if "submission confirmation" in lowered_instruction:
            record = context.get("backend_result", {})
            return f"Your grievance receipt is ready. ID: {record.get('grievance_id')}. Send it to {context.get('recipient')}."
        if "grievance id" in lowered_instruction and "escalation" in intent:
            return "Please share your grievance ID and why the issue remains unresolved."
        if "grievance id" in lowered_instruction:
            return "Please share your grievance ID so I can check its status."
        if "draft is ready for review" in lowered_instruction:
            return "Please review the grievance details below. Nothing will be submitted until you approve."
        if "already submitted" in lowered_instruction:
            return "This grievance has already been submitted."
        if "confirm the escalation" in lowered_instruction:
            return "The grievance has been escalated."
        if "report the grievance status" in lowered_instruction:
            record = context.get("backend_result", {})
            return f"{record.get('grievance_id')} is currently {record.get('status')}."
        if "ask whether the user wants customer-care" in lowered_instruction:
            return "Would you like customer-care contact instructions?"
        if "configured customer-care contact" in lowered_instruction and context.get("customer_care_contact"):
            return f"I can't transfer this chat directly, but you can contact customer care at {context['customer_care_contact']}."
        if "validation issue" in lowered_instruction and "10-digit" in str(context.get("validation_error", "")):
            return "Please enter a valid 10-digit phone number or email address."
        if "validation issue" in lowered_instruction:
            return "Please provide a valid phone number or email address."
        if "reference documents do not provide an answer" in lowered_instruction:
            return "The reference documents do not provide that information. I can help prepare a grievance."
        if "no uploaded reference documents" in lowered_instruction:
            return "I can't support this request because no reference documents are uploaded."
        if "outside the topic covered" in lowered_instruction:
            return "This request is outside the topic covered by the uploaded reference documents."
        if "cancelled" in lowered_instruction:
            return "The draft was cancelled. No grievance was submitted."
        return "I can help with that."

    monkeypatch.setattr(main.llm, "classify_intent", classify_intent)
    monkeypatch.setattr(main.llm, "is_topic_relevant", lambda message, documents, context=None: bool(documents))
    monkeypatch.setattr(main.llm, "generate_reply", generate_reply)
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents=None: None)
    monkeypatch.setattr(
        main.llm,
        "embed_texts",
        lambda texts, task_type: [[0.0] * 768 for _ in texts],
    )
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [])
    monkeypatch.setattr(main.store, "list_unindexed_documents", lambda: [])
    monkeypatch.setattr(
        main.store,
        "search_knowledge_documents",
        lambda embedding, limit=5: main.store.list_knowledge_documents(),
    )
    monkeypatch.setattr(
        main.llm,
        "answer_general",
        lambda question, documents=None: "The reference documents do not provide that information. I can help prepare a grievance.",
    )
