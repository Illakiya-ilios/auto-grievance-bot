import uuid

import pytest
from fastapi.testclient import TestClient

from app import main
from app.intent_detection import detect_message_intent
from app.intent_router import route_intent


class FixedClassifier:
    def __init__(self, result):
        self.result = result

    def classify_intent(self, message, **kwargs):
        return self.result


def test_intent_detection_uses_llm_for_grievance_messages():
    assert detect_message_intent("The chair is not working.", FixedClassifier("registration")) == "registration"


def test_model_classification_is_used_for_tracking_intent():
    assert detect_message_intent(
        "The television has a large screen.",
        FixedClassifier("tracking"),
    ) == "tracking"


def test_document_procedure_question_uses_model_general_intent():
    assert detect_message_intent(
        "How do I file a claim?",
        FixedClassifier("general"),
    ) == "general"


def test_escalation_request_with_unresolved_text_is_not_general():
    assert detect_message_intent(
        "I need to escalate an unresolved grievance.",
        FixedClassifier("escalation"),
    ) == "escalation"


def test_registration_session_keeps_follow_up_answers_in_grievance_route():
    assert detect_message_intent(
        "The chair is in room 3.",
        FixedClassifier("registration"),
        session_mode="registration",
    ) == "registration"


def test_tracking_request_exits_active_registration_without_contact_validation(monkeypatch):
    monkeypatch.setattr(
        main.store,
        "list_knowledge_documents",
        lambda: [{"filename": "service-guide.txt", "content": "Service grievance policy."}],
    )
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {"category": "Facility complaint"})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    client = TestClient(main.app)

    start = client.post("/api/chat", json={"message": "The chair is not working."})
    conversation_id = start.json()["conversation_id"]
    tracking = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I want to track a grievance.",
    })

    assert start.status_code == 200
    assert "what name should i put" in start.json()["message"].lower()
    assert tracking.status_code == 200
    assert tracking.json()["intent"] == "tracking"
    assert "grievance id" in tracking.json()["message"].lower()
    assert "contact" not in main.conversations[conversation_id]["details"]


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        ("greeting", "greeting"),
        ("registration", "grievance"),
        ("tracking", "tracking"),
        ("escalation", "escalation"),
        ("general", "document_question"),
    ],
)
def test_intent_router_selects_processing_path(intent, expected_route):
    assert route_intent(intent) == expected_route


def test_tracking_route_does_not_require_knowledge_documents(monkeypatch):
    def unexpected_document_lookup():
        raise AssertionError("Tracking existing grievances should not need knowledge documents.")

    monkeypatch.setattr(main.store, "list_knowledge_documents", unexpected_document_lookup)
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "I want to track a grievance."})

    assert response.status_code == 200
    assert response.json()["intent"] == "tracking"
    assert "grievance ID" in response.json()["message"]


@pytest.mark.parametrize("completed_session", [False, True])
def test_stale_or_completed_conversation_id_is_replaced(completed_session):
    conversation_id = str(uuid.uuid4())
    if completed_session:
        main.conversations[conversation_id] = {
            "details": {},
            "grievance_id": "GRV-2026-A1B2C3D4",
        }
    client = TestClient(main.app)

    response = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I want to track a grievance.",
    })

    assert response.status_code == 200
    assert response.json()["conversation_id"] != conversation_id
    assert response.json()["intent"] == "tracking"


def test_good_evening_is_routed_as_a_greeting(monkeypatch):
    monkeypatch.setattr(
        main.store,
        "list_knowledge_documents",
        lambda: (_ for _ in ()).throw(AssertionError("Greetings should not load documents.")),
    )
    monkeypatch.setattr(main.llm, "answer_greeting", lambda message: "Hi, how can I help you today?")
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "good evening"})

    assert response.status_code == 200
    assert response.json()["message"] == "Hi, how can I help you today?"
