from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(autouse=True)
def uploaded_knowledge_documents_by_default(monkeypatch):
    monkeypatch.setattr(
        main.store,
        "list_knowledge_documents",
        lambda: [{"filename": "service-guide.txt", "content": "Service grievance policy and process."}],
    )


def _prepare_review(client: TestClient, monkeypatch) -> str:
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {"category": "Service complaint"})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    first = client.post("/api/chat", json={
        "message": "I have not received a response to my service request for 10 days.",
    })
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    assert "what name should i put" in first.json()["message"].lower()
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Request number 7712. My name is Arun and email arun@example.com",
    })
    assert second.status_code == 200
    third = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Public Works Department",
    })
    assert third.status_code == 200
    assert "impact" in third.json()["message"].lower()
    fourth = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "The delay prevented me from getting the service I needed.",
    })
    assert "expected resolution" in fourth.json()["message"].lower()
    fifth = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I want the service request completed.",
    })
    assert fifth.status_code == 200
    assert fifth.json()["review"]["name"] == "Arun"
    assert fifth.json()["review"]["recipient"] == "Public Works Department"
    return conversation_id


def test_registration_request_prompts_for_a_real_issue_before_llm_extraction(monkeypatch):
    extraction_calls = []
    monkeypatch.setattr(
        main.llm,
        "extract_details",
        lambda message: extraction_calls.append(message) or {
            "description": "The user wants to register a grievance.",
        },
    )
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "I want to register a grievance."})

    conversation_id = response.json()["conversation_id"]
    assert response.status_code == 200
    assert "please describe what happened" in response.json()["message"].lower()
    assert "description" not in main.conversations[conversation_id]["details"]
    assert extraction_calls == []


def test_user_issue_text_takes_precedence_over_generated_description(monkeypatch):
    document = {"filename": "facility-guide.txt", "content": "Report damaged furniture to the facilities manager."}
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [document])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {
        "description": "The user wants to register a grievance.",
        "category": "General Grievance",
    })
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    client = TestClient(main.app)

    intro = client.post("/api/chat", json={"message": "I want to register a grievance."})
    conversation_id = intro.json()["conversation_id"]
    complaint = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Chair is not working.",
    })

    assert complaint.status_code == 200
    assert main.conversations[conversation_id]["details"]["description"] == "Chair is not working."
    assert main.conversations[conversation_id]["details"]["subject"] == "Chair is not working."


def test_mobile_and_submission_fields_are_not_duplicated_as_supporting_information():
    details = {
        "name": "Nivetha",
        "contact": "7200677501",
        "mobile_number": "7200677501",
        "submission_type": "Grievance Submission",
        "category": "Facility complaint",
        "subject": "Chair is not working",
        "description": "Chair is not working.",
        "recipient": "Manager",
    }

    main._consolidate_generic_template_details(details)

    assert not details.get("supporting_information")
    assert details["mobile_number"] == "7200677501"
    assert details["submission_type"] == "Grievance Submission"


def test_generic_template_details_are_included_in_review_information(monkeypatch):
    monkeypatch.setattr(
        main.llm,
        "extract_details",
        lambda message: {
            "category": "Water service",
            "location": "North Ward",
            "expected_resolution": "Repair the pipe",
        },
    )
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    client = TestClient(main.app)

    first = client.post("/api/chat", json={
        "message": "I want to register a grievance. The water pipe has been leaking.",
    })
    conversation_id = first.json()["conversation_id"]
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "My name is Mira and mira@example.com",
    })
    third = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Water Department",
    })
    fourth = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "The leak is damaging the kitchen floor.",
    })

    assert second.status_code == 200
    assert third.status_code == 200
    assert "impact" in third.json()["message"].lower()
    assert "Location: North Ward" in fourth.json()["review"]["supporting_information"]
    assert "Expected resolution: Repair the pipe" in fourth.json()["review"]["supporting_information"]
    assert "Impact: The leak is damaging the kitchen floor." in fourth.json()["review"]["supporting_information"]
    assert "location" not in fourth.json()["review"]


def test_document_context_is_preserved_for_issue_follow_up_questions(monkeypatch):
    document = {"filename": "billing-guide.txt", "content": "Duplicate charges can be disputed within 30 days."}
    received_documents = []
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [document])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})

    def follow_up_question(description, details, documents):
        received_documents.append(documents)
        return "When did you notice the duplicate charge?"

    monkeypatch.setattr(main.llm, "follow_up_question", follow_up_question)
    client = TestClient(main.app)

    first = client.post("/api/chat", json={"message": "I was charged twice for my bill."})
    conversation_id = first.json()["conversation_id"]
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I noticed it yesterday.",
    })

    assert first.status_code == 200
    assert second.status_code == 200
    assert received_documents == [[document], [document]]


def test_grievance_is_saved_only_after_approval(monkeypatch):
    saved = []

    def save(conversation_id, details):
        saved.append(details)
        return {
            **details,
            "grievance_id": "GRV-2026-1234ABCD",
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc),
            "history": [{"status": "submitted", "note": "Submitted with user approval"}],
        }

    monkeypatch.setattr(main.store, "create_grievance", save)
    client = TestClient(main.app)
    conversation_id = _prepare_review(client, monkeypatch)

    assert saved == []
    response = client.post(f"/api/conversations/{conversation_id}/decision", json={"action": "approve"})

    assert response.status_code == 200
    assert response.json()["grievance"]["grievance_id"] == "GRV-2026-1234ABCD"
    assert "send it to public works department" in response.json()["message"].lower()
    assert len(saved) == 1


def test_clear_problem_description_is_not_overridden_by_general_model_intent(monkeypatch):
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    monkeypatch.setattr(main.llm, "answer_greeting", lambda message: "Hi, how can I help you today?")
    monkeypatch.setattr(main.llm, "answer_general", lambda question, documents: "Hi! What would you like help with today?")
    client = TestClient(main.app)
    greeting = client.post("/api/chat", json={"message": "hi"})
    conversation_id = greeting.json()["conversation_id"]

    response = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I was charged twice for my bill.",
    })

    assert response.status_code == 200
    assert response.json()["intent"] == "registration"
    assert "what name should i put" in response.json()["message"].lower()


def test_plain_name_answer_advances_to_contact_question(monkeypatch):
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    monkeypatch.setattr(main.llm, "answer_greeting", lambda message: "Hi, how can I help you today?")
    monkeypatch.setattr(main.llm, "answer_general", lambda question, documents: "Hi! What would you like help with today?")
    client = TestClient(main.app)
    greeting = client.post("/api/chat", json={"message": "hi"})
    conversation_id = greeting.json()["conversation_id"]

    complaint = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I paid my bill twice.",
    })
    assert "what name should i put" in complaint.json()["message"].lower()

    name = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Mira",
    })

    assert "phone number or email" in name.json()["message"].lower()
    assert main.conversations[conversation_id]["details"]["name"] == "Mira"


def test_follow_up_questions_use_the_users_previous_answer(monkeypatch):
    seen_details = []
    questions = iter(["When did this happen?", "Do you have a transaction reference?"])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})

    def follow_up_question(description, details, documents):
        seen_details.append(dict(details))
        return next(questions)

    monkeypatch.setattr(main.llm, "follow_up_question", follow_up_question)
    client = TestClient(main.app)
    first = client.post("/api/chat", json={"message": "I paid my bill twice."})
    conversation_id = first.json()["conversation_id"]
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "It happened last Friday.",
    })

    assert "When did this happen?" in first.json()["message"]
    assert "Do you have a transaction reference?" in second.json()["message"]
    assert seen_details[1]["supporting_information"] == "It happened last Friday."


def test_declining_follow_up_prompts_for_a_different_useful_detail(monkeypatch):
    follow_up_calls = []
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})

    def follow_up_question(description, details, documents):
        follow_up_calls.append(dict(details))
        if details.get("_follow_up_declined") == "true":
            return "Which store or service provider was involved?"
        return "What was the nature of the order?"

    monkeypatch.setattr(main.llm, "follow_up_question", follow_up_question)
    client = TestClient(main.app)
    first = client.post("/api/chat", json={
        "message": "I want to file a grievance. The table and chairs were not provided.",
    })
    conversation_id = first.json()["conversation_id"]
    declined = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "no",
    })

    assert "What was the nature of the order?" in first.json()["message"]
    assert "Which store or service provider was involved?" in declined.json()["message"]
    assert len(follow_up_calls) == 2
    assert "supporting_information" not in main.conversations[conversation_id]["details"]


def test_issue_questions_finish_before_contact_and_invalid_phone_is_reprompted(monkeypatch):
    follow_up_calls = []
    questions = iter([
        "Where is the non-working street light?",
        None,
    ])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})

    def follow_up_question(description, details, documents):
        follow_up_calls.append(details)
        return next(questions)

    monkeypatch.setattr(main.llm, "follow_up_question", follow_up_question)
    client = TestClient(main.app)
    first = client.post("/api/chat", json={"message": "Our village street light is not working."})
    conversation_id = first.json()["conversation_id"]
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Near the temple in Mahabalipuram.",
    })
    name = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Mira",
    })
    invalid_phone = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "32235",
    })

    assert "Where is the non-working street light?" in first.json()["message"]
    assert "What name should I put" in second.json()["message"]
    assert "phone number or email" in name.json()["message"].lower()
    assert "10-digit phone number" in invalid_phone.json()["message"]
    assert main.conversations[conversation_id]["awaiting"] == "contact"
    assert "contact" not in main.conversations[conversation_id]["details"]
    assert len(follow_up_calls) == 2


def test_similar_follow_up_question_is_not_asked_again(monkeypatch):
    questions = iter([
        "Can you specify the location of the non-working street light?",
        "What is the specific location of the non-working street light?",
    ])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: next(questions))
    client = TestClient(main.app)
    first = client.post("/api/chat", json={
        "message": "Our village street light is not working.",
    })
    conversation_id = first.json()["conversation_id"]
    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Chengalpattu",
    })

    assert "Can you specify the location" in first.json()["message"]
    assert "specific location of the non-working street light" not in second.json()["message"]
    assert "What name should I put" in second.json()["message"]
    assert len(main.conversations[conversation_id]["follow_up_questions"]) == 1


def test_user_can_ask_a_general_question_during_grievance_intake(monkeypatch):
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    monkeypatch.setattr(main.llm, "answer_general", lambda question, documents: "A grievance is a formal complaint.")
    client = TestClient(main.app)
    complaint = client.post("/api/chat", json={"message": "I paid my bill twice."})
    conversation_id = complaint.json()["conversation_id"]

    answer = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "What is a grievance?",
    })

    assert answer.status_code == 200
    assert answer.json()["intent"] == "general"
    assert answer.json()["message"] == "A grievance is a formal complaint."


def test_unrelated_question_during_intake_uses_llm_document_answer(monkeypatch):
    document = {"filename": "streetlight-guide.txt", "content": "Report a streetlight fault to the service department."}
    monkeypatch.setattr(main, "CUSTOMER_CARE_CONTACT", "")
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [document])
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    monkeypatch.setattr(
        main.llm,
        "answer_general",
        lambda question, documents: "The reference documents do not provide that information.",
    )
    client = TestClient(main.app)

    complaint = client.post("/api/chat", json={"message": "Our street light is not working."})
    conversation_id = complaint.json()["conversation_id"]
    unrelated = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Who won the football match?",
    })

    assert unrelated.status_code == 200
    assert unrelated.json()["intent"] == "general"
    assert unrelated.json()["message"] == "The reference documents do not provide that information."
    assert main.conversations[conversation_id]["details"]["description"] == "Our street light is not working."


def test_frustration_does_not_start_escalation_when_model_misclassifies_it(monkeypatch):
    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    monkeypatch.setattr(main.llm, "classify_intent", lambda message, **kwargs: "general")
    monkeypatch.setattr(main.llm, "answer_general", lambda question, documents: "I'm sorry. What help do you need?")
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Don't help me."})

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert response.json()["message"] == "I'm sorry. What help do you need?"
    assert main.conversations[response.json()["conversation_id"]]["mode"] is None


def test_repeated_invalid_contact_offers_customer_care_instructions(monkeypatch):
    details = {
        "name": "Arun",
        "contact": "not-a-valid-contact",
        "recipient": "Public Works Department",
        "category": "Service complaint",
        "subject": "Duplicate payment",
        "description": "I paid the electricity bill twice.",
        "impact": "The duplicate charge reduced my available balance.",
        "expected_resolution": "Refund the extra payment.",
    }
    monkeypatch.setattr(main.llm, "extract_details", lambda message: details)
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, known, documents: None)
    monkeypatch.setattr(main, "CUSTOMER_CARE_CONTACT", "1800-555-0100")
    client = TestClient(main.app)

    first = client.post("/api/chat", json={"message": "I paid my bill twice."})
    conversation_id = first.json()["conversation_id"]
    assert "phone number or email" in first.json()["message"]

    second = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "That contact is still incorrect.",
    })
    assert "Would you like customer-care contact instructions?" in second.json()["message"]

    consent = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Yes, please.",
    })

    assert consent.json()["intent"] == "customer_care"
    assert "1800-555-0100" in consent.json()["message"]
    assert "transfer this chat directly" in consent.json()["message"]


def test_general_question_fails_when_gemini_is_unavailable(monkeypatch):
    monkeypatch.setattr(main.llm, "client", None)
    monkeypatch.setattr(
        main.llm,
        "classify_intent",
        lambda message, **kwargs: main.LLMService.classify_intent(main.llm, message, **kwargs),
    )
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Who was Ada Lovelace?"})

    assert response.status_code == 503
    assert "could not classify" in response.json()["detail"]


def test_chat_rejects_blank_messages_and_validation_failures_stop_intake(monkeypatch):
    client = TestClient(main.app)
    blank = client.post("/api/chat", json={"message": "   "})
    assert blank.status_code == 422

    monkeypatch.setattr(main.llm, "extract_details", lambda message: {})

    def assessment_failed(description, details, documents):
        raise main.LLMServiceError("Gemini could not assess the grievance details.")

    monkeypatch.setattr(main.llm, "follow_up_question", assessment_failed)
    response = client.post("/api/chat", json={"message": "Our village street light is not working."})

    assert response.status_code == 503
    assert "could not validate" in response.json()["detail"]


def test_contact_and_name_validation_reject_invalid_values():
    main._validate_contact({"contact": "mira@example.com"})
    main._validate_contact({"contact": "98765-43210"})
    main._validate_name({"name": "Mira O'Neil"})

    with pytest.raises(HTTPException, match="valid 10-digit phone number or email"):
        main._validate_contact({"contact": "--------"})
    with pytest.raises(HTTPException, match="valid 10-digit phone number or email"):
        main._validate_contact({"contact": "987654321"})
    with pytest.raises(HTTPException, match="valid 10-digit phone number or email"):
        main._validate_contact({"contact": "98765432101"})
    with pytest.raises(HTTPException, match="using letters"):
        main._validate_name({"name": "Mira123"})


def test_cancelling_a_review_does_not_save(monkeypatch):
    saved = []
    monkeypatch.setattr(main.store, "create_grievance", lambda *args: saved.append(args))
    client = TestClient(main.app)
    conversation_id = _prepare_review(client, monkeypatch)

    response = client.post(f"/api/conversations/{conversation_id}/decision", json={"action": "cancel"})

    assert response.status_code == 200
    assert "cancelled" in response.json()["message"].lower()
    assert saved == []


def test_invalid_contact_does_not_lock_conversation_in_pending_review(monkeypatch):
    monkeypatch.setattr(
        main.llm,
        "extract_details",
        lambda message: {
            "category": "Service complaint",
            "impact": "The delay prevented me from getting the service.",
            "expected_resolution": "Complete the service request.",
            **({"contact": "not a contact"} if "My name is" in message else {}),
        },
    )
    monkeypatch.setattr(main.llm, "follow_up_question", lambda description, details, documents: None)
    client = TestClient(main.app)
    first = client.post("/api/chat", json={
        "message": "I have not received a response to my service request for 10 days.",
    })
    conversation_id = first.json()["conversation_id"]
    invalid = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "My name is Arun",
    })

    assert invalid.status_code == 200
    assert "person or department" in invalid.json()["message"]

    recipient = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Public Works Department",
    })
    assert recipient.status_code == 200
    assert "phone number or email" in recipient.json()["message"]

    corrected = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "arun@example.com",
    })
    assert corrected.status_code == 200
    assert corrected.json()["review"]["contact"] == "arun@example.com"
    assert corrected.json()["review"]["recipient"] == "Public Works Department"


def test_edit_resumes_graph_into_a_new_review_without_saving(monkeypatch):
    saved = []

    def save(conversation_id, details):
        saved.append(details)
        return {
            **details,
            "grievance_id": "GRV-2026-1234ABCD",
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc),
            "history": [],
        }

    monkeypatch.setattr(main.store, "create_grievance", save)
    client = TestClient(main.app)
    conversation_id = _prepare_review(client, monkeypatch)

    response = client.post(
        f"/api/conversations/{conversation_id}/decision",
        json={"action": "edit", "details": {"subject": "Revised service delay"}},
    )

    assert response.status_code == 200
    assert response.json()["review"]["subject"] == "Revised service delay"
    assert saved == []

    approved = client.post(
        f"/api/conversations/{conversation_id}/decision",
        json={"action": "approve"},
    )

    assert approved.status_code == 200
    assert saved[0]["subject"] == "Revised service delay"
