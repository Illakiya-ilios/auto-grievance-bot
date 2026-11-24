from app.workflow import extract_local_details, workflow as graph


class TestLLM:
    def extract_details(self, message):
        return {}

    def generate_reply(self, message, intent, instruction, context=None, documents=None):
        field = (context or {}).get("missing_field")
        if field:
            if field == "recipient":
                return "Please provide the person or department that should receive the grievance."
            return f"Please provide {field.replace('_', ' ')}."
        if "draft is ready for review" in instruction:
            return "Please review the details. Nothing will be submitted until you approve."
        return "Please review the details before continuing."

    def follow_up_question(self, description, details, documents=None):
        return None


class WorkflowWithTestLLM:
    def invoke(self, state, config=None):
        state = dict(state)
        state.setdefault("intent", "registration")
        config = dict(config or {})
        configurable = dict(config.get("configurable", {}))
        configurable.setdefault("llm_service", TestLLM())
        config["configurable"] = configurable
        return graph.invoke(state, config=config)


workflow = WorkflowWithTestLLM()


def test_workflow_requires_llm_classified_intent():
    try:
        graph.invoke(
            {"message": "I need to escalate an unresolved grievance."},
            config={"configurable": {"thread_id": "missing-intent"}},
        )
    except RuntimeError as exc:
        assert "LLM-classified intent" in str(exc)
    else:
        raise AssertionError("The workflow must not locally classify intents.")


def test_registration_keeps_prior_details_and_asks_only_for_missing_fields():
    first = workflow.invoke({
        "message": "I want to register a grievance. My service request has had no response for ten days.",
    }, config={"configurable": {"thread_id": "registration-details"}})

    assert first["intent"] == "registration"
    assert "action" not in first
    assert first["details"]["description"]
    assert "name" in first["missing"]

    second = workflow.invoke({
        "message": "My name is Arun",
        "intent": "registration",
        "details": first["details"],
    }, config={"configurable": {"thread_id": "registration-details"}})

    assert second["intent"] == "registration"
    assert second["details"]["description"] == first["details"]["description"]
    assert second["details"]["name"] == "Arun"
    assert "contact" in second["missing"]
    assert "description" not in second["missing"]


def test_registration_request_is_not_mistaken_for_the_complaint_description():
    result = workflow.invoke(
        {"message": "I want to register a grievance."},
        config={"configurable": {"thread_id": "registration-intent"}},
    )

    assert "description" not in result["details"]
    assert result["missing"][0] == "description"
    assert "description" in result["missing"]
    assert "action" not in result


def test_grievance_category_is_not_assigned_a_hardcoded_default():
    details = extract_local_details("The water service is interrupted in North Ward.", {})

    assert "category" not in details


def test_local_extraction_keeps_mobile_and_email_separate():
    details = extract_local_details(
        "My name is Mira, email mira@example.com, phone +1 555 123 4567.",
        {},
    )

    assert details["name"] == "Mira"
    assert details["email"] == "mira@example.com"
    assert details["mobile_number"] == "+1 555 123 4567"
    assert details["contact"] == "mira@example.com"


def test_short_answers_fill_the_field_being_requested():
    result = workflow.invoke({
        "message": "Arun",
        "intent": "registration",
        "details": {"description": "My service was not completed."},
        "awaiting": "name",
    }, config={"configurable": {"thread_id": "short-answer"}})

    assert result["details"]["name"] == "Arun"
    assert result["missing"][0] == "contact"


def test_procedure_question_does_not_route_to_intake_or_module_menu():
    result = workflow.invoke(
        {"message": "How do I file a grievance?", "intent": "general"},
        config={"configurable": {"thread_id": "procedure-question"}},
    )

    assert result["intent"] == "general"
    assert "response" not in result


def test_question_about_grievances_is_not_mistaken_for_a_filing_request():
    result = workflow.invoke(
        {"message": "What is a grievance?", "intent": "general"},
        config={"configurable": {"thread_id": "grievance-definition"}},
    )

    assert result["intent"] == "general"


def test_unrecognized_question_is_left_for_llm_to_answer():
    result = workflow.invoke(
        {"message": "Who was Ada Lovelace?", "intent": "general"},
        config={"configurable": {"thread_id": "general-question"}},
    )

    assert result["intent"] == "general"
    assert "response" not in result


def test_greeting_is_left_for_llm_to_answer():
    result = workflow.invoke(
        {"message": "hi", "intent": "general"},
        config={"configurable": {"thread_id": "friendly-greeting"}},
    )

    assert result["intent"] == "general"
    assert "response" not in result


def test_follow_up_question_is_returned_without_local_response_text():
    result = workflow.invoke(
        {
            "message": "I paid the electricity bill twice.",
            "intent": "registration",
            "follow_up_question": "When did the duplicate payment happen?",
        },
        config={"configurable": {"thread_id": "acknowledge-before-follow-up"}},
    )

    assert result["response"] == "When did the duplicate payment happen?"


def test_registration_can_ask_another_follow_up_after_the_first_answer():
    first = workflow.invoke(
        {
            "message": "I paid the electricity bill twice.",
            "intent": "registration",
            "follow_up_question": "When did the duplicate payment happen?",
        },
        config={"configurable": {"thread_id": "multiple-follow-ups"}},
    )

    second = workflow.invoke(
        {
            "message": "It happened last Friday.",
            "intent": "registration",
            "details": first["details"],
            "awaiting": first["awaiting"],
            "follow_up_question": "Do you have a transaction reference?",
        },
        config={"configurable": {"thread_id": "multiple-follow-ups"}},
    )

    assert second["details"]["supporting_information"] == "It happened last Friday."
    assert second["awaiting"] == "supporting_information"
    assert second["response"].endswith("Do you have a transaction reference?")


def test_registration_only_marks_complete_details_for_review():
    result = workflow.invoke({
        "message": "My name is Arun, arun@example.com",
        "intent": "registration",
        "details": {
            "category": "Service complaint",
            "subject": "Delayed response",
            "description": "No response to service request for ten days.",
            "recipient": "Public Works Department",
        },
        }, config={"configurable": {"thread_id": "complete-review"}})

    assert result["missing"] == []
    assert result["action"] == "review"
    assert "Nothing will be submitted" in result["response"]


def test_registration_collects_grievance_recipient_before_review():
    first = workflow.invoke({
        "message": "My name is Arun, arun@example.com",
        "intent": "registration",
        "details": {
            "category": "Service complaint",
            "subject": "Delayed response",
            "description": "No response to service request for ten days.",
        },
    }, config={"configurable": {"thread_id": "recipient-required"}})

    assert first["missing"] == ["recipient"]
    assert "person or department" in first["response"]

    second = workflow.invoke({
        "message": "Public Works Department",
        "intent": "registration",
        "details": first["details"],
        "awaiting": first["awaiting"],
    }, config={"configurable": {"thread_id": "recipient-required"}})

    assert second["details"]["recipient"] == "Public Works Department"
    assert second["missing"] == []
    assert second["action"] == "review"
