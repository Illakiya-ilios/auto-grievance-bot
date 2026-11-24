from fastapi.testclient import TestClient

from app import main


def test_generic_template_does_not_embed_follow_up_prompts():
    for section in ("grievance_information", "personal_information", "conditional_fields"):
        for field in main.llm.grievance_template[section]:
            assert "follow_up_prompt" not in field


def test_template_management_endpoints_are_removed():
    client = TestClient(main.app)

    assert client.get("/api/templates").status_code == 404
    assert client.get("/api/admin/templates").status_code == 404
    assert client.post("/api/admin/templates", json={}).status_code == 404


def test_chat_no_longer_accepts_admin_template_routing(monkeypatch):
    monkeypatch.setattr(main, "CUSTOMER_CARE_CONTACT", "")
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [])
    client = TestClient(main.app)

    response = client.post(
        "/api/chat",
        json={"message": "My flight was cancelled.", "template_id": "legacy-template-id"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert "no reference documents are uploaded" in response.json()["message"]
