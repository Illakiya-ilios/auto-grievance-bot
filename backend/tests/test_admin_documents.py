from datetime import datetime, timezone
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import main


def _text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): writer._add_object(font),
        }),
    })
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _admin_token(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_PASSWORD", "configured-test-password")
    response = client.post("/api/admin/login", json={"password": "configured-test-password"})
    assert response.status_code == 200
    return response.json()["token"]


def test_generated_api_documentation_is_disabled():
    client = TestClient(main.app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_admin_can_upload_list_and_delete_text_documents(monkeypatch):
    client = TestClient(main.app)
    token = _admin_token(client, monkeypatch)
    headers = {"Authorization": f"Bearer {token}"}
    saved = {}
    document = {
        "document_id": "document-1",
        "filename": "service-policy.txt",
        "character_count": 41,
        "created_at": datetime.now(timezone.utc),
    }

    def create_document_with_embeddings(filename, content, chunks, file_data):
        saved.update(filename=filename, content=content, chunks=chunks, file_data=file_data)
        return document

    monkeypatch.setattr(main.store, "create_document_with_embeddings", create_document_with_embeddings)
    monkeypatch.setattr(main.store, "list_documents", lambda: [document])
    monkeypatch.setattr(main.store, "delete_document", lambda document_id: document_id == "document-1")

    upload = client.post(
        "/api/admin/documents",
        headers=headers,
        files={"file": ("service-policy.txt", b"Claims must be reviewed within 30 days.", "text/plain")},
    )
    listing = client.get("/api/admin/documents", headers=headers)
    deletion = client.delete("/api/admin/documents/document-1", headers=headers)

    assert upload.status_code == 200
    assert saved == {
        "filename": "service-policy.txt",
        "content": "Claims must be reviewed within 30 days.",
        "chunks": [(
            "Claims must be reviewed within 30 days.",
            [0.0] * 768,
        )],
        "file_data": b"Claims must be reviewed within 30 days.",
    }
    assert listing.status_code == 200
    assert listing.json()["documents"][0]["filename"] == "service-policy.txt"
    assert deletion.status_code == 200


def test_document_upload_requires_admin_login():
    client = TestClient(main.app)

    response = client.post(
        "/api/admin/documents",
        files={"file": ("service-policy.txt", b"Policy text.", "text/plain")},
    )

    assert response.status_code == 401


def test_pdf_upload_stores_original_bytes_with_extracted_text(monkeypatch):
    client = TestClient(main.app)
    token = _admin_token(client, monkeypatch)
    pdf_data = _text_pdf("Claims must be filed within 30 days.")
    saved = {}
    monkeypatch.setattr(
        main.store,
        "create_document_with_embeddings",
        lambda filename, content, chunks, file_data: (
            saved.update(
                filename=filename,
                content=content,
                chunks=chunks,
                file_data=file_data,
            )
            or {"document_id": "pdf-document"}
        ),
    )

    response = client.post(
        "/api/admin/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("service-policy.pdf", pdf_data, "application/pdf")},
    )

    assert response.status_code == 200
    assert saved["filename"] == "service-policy.pdf"
    assert saved["content"] == "Claims must be filed within 30 days."
    assert saved["file_data"] == pdf_data
    assert saved["chunks"][0][0] == saved["content"]


def test_document_chunking_overlaps_adjacent_chunks():
    content = "a" * (main.KNOWLEDGE_CHUNK_CHARACTERS + 100)

    chunks = main._chunk_document(content)

    assert len(chunks) == 2
    assert chunks[0][-main.KNOWLEDGE_CHUNK_OVERLAP:] == chunks[1][:main.KNOWLEDGE_CHUNK_OVERLAP]
    assert chunks[0] + chunks[1][main.KNOWLEDGE_CHUNK_OVERLAP:] == content


def test_chat_uses_retrieved_documents_as_context(monkeypatch):
    documents = [
        {"filename": "service-policy.pdf", "content": "Claims must be filed within 30 days."},
        {"filename": "contact-guide.txt", "content": "Contact support through the service portal."},
    ]
    received = {}
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: documents)

    def answer_general(question, documents):
        received.update(question=question, documents=documents)
        return "The policy allows 30 days."

    monkeypatch.setattr(main.llm, "answer_general", answer_general)
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Who won the football match?"})

    assert response.status_code == 200
    assert response.json()["message"] == "The policy allows 30 days."
    assert received == {
        "question": "Who won the football match?",
        "documents": documents,
    }


def test_chat_indexes_legacy_documents_and_searches_with_query_embedding(monkeypatch):
    legacy_content = "P" * 2000
    embedding_calls = []
    indexed_chunks = []
    searched = {}
    retrieved = [{"filename": "service-guide.txt", "content": "Relevant policy excerpt."}]
    monkeypatch.setattr(main.store, "list_unindexed_documents", lambda: [{
        "document_id": "legacy-document",
        "filename": "service-guide.txt",
        "content": legacy_content,
    }])

    def embed_texts(texts, task_type):
        embedding_calls.append((texts, task_type))
        return [[float(len(embedding_calls))] * 768 for _ in texts]

    def index_document_chunks(document_id, chunks):
        indexed_chunks.append((document_id, chunks))

    def search_knowledge_documents(embedding, limit):
        searched.update(embedding=embedding, limit=limit)
        return retrieved

    received = {}
    monkeypatch.setattr(main.llm, "embed_texts", embed_texts)
    monkeypatch.setattr(main.store, "index_document_chunks", index_document_chunks)
    monkeypatch.setattr(main.store, "search_knowledge_documents", search_knowledge_documents)
    monkeypatch.setattr(
        main.llm,
        "answer_general",
        lambda question, documents: received.update(question=question, documents=documents)
        or "The relevant policy excerpt is available.",
    )

    response = TestClient(main.app).post("/api/chat", json={"message": "How does the policy work?"})

    assert response.status_code == 200
    assert response.json()["message"] == "The relevant policy excerpt is available."
    assert [task_type for _, task_type in embedding_calls] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]
    assert indexed_chunks[0][0] == "legacy-document"
    assert len(indexed_chunks[0][1]) == len(main._chunk_document(legacy_content))
    assert (
        indexed_chunks[0][1][0][0][-main.KNOWLEDGE_CHUNK_OVERLAP:]
        == indexed_chunks[0][1][1][0][:main.KNOWLEDGE_CHUNK_OVERLAP]
    )
    assert searched == {"embedding": [2.0] * 768, "limit": main.KNOWLEDGE_SEARCH_LIMIT}
    assert received == {
        "question": "How does the policy work?",
        "documents": retrieved,
    }


def test_chat_rejects_knowledge_context_over_limit_without_truncating(monkeypatch):
    oversized_content = "x" * (main.MAX_KNOWLEDGE_CONTEXT_CHARACTERS + 1)
    monkeypatch.setattr(
        main.store,
        "list_knowledge_documents",
        lambda: [{"filename": "oversized.txt", "content": oversized_content}],
    )
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Who won the football match?"})

    assert response.status_code == 413
    assert "Remove documents" in response.json()["detail"]


def test_chat_answers_greeting_without_checking_uploaded_documents(monkeypatch):
    def unexpected_document_lookup():
        raise AssertionError("A greeting should not require uploaded documents.")

    monkeypatch.setattr(main.store, "list_knowledge_documents", unexpected_document_lookup)
    monkeypatch.setattr(main.llm, "answer_greeting", lambda message: "Hi, how can I help you today?")
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert response.json()["message"] == "Hi, how can I help you today?"


def test_chat_uses_llm_general_intent_to_check_uploaded_documents(monkeypatch):
    documents = [{"filename": "service-guide.txt", "content": "Contact support through the portal."}]
    received = {}
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: documents)
    monkeypatch.setattr(main.llm, "classify_intent", lambda message, **kwargs: "general")

    def answer_general(question, context):
        received.update(question=question, documents=context)
        return "The reference documents do not provide that information."

    monkeypatch.setattr(main.llm, "answer_general", answer_general)
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "TV is working."})

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert response.json()["message"] == "The reference documents do not provide that information."
    assert received == {"question": "TV is working.", "documents": documents}


def test_chat_uses_llm_tracking_intent_without_local_keyword_override(monkeypatch):
    monkeypatch.setattr(main.llm, "classify_intent", lambda message, **kwargs: "tracking")
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "The television has a large screen."})

    assert response.status_code == 200
    assert response.json()["intent"] == "tracking"
    assert "grievance ID" in response.json()["message"]


def test_chat_rejects_topics_when_no_documents_are_uploaded(monkeypatch):
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [])

    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Who won the football match?"})

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert "no reference documents are uploaded" in response.json()["message"]


def test_unrelated_grievance_is_rejected_before_intake(monkeypatch):
    documents = [{"filename": "watch-guide.txt", "content": "Watch servicing and warranty policy."}]
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: documents)
    monkeypatch.setattr(main.llm, "classify_intent", lambda message, **kwargs: "registration")
    monkeypatch.setattr(main.llm, "is_topic_relevant", lambda message, docs, context=None: False)

    def unexpected_extraction(message):
        raise AssertionError("Out-of-topic grievances must not enter grievance intake.")

    monkeypatch.setattr(main.llm, "extract_details", unexpected_extraction)
    response = TestClient(main.app).post("/api/chat", json={"message": "My TV is not working."})

    assert response.status_code == 200
    assert response.json()["intent"] == "general"
    assert "outside the topic" in response.json()["message"]
    assert main.conversations[response.json()["conversation_id"]]["details"] == {}


def test_related_watch_grievance_enters_intake(monkeypatch):
    documents = [{"filename": "watch-guide.txt", "content": "Information about watches, their types, care, and features."}]
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: documents)

    response = TestClient(main.app).post("/api/chat", json={"message": "My watch is not working."})

    assert response.status_code == 200
    assert response.json()["intent"] == "registration"
    assert main.conversations[response.json()["conversation_id"]]["details"]["description"] == "My watch is not working."
    assert "outside the topic" not in response.json()["message"]


def test_registration_follow_up_is_not_rejected_as_out_of_topic(monkeypatch):
    documents = [{"filename": "watch-guide.txt", "content": "Information about watches and their care."}]
    topic_checks = []
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: documents)

    def is_topic_relevant(message, retrieved_documents, context=None):
        topic_checks.append(message)
        return message == "My watch is not working."

    monkeypatch.setattr(main.llm, "is_topic_relevant", is_topic_relevant)
    client = TestClient(main.app)

    first = client.post("/api/chat", json={"message": "My watch is not working."})
    conversation_id = first.json()["conversation_id"]
    response = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "Nivetha",
    })

    assert first.status_code == 200
    assert response.status_code == 200
    assert response.json()["intent"] == "registration"
    assert "outside the topic" not in response.json()["message"]
    assert topic_checks == ["My watch is not working."]


def test_required_impact_and_resolution_are_collected_before_review(monkeypatch):
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [
        {"filename": "laptop-guide.pdf", "content": "Laptop faults and repair information."},
    ])
    client = TestClient(main.app)

    first = client.post("/api/chat", json={"message": "My new laptop is not working."})
    conversation_id = first.json()["conversation_id"]
    answers = [
        "Nivetha",
        "9876543219",
        "Laptop service",
        "Laptop shop",
    ]
    responses = []
    for answer in answers:
        responses.append(client.post("/api/chat", json={
            "conversation_id": conversation_id,
            "message": answer,
        }))

    assert "impact" in responses[-1].json()["message"].lower()
    impact = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I cannot use the laptop for my work.",
    })
    assert "expected resolution" in impact.json()["message"].lower()
    resolution = client.post("/api/chat", json={
        "conversation_id": conversation_id,
        "message": "I want the laptop repaired or replaced.",
    })

    assert first.status_code == 200
    assert all(response.status_code == 200 for response in responses)
    assert resolution.status_code == 200
    assert "review" in resolution.json()
    assert "Impact: I cannot use the laptop for my work." in resolution.json()["review"]["supporting_information"]
    assert "Expected resolution: I want the laptop repaired or replaced." in resolution.json()["review"]["supporting_information"]


def test_unknown_topic_does_not_invent_customer_care_contact(monkeypatch):
    monkeypatch.setattr(main, "CUSTOMER_CARE_CONTACT", "1800-555-0100")
    monkeypatch.setattr(main.store, "list_knowledge_documents", lambda: [])
    client = TestClient(main.app)

    response = client.post("/api/chat", json={"message": "Who won the football match?"})

    assert response.status_code == 200
    assert "1800-555-0100" not in response.json()["message"]


def test_upload_rejects_scanned_pdf_without_extracted_text():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = BytesIO()
    writer.write(pdf)

    try:
        main._extract_document_text("scanned.pdf", pdf.getvalue())
    except main.HTTPException as exc:
        assert exc.status_code == 422
        assert "Scanned PDFs need OCR" in exc.detail
    else:
        raise AssertionError("A PDF without extractable text should be rejected.")


def test_upload_extracts_text_from_pdf():
    filename, text = main._extract_document_text(
        "service-policy.pdf",
        _text_pdf("Claims must be filed within 30 days."),
    )

    assert filename == "service-policy.pdf"
    assert text == "Claims must be filed within 30 days."
