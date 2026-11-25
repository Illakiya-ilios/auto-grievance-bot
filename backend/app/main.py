from __future__ import annotations

import re
import uuid
import os
import secrets
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from app import store
from app.intent_detection import detect_message_intent
from app.intent_router import route_intent
from app.llm import LLMService, LLMServiceError
from app.workflow import (
    REQUIRED_FIELDS,
    extract_local_details,
    workflow,
)


app = FastAPI(
    title="Grievance Chatbot",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
CUSTOMER_CARE_CONTACT = os.getenv("CUSTOMER_CARE_CONTACT", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_TOKEN_TTL_SECONDS = 8 * 60 * 60
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_PAGES = 200
MAX_DOCUMENT_CHARACTERS = 500_000
MAX_KNOWLEDGE_CONTEXT_CHARACTERS = 1_000_000
KNOWLEDGE_CHUNK_CHARACTERS = 1200
KNOWLEDGE_CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 32
KNOWLEDGE_SEARCH_LIMIT = 5
admin_tokens: dict[str, float] = {}
MAX_INVALID_CONTACT_ATTEMPTS = 2


def _chunk_document(content: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + KNOWLEDGE_CHUNK_CHARACTERS, len(content))
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(content):
            break
        start = end - KNOWLEDGE_CHUNK_OVERLAP
    return chunks


def _embed_document_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        embeddings.extend(
            llm.embed_texts(chunks[start : start + EMBEDDING_BATCH_SIZE], "RETRIEVAL_DOCUMENT")
        )
    if len(embeddings) != len(chunks):
        raise LLMServiceError("Gemini returned an invalid number of document embeddings.")
    return embeddings


def _retrieve_knowledge_documents(query: str) -> list[dict[str, str]]:
    for document in store.list_unindexed_documents():
        chunks = _chunk_document(document["content"])
        embeddings = _embed_document_chunks(chunks)
        store.index_document_chunks(
            document["document_id"],
            list(zip(chunks, embeddings)),
        )

    query_embedding = llm.embed_texts([query], "RETRIEVAL_QUERY")[0]
    return store.search_knowledge_documents(query_embedding, KNOWLEDGE_SEARCH_LIMIT)


def _consolidate_generic_template_details(details: dict[str, str]) -> None:
    standard_fields = {
        "name", "contact", "recipient", "category", "subject", "description",
        "related_date", "supporting_information", "mobile_number", "email",
        "submission_type",
    }
    additional_information = []
    for section in ("grievance_information", "personal_information", "conditional_fields"):
        for field in llm.grievance_template.get(section, []):
            key = field["key"]
            value = details.get(key, "").strip()
            if key in standard_fields or not value:
                continue
            additional_information.append(f"{field['field']}: {value}")
            details.pop(key, None)

    existing_information = details.get("supporting_information", "").strip()
    details["supporting_information"] = "\n".join(
        information for information in [existing_information, *additional_information] if information
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm = LLMService()
conversations: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class DecisionRequest(BaseModel):
    action: Literal["approve", "edit", "cancel"]
    details: dict[str, str] | None = None


class EscalationRequest(BaseModel):
    reason: str = Field(min_length=8, max_length=2000)


class AdminLoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def _require_admin(authorization: str | None = Header(default=None)) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Admin login is required.")
    expires_at = admin_tokens.get(token)
    if expires_at is None or expires_at <= time.time():
        admin_tokens.pop(token, None)
        raise HTTPException(status_code=401, detail="Admin session expired. Please log in again.")
    return token


def _public_grievance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "grievance_id": record["grievance_id"],
        "category": record["category"],
        "recipient": record["recipient"],
        "subject": record["subject"],
        "description": record["description"],
        "supporting_information": record.get("supporting_information"),
        "submitted_at": record["submitted_at"].isoformat() if record.get("submitted_at") else None,
        "status": record["status"],
        "history": record.get("history", []),
    }


def _unsupported_topic_reply(message: str) -> str:
    return _assistant_reply(
        message,
        "general",
        "No uploaded reference documents are available to establish a supported topic. Explain that this request cannot be answered or registered until relevant reference material is uploaded. Do not invent a customer-care transfer.",
    )


def _assistant_reply(
    message: str,
    intent: str,
    instruction: str,
    context: dict[str, Any] | None = None,
    documents: list[dict[str, str]] | None = None,
) -> str:
    try:
        return llm.generate_reply(message, intent, instruction, context, documents)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini could not generate a response. Please try again later.",
        ) from exc


def _validate_contact(details: dict[str, str]) -> None:
    contact = details.get("contact", "").strip()
    if not contact:
        return
    if len(contact) <= 254 and "@" in contact:
        local, _, domain = contact.rpartition("@")
        labels = domain.split(".")
        valid_email = (
            bool(local)
            and len(local) <= 64
            and not local.startswith(".")
            and not local.endswith(".")
            and ".." not in local
            and bool(re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local))
            and len(labels) >= 2
            and all(
                1 <= len(label) <= 63
                and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            )
        )
        if valid_email:
            return
    digits = re.sub(r"\D", "", contact)
    if re.fullmatch(r"[\d ()-]+", contact) and len(digits) == 10:
        return
    raise HTTPException(status_code=422, detail="Enter a valid 10-digit phone number or email address.")


def _validate_name(details: dict[str, str]) -> None:
    name = details.get("name", "").strip()
    if not name:
        return
    valid_characters = all(
        character.isalpha()
        or character in " '-."
        or unicodedata.category(character).startswith("M")
        for character in name
    )
    if len(name) > 120 or not valid_characters or not any(character.isalpha() for character in name):
        raise HTTPException(status_code=422, detail="Enter a name using letters, spaces, apostrophes, or hyphens.")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/admin/login")
def admin_login(request: AdminLoginRequest) -> dict[str, str]:
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin login is not configured.")
    if not secrets.compare_digest(request.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid admin password.")
    now = time.time()
    for token, expires_at in list(admin_tokens.items()):
        if expires_at <= now:
            admin_tokens.pop(token, None)
    token = secrets.token_urlsafe(32)
    admin_tokens[token] = now + ADMIN_TOKEN_TTL_SECONDS
    return {"token": token}


@app.post("/api/admin/logout")
def admin_logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    token = _require_admin(authorization)
    admin_tokens.pop(token, None)
    return {"message": "Signed out."}


def _extract_document_text(filename: str, data: bytes) -> tuple[str, str]:
    safe_filename = re.split(r"[\\/]", filename)[-1].strip()[:255]
    if not safe_filename:
        raise HTTPException(status_code=422, detail="Choose a document with a filename.")

    extension = Path(safe_filename).suffix.lower()
    if extension == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Text documents must use UTF-8 encoding.") from exc
    elif extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF.")
        try:
            reader = PdfReader(BytesIO(data), strict=False)
            if len(reader.pages) > MAX_DOCUMENT_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF documents can contain no more than {MAX_DOCUMENT_PAGES} pages.",
                )
            pages = []
            character_count = 0
            for page in reader.pages:
                page_text = page.extract_text() or ""
                character_count += len(page_text)
                if character_count > MAX_DOCUMENT_CHARACTERS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Documents can contain no more than {MAX_DOCUMENT_CHARACTERS:,} extracted characters.",
                    )
                pages.append(page_text)
            text = "\n".join(pages)
        except PdfReadError as exc:
            raise HTTPException(status_code=422, detail="The uploaded PDF could not be read.") from exc
    else:
        raise HTTPException(status_code=415, detail="Upload a PDF or UTF-8 text document.")

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No readable text was found. Scanned PDFs need OCR before they can be uploaded.",
        )
    if len(text) > MAX_DOCUMENT_CHARACTERS:
        raise HTTPException(
            status_code=413,
            detail=f"Documents can contain no more than {MAX_DOCUMENT_CHARACTERS:,} extracted characters.",
        )
    return safe_filename, text


@app.get("/api/admin/documents")
def admin_list_documents(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_admin(authorization)
    try:
        documents = store.list_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The document service is temporarily unavailable.") from exc
    return {"documents": documents}


@app.post("/api/admin/documents")
def admin_upload_document(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_admin(authorization)
    data = file.file.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="Documents must be 10 MB or smaller.")
    filename, content = _extract_document_text(file.filename or "", data)
    try:
        chunks = _chunk_document(content)
        embeddings = _embed_document_chunks(chunks)
        document = store.create_document_with_embeddings(
            filename,
            content,
            list(zip(chunks, embeddings)),
            data,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail="The document could not be embedded. Check the Gemini configuration or try again later.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The document could not be saved.") from exc
    return {"document": document}


@app.delete("/api/admin/documents/{document_id}")
def admin_delete_document(
    document_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _require_admin(authorization)
    try:
        deleted = store.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The document could not be deleted.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted."}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")
    conversation_id = request.conversation_id
    if (
        not conversation_id
        or conversation_id not in conversations
        or conversations[conversation_id].get("grievance_id")
    ):
        conversation_id = str(uuid.uuid4())
    session = conversations.setdefault(
        conversation_id,
        {
            "details": {},
            "mode": None,
            "pending": False,
            "awaiting": None,
            "follow_up_declined": False,
            "follow_up_questions": [],
            "documents": [],
        },
    )

    if session.get("pending"):
        return {
            "conversation_id": conversation_id,
            "intent": "registration",
            "message": _assistant_reply(
                message,
                "registration",
                "Remind the user that the draft is pending and ask them to approve, edit, or cancel it.",
                {"known_details": session["details"]},
            ),
            "review": session["details"],
        }

    if session.get("mode") == "customer_care_offer":
        answer = re.sub(r"[.!?,]+", "", message.lower()).strip()
        if answer in {"yes", "y", "yes please", "please connect", "connect me", "customer care"}:
            session["mode"] = None
            response_message = _assistant_reply(
                message,
                "customer_care",
                "Explain that this chat cannot transfer the user. Provide the configured customer-care contact if present; otherwise direct the user to the provider's official contact channel.",
                {"customer_care_contact": CUSTOMER_CARE_CONTACT},
            )
            return {"conversation_id": conversation_id, "intent": "customer_care", "message": response_message}
        if answer in {"no", "n", "no thanks", "not now"}:
            session["mode"] = "registration"
            session["awaiting"] = "contact"
            return {
                "conversation_id": conversation_id,
                "intent": "registration",
                "message": _assistant_reply(
                    message,
                    "registration",
                    "Acknowledge that the user declined customer-care instructions, then ask for a valid phone number or email address to continue.",
                ),
            }
        return {
            "conversation_id": conversation_id,
            "intent": "customer_care",
            "message": _assistant_reply(
                message,
                "customer_care",
                "Ask whether the user wants customer-care contact instructions and request a yes or no answer.",
            ),
        }

    if session.get("mode") == "tracking":
        match = re.search(r"GRV-\d{4}-[A-F0-9]{8}", message, re.I)
        if not match:
            return {"conversation_id": conversation_id, "intent": "tracking", "message": _assistant_reply(
                message, "tracking", "Ask the user to provide the grievance ID to check its status."
            )}
        try:
            record = store.get_grievance(match.group(0))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
        if not record:
            return {"conversation_id": conversation_id, "intent": "tracking", "message": _assistant_reply(
                message, "tracking", "Explain that no grievance matched the supplied ID and ask the user to check it.",
                {"grievance_id": match.group(0)},
            )}
        session["mode"] = None
        public_record = _public_grievance(record)
        return {"conversation_id": conversation_id, "intent": "tracking", "message": _assistant_reply(
            message, "tracking", "Report the grievance status from the backend result without adding other status claims.",
            {"backend_result": public_record},
        ), "grievance": public_record}

    if session.get("mode") == "escalation":
        match = re.search(r"GRV-\d{4}-[A-F0-9]{8}", message, re.I)
        if not match:
            return {"conversation_id": conversation_id, "intent": "escalation", "message": _assistant_reply(
                message, "escalation", "Ask for the grievance ID and a concise reason for escalation."
            )}
        reason = message.replace(match.group(0), "").strip(" ,.;:-")
        if len(reason) < 8:
            return {"conversation_id": conversation_id, "intent": "escalation", "message": _assistant_reply(
                message, "escalation", "Ask the user to briefly explain why the grievance remains unresolved.",
                {"grievance_id": match.group(0)},
            )}
        try:
            record = store.escalate_grievance(match.group(0), reason)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
        session["mode"] = None
        if not record:
            return {"conversation_id": conversation_id, "intent": "escalation", "message": _assistant_reply(
                message, "escalation", "Explain that no grievance matched the supplied ID.",
                {"grievance_id": match.group(0)},
            )}
        if record["status"] != "escalated":
            public_record = _public_grievance(record)
            return {"conversation_id": conversation_id, "intent": "escalation", "message": _assistant_reply(
                message, "escalation", "Explain that this grievance could not be escalated because of its current status. State the supplied status.",
                {"backend_result": public_record},
            )}
        public_record = _public_grievance(record)
        return {"conversation_id": conversation_id, "intent": "escalation", "message": _assistant_reply(
            message, "escalation", "Confirm the escalation only if the backend result shows status escalated.",
            {"backend_result": public_record},
        ), "grievance": public_record}

    has_grievance_context = bool(session.get("details", {}).get("description"))
    try:
        intent = detect_message_intent(
            message,
            llm,
            session_mode=session.get("mode"),
            has_grievance_context=has_grievance_context,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini could not classify the message. Please try again later.",
        ) from exc
    route = route_intent(intent)
    if route == "greeting":
        try:
            greeting = llm.answer_greeting(message)
        except LLMServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail="The AI assistant could not respond to the greeting. Please try again later.",
            ) from exc
        return {
            "conversation_id": conversation_id,
            "intent": "general",
            "message": greeting,
        }
    is_registration_introduction = (
        not has_grievance_context
        and intent == "registration"
        and not extract_local_details(message, {}).get("description")
    )
    documents_for_response = session.get("documents", [])
    needs_document_lookup = (
        not is_registration_introduction
        and route in {"grievance", "document_question"}
        and (not has_grievance_context or route == "document_question")
    )
    if needs_document_lookup:
        try:
            documents = _retrieve_knowledge_documents(message)
        except LLMServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail="Gemini could not search the uploaded reference documents. Please try again later.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="The document service is temporarily unavailable.",
            ) from exc
        if sum(len(document["content"]) for document in documents) > MAX_KNOWLEDGE_CONTEXT_CHARACTERS:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded knowledge documents exceed the 1,000,000-character chat context limit. "
                    "Remove documents from the admin dashboard before continuing."
                ),
            )
        if documents:
            documents_for_response = documents
            if not has_grievance_context or intent != "general":
                session["documents"] = documents
        elif route == "document_question" or not has_grievance_context:
            return {
                "conversation_id": conversation_id,
                "intent": "general",
                "message": _unsupported_topic_reply(message),
            }

    should_check_document_topic = (
        not is_registration_introduction
        and (
            route == "document_question"
            or (route == "grievance" and not has_grievance_context)
        )
    )
    if should_check_document_topic:
        try:
            in_scope = llm.is_topic_relevant(
                message,
                documents_for_response,
                session.get("details", {}),
            )
        except LLMServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail="Gemini could not check whether this topic is supported. Please try again later.",
            ) from exc
        if not in_scope:
            return {
                "conversation_id": conversation_id,
                "intent": "general",
                "message": _assistant_reply(
                    message,
                    "general",
                    "Explain that the request is outside the topic covered by the uploaded reference documents. Briefly identify the supported topic only if the documents clearly establish it. Do not answer the unrelated question or start grievance registration.",
                    {"prior_grievance_details": session.get("details", {})},
                    documents_for_response,
                ),
            }

    previous_details = session.get("details", {})
    if route == "grievance":
        merged_details = dict(previous_details)
        if session.get("awaiting") == "contact":
            merged_details = extract_local_details(message, previous_details)
    else:
        merged_details = dict(previous_details)

    if route == "grievance" and session.get("awaiting") == "contact":
        submitted_contact = merged_details.get("contact")
        if not submitted_contact or submitted_contact == previous_details.get("contact"):
            submitted_contact = message
        try:
            _validate_contact({"contact": submitted_contact})
        except HTTPException as exc:
            attempts = session.get("invalid_contact_attempts", 0) + 1
            session["invalid_contact_attempts"] = attempts
            session["details"] = previous_details
            session["pending"] = False
            session["awaiting"] = "contact"
            if attempts >= MAX_INVALID_CONTACT_ATTEMPTS:
                session["mode"] = "customer_care_offer"
                return {
                    "conversation_id": conversation_id,
                    "intent": "customer_care",
                    "message": _assistant_reply(
                        message,
                        "customer_care",
                        "Explain the contact validation issue, then ask whether the user wants customer-care contact instructions.",
                        {"validation_error": exc.detail, "customer_care_contact": CUSTOMER_CARE_CONTACT},
                    ),
                }
            session["mode"] = "registration"
            return {
                "conversation_id": conversation_id,
                "intent": "registration",
                "message": _assistant_reply(
                    message,
                    "registration",
                    "Explain the contact validation issue and ask the user to enter a valid phone number or email address again.",
                    {"validation_error": exc.detail},
                ),
            }

    normalized_answer = re.sub(r"[.!?,]+", "", message.lower()).strip()
    declined_follow_up = session.get("awaiting") == "supporting_information" and normalized_answer in {
        "no", "nope", "i don't know", "i do not know", "not sure", "i'm not sure",
        "i am not sure", "i don't remember", "i do not remember",
    }
    if declined_follow_up:
        session["follow_up_declined"] = True
    else:
        session["follow_up_declined"] = False
    if not declined_follow_up and session.get("awaiting") == "supporting_information" and intent != "general":
        answer = merged_details.get("supporting_information") or message
        previous_information = previous_details.get("supporting_information", "")
        if answer and answer != previous_information:
            merged_details["supporting_information"] = (
                f"{previous_information}\n{answer}".strip() if previous_information else answer
            )
    state: dict[str, Any] = {
        "message": message,
        "details": merged_details,
        "required_follow_up_fields": [
            field["key"]
            for field in llm.grievance_template.get("grievance_information", [])
            if isinstance(field.get("key"), str) and field.get("follow_up_required") is True
        ],
        "awaiting": None if declined_follow_up else session.get("awaiting"),
        "follow_up_question": "",
        "previous_follow_up_questions": session.get("follow_up_questions", []),
        "follow_up_declined": declined_follow_up,
        "intent": intent,
    }
    try:
        result = workflow.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": conversation_id,
                    "llm_service": llm,
                    "documents": session.get("documents", []),
                }
            },
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI assistant could not validate the grievance details. Please try again later.",
        ) from exc
    intent = result["intent"]
    route = route_intent(intent)
    session["details"] = result.get("details", merged_details)
    session["follow_up_questions"] = result.get(
        "previous_follow_up_questions",
        session.get("follow_up_questions", []),
    )

    if route == "grievance":
        session["mode"] = "registration"
        review = result.get("action") == "review"
        session["awaiting"] = result.get("awaiting") or (result.get("missing", [None])[0] if result.get("missing") else None)
        response: dict[str, Any] = {
            "conversation_id": conversation_id,
            "intent": intent,
            "message": result["response"],
        }
        if review:
            try:
                _validate_name(session["details"])
            except HTTPException as exc:
                session["details"].pop("name", None)
                session["pending"] = False
                session["awaiting"] = "name"
                response["message"] = _assistant_reply(
                    message,
                    "registration",
                    "Explain the name validation issue and ask the user to provide a valid name.",
                    {"validation_error": exc.detail, "known_details": session["details"]},
                )
                return response
            try:
                _validate_contact(session["details"])
            except HTTPException as exc:
                attempts = session.get("invalid_contact_attempts", 0) + 1
                session["invalid_contact_attempts"] = attempts
                session["pending"] = False
                session["awaiting"] = "contact"
                if attempts >= MAX_INVALID_CONTACT_ATTEMPTS:
                    session["mode"] = "customer_care_offer"
                    response["message"] = _assistant_reply(
                        message,
                        "customer_care",
                        "Explain the contact validation issue, then ask whether the user wants customer-care contact instructions.",
                        {"validation_error": exc.detail, "customer_care_contact": CUSTOMER_CARE_CONTACT},
                    )
                else:
                    response["message"] = _assistant_reply(
                        message,
                        "registration",
                        "Explain the contact validation issue and ask the user to provide a valid phone number or email address.",
                        {"validation_error": exc.detail},
                    )
                return response
            session["invalid_contact_attempts"] = 0
            _consolidate_generic_template_details(session["details"])
            response["review"] = session["details"]
        session["pending"] = review
        return response

    if route == "tracking":
        session["mode"] = "tracking"
        match = re.search(r"GRV-\d{4}-[A-F0-9]{8}", message, re.I)
        if match:
            try:
                record = store.get_grievance(match.group(0))
            except Exception as exc:
                raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
            if record:
                session["mode"] = None
                public_record = _public_grievance(record)
                return {"conversation_id": conversation_id, "intent": intent, "message": _assistant_reply(
                    message, "tracking", "Report the grievance status from the backend result without adding other status claims.",
                    {"backend_result": public_record},
                ), "grievance": public_record}
        return {"conversation_id": conversation_id, "intent": intent, "message": result["response"]}

    if route == "escalation":
        session["mode"] = "escalation"
        match = re.search(r"GRV-\d{4}-[A-F0-9]{8}", message, re.I)
        if match:
            reason = message.replace(match.group(0), "").strip(" ,.;:-")
            if len(reason) >= 8:
                try:
                    record = store.escalate_grievance(match.group(0), reason)
                except Exception as exc:
                    raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
                session["mode"] = None
                if record and record["status"] == "escalated":
                    public_record = _public_grievance(record)
                    return {"conversation_id": conversation_id, "intent": intent, "message": _assistant_reply(
                        message, "escalation", "Confirm the escalation only if the backend result shows status escalated.",
                        {"backend_result": public_record},
                    ), "grievance": public_record}
        return {"conversation_id": conversation_id, "intent": intent, "message": result["response"]}

    try:
        answer = llm.answer_general(message, documents_for_response)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI assistant is unavailable. Check the Gemini configuration or try again later.",
        ) from exc
    return {"conversation_id": conversation_id, "intent": intent, "message": answer}


@app.post("/api/conversations/{conversation_id}/decision")
def decide(conversation_id: str, request: DecisionRequest) -> dict[str, Any]:
    session = conversations.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    if request.action == "cancel":
        if not session.get("pending"):
            raise HTTPException(status_code=409, detail=_assistant_reply(
                "", "registration", "Explain that there is no pending grievance to cancel."
            ))
        cancelled_details = dict(session["details"])
        workflow.invoke(
            Command(resume="cancel"),
            config={"configurable": {"thread_id": conversation_id}},
        )
        session.update({
            "details": {},
            "mode": None,
            "pending": False,
            "awaiting": None,
            "documents": [],
            "invalid_contact_attempts": 0,
            "follow_up_declined": False,
            "follow_up_questions": [],
        })
        return {"conversation_id": conversation_id, "message": _assistant_reply(
            "", "registration", "Confirm that the draft was cancelled and no grievance was submitted.",
            {"cancelled_draft": cancelled_details},
        )}

    if request.action == "edit":
        if not session.get("pending"):
            raise HTTPException(status_code=409, detail=_assistant_reply(
                "", "registration", "Explain that there is no pending grievance to edit."
            ))
        details = {**session["details"], **(request.details or {})}
        missing = [field for field in REQUIRED_FIELDS if not details.get(field, "").strip()]
        if missing:
            raise HTTPException(status_code=422, detail={
                "message": _assistant_reply(
                    "", "registration", "Tell the user which required fields are missing and ask them to complete the draft.",
                    {"missing_fields": missing},
                ),
                "missing": missing,
            })
        try:
            _validate_name(details)
        except HTTPException as exc:
            raise HTTPException(status_code=422, detail=_assistant_reply(
                "", "registration", "Explain the name validation issue and request a valid name.",
                {"validation_error": exc.detail},
            )) from exc
        try:
            _validate_contact(details)
        except HTTPException as exc:
            raise HTTPException(status_code=422, detail=_assistant_reply(
                "", "registration", "Explain the contact validation issue and request a valid phone number or email.",
                {"validation_error": exc.detail},
            )) from exc
        result = workflow.invoke(
            Command(resume={"action": "edit", "details": details}),
            config={"configurable": {"thread_id": conversation_id}},
        )
        interrupts = result.get("__interrupt__", [])
        if interrupts and isinstance(interrupts[-1].value, dict):
            details = interrupts[-1].value.get("details", details)
        session["details"] = details
        return {"conversation_id": conversation_id, "message": _assistant_reply(
            "", "registration", "Confirm the changes were saved to the draft and ask the user to review before submitting.",
            {"known_details": session["details"]},
        ), "review": session["details"]}

    if not session.get("pending"):
        if session.get("grievance_id"):
            try:
                existing = store.get_grievance(session["grievance_id"])
            except Exception as exc:
                raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
            if existing:
                public_record = _public_grievance(existing)
                return {"conversation_id": conversation_id, "message": _assistant_reply(
                    "", "registration", "Explain that this grievance was already submitted; do not submit it again.",
                    {"backend_result": public_record},
                ), "grievance": public_record}
        raise HTTPException(status_code=409, detail=_assistant_reply(
            "", "registration", "Explain that there is no pending grievance awaiting approval."
        ))

    details = session["details"]
    missing = [field for field in REQUIRED_FIELDS if not details.get(field, "").strip()]
    if missing:
        raise HTTPException(status_code=422, detail={
            "message": _assistant_reply(
                "", "registration", "Tell the user which required fields are missing and ask them to complete the draft.",
                {"missing_fields": missing},
            ),
            "missing": missing,
        })
    try:
        _validate_name(details)
    except HTTPException as exc:
        raise HTTPException(status_code=422, detail=_assistant_reply(
            "", "registration", "Explain the name validation issue and request a valid name.",
            {"validation_error": exc.detail},
        )) from exc
    try:
        _validate_contact(details)
    except HTTPException as exc:
        raise HTTPException(status_code=422, detail=_assistant_reply(
            "", "registration", "Explain the contact validation issue and request a valid phone number or email.",
            {"validation_error": exc.detail},
        )) from exc
    result = workflow.invoke(
        Command(resume="approve"),
        config={"configurable": {"thread_id": conversation_id}},
    )
    if not result.get("approved"):
        raise HTTPException(status_code=409, detail=_assistant_reply(
            "", "registration", "Explain that the grievance was not approved and therefore was not submitted."
        ))
    details = result.get("details", details)
    try:
        record = store.create_grievance(conversation_id, details)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The grievance could not be saved. Please try again.") from exc
    session.update({
        "details": {},
        "mode": None,
        "pending": False,
        "awaiting": None,
        "documents": [],
        "invalid_contact_attempts": 0,
        "follow_up_declined": False,
        "follow_up_questions": [],
        "grievance_id": record["grievance_id"],
    })
    public_record = _public_grievance(record)
    return {
        "conversation_id": conversation_id,
        "message": _assistant_reply(
            "", "registration", "Provide a concise submission confirmation with the grievance ID and tell the user who should receive the receipt.",
            {"backend_result": public_record, "recipient": details["recipient"]},
        ),
        "grievance": public_record,
    }


@app.get("/api/grievances/{grievance_id}")
def track(grievance_id: str) -> dict[str, Any]:
    try:
        record = store.get_grievance(grievance_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
    if not record:
        raise HTTPException(status_code=404, detail="Grievance not found.")
    return _public_grievance(record)


@app.post("/api/grievances/{grievance_id}/escalate")
def escalate(grievance_id: str, request: EscalationRequest) -> dict[str, Any]:
    try:
        record = store.escalate_grievance(grievance_id, request.reason.strip())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The grievance service is temporarily unavailable.") from exc
    if not record:
        raise HTTPException(status_code=404, detail="Grievance not found.")
    if record["status"] != "escalated":
        raise HTTPException(status_code=409, detail=f"A grievance with status '{record['status']}' cannot be escalated.")
    return _public_grievance(record)


