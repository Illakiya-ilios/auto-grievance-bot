from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be configured before importing the PostgreSQL store.")
EMBEDDING_DIMENSIONS = 768
_initialized = False


def _connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def initialize() -> None:
    global _initialized
    with _connect() as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS grievances (
                grievance_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                contact TEXT NOT NULL,
                recipient TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                related_date TEXT,
                supporting_information TEXT,
                submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                approval_status TEXT NOT NULL CHECK (approval_status = 'approved'),
                status TEXT NOT NULL,
                history JSONB NOT NULL DEFAULT '[]'::jsonb
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS grievance_templates (
                template_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                recipient TEXT NOT NULL,
                guidance TEXT NOT NULL DEFAULT '',
                keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                questions JSONB NOT NULL DEFAULT '[]'::jsonb,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                file_data BYTEA,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        connection.execute("ALTER TABLE admin_documents ADD COLUMN IF NOT EXISTS file_data BYTEA")
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS admin_document_chunks (
                document_id TEXT NOT NULL REFERENCES admin_documents(document_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                PRIMARY KEY (document_id, chunk_index)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS admin_document_chunks_embedding_idx
            ON admin_document_chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
        connection.execute(
            "ALTER TABLE grievance_templates ADD COLUMN IF NOT EXISTS guidance TEXT NOT NULL DEFAULT ''"
        )
        connection.execute(
            "ALTER TABLE grievances ADD COLUMN IF NOT EXISTS recipient TEXT NOT NULL DEFAULT ''"
        )
    _initialized = True


def _ensure_initialized() -> None:
    if not _initialized:
        initialize()


def list_templates(active_only: bool = True) -> list[dict[str, Any]]:
    _ensure_initialized()
    query = "SELECT * FROM grievance_templates"
    if active_only:
        query += " WHERE active = TRUE"
    query += " ORDER BY title"
    with _connect() as connection:
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_template(template_id: str, active_only: bool = True) -> dict[str, Any] | None:
    _ensure_initialized()
    query = "SELECT * FROM grievance_templates WHERE template_id = %s"
    if active_only:
        query += " AND active = TRUE"
    with _connect() as connection:
        row = connection.execute(query, (template_id,)).fetchone()
    return dict(row) if row else None


def create_template(template: dict[str, Any]) -> dict[str, Any]:
    _ensure_initialized()
    template_id = uuid.uuid4().hex
    with _connect() as connection:
        row = connection.execute(
            """
            INSERT INTO grievance_templates (
                template_id, title, category, recipient, guidance, keywords, questions, active
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                template_id,
                template["title"],
                template["category"],
                template["recipient"],
                template["guidance"],
                Jsonb(template["keywords"]),
                Jsonb(template["questions"]),
                template["active"],
            ),
        ).fetchone()
    if not row:
        raise RuntimeError("The grievance template was not returned after creation.")
    return dict(row)


def update_template(template_id: str, template: dict[str, Any]) -> dict[str, Any] | None:
    _ensure_initialized()
    with _connect() as connection:
        row = connection.execute(
            """
            UPDATE grievance_templates
            SET title = %s, category = %s, recipient = %s, guidance = %s, keywords = %s,
                questions = %s, active = %s, updated_at = now()
            WHERE template_id = %s
            RETURNING *
            """,
            (
                template["title"],
                template["category"],
                template["recipient"],
                template["guidance"],
                Jsonb(template["keywords"]),
                Jsonb(template["questions"]),
                template["active"],
                template_id,
            ),
        ).fetchone()
    return dict(row) if row else None


def deactivate_template(template_id: str) -> bool:
    _ensure_initialized()
    with _connect() as connection:
        row = connection.execute(
            """
            UPDATE grievance_templates SET active = FALSE, updated_at = now()
            WHERE template_id = %s
            RETURNING template_id
            """,
            (template_id,),
        ).fetchone()
    return row is not None


def list_documents() -> list[dict[str, Any]]:
    _ensure_initialized()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT document_id, filename, length(content) AS character_count, created_at
            FROM admin_documents
            ORDER BY created_at DESC, filename
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_document(filename: str, content: str) -> dict[str, Any]:
    _ensure_initialized()
    document_id = uuid.uuid4().hex
    with _connect() as connection:
        row = connection.execute(
            """
            INSERT INTO admin_documents (document_id, filename, content)
            VALUES (%s, %s, %s)
            RETURNING document_id, filename, length(content) AS character_count, created_at
            """,
            (document_id, filename, content),
        ).fetchone()
    if not row:
        raise RuntimeError("The uploaded document was not returned after creation.")
    return dict(row)


def _insert_document_chunks(
    connection: psycopg.Connection,
    document_id: str,
    chunks: list[tuple[str, list[float]]],
) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO admin_document_chunks (document_id, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
            """,
            [
                (
                    document_id,
                    index,
                    content,
                    "[" + ",".join(str(float(value)) for value in embedding) + "]",
                )
                for index, (content, embedding) in enumerate(chunks)
            ],
        )


def create_document_with_embeddings(
    filename: str,
    content: str,
    chunks: list[tuple[str, list[float]]],
    file_data: bytes,
) -> dict[str, Any]:
    _ensure_initialized()
    document_id = uuid.uuid4().hex
    with _connect() as connection:
        row = connection.execute(
            """
            INSERT INTO admin_documents (document_id, filename, content, file_data)
            VALUES (%s, %s, %s, %s)
            RETURNING document_id, filename, length(content) AS character_count, created_at
            """,
            (document_id, filename, content, file_data),
        ).fetchone()
        if not row:
            raise RuntimeError("The uploaded document was not returned after creation.")
        _insert_document_chunks(connection, document_id, chunks)
    return dict(row)


def list_unindexed_documents() -> list[dict[str, str]]:
    _ensure_initialized()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT d.document_id, d.filename, d.content
            FROM admin_documents AS d
            WHERE NOT EXISTS (
                SELECT 1 FROM admin_document_chunks AS c
                WHERE c.document_id = d.document_id
            )
            ORDER BY d.created_at, d.filename
            """
        ).fetchall()
    return [
        {"document_id": row["document_id"], "filename": row["filename"], "content": row["content"]}
        for row in rows
    ]


def index_document_chunks(
    document_id: str,
    chunks: list[tuple[str, list[float]]],
) -> None:
    _ensure_initialized()
    with _connect() as connection:
        _insert_document_chunks(connection, document_id, chunks)


def search_knowledge_documents(
    embedding: list[float],
    limit: int = 5,
) -> list[dict[str, str]]:
    _ensure_initialized()
    vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT d.filename, c.content
            FROM admin_document_chunks AS c
            JOIN admin_documents AS d ON d.document_id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector, limit),
        ).fetchall()
    return [{"filename": row["filename"], "content": row["content"]} for row in rows]


def delete_document(document_id: str) -> bool:
    _ensure_initialized()
    with _connect() as connection:
        row = connection.execute(
            "DELETE FROM admin_documents WHERE document_id = %s RETURNING document_id",
            (document_id,),
        ).fetchone()
    return row is not None


def list_knowledge_documents() -> list[dict[str, str]]:
    _ensure_initialized()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT filename, content
            FROM admin_documents
            ORDER BY created_at DESC, filename
            """,
        ).fetchall()
    return [{"filename": row["filename"], "content": row["content"]} for row in rows]


def create_grievance(conversation_id: str, details: dict[str, str]) -> dict[str, Any]:
    _ensure_initialized()
    grievance_id = f"GRV-{datetime.now().year}-{uuid.uuid4().hex[:8].upper()}"
    event = {"status": "submitted", "at": datetime.now().astimezone().isoformat(), "note": "Submitted with user approval"}
    with _connect() as connection:
        inserted = connection.execute(
            """
            INSERT INTO grievances (
                grievance_id, conversation_id, name, contact, recipient, category, subject,
                description, related_date, supporting_information, approval_status, status, history
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'approved', 'submitted', %s)
            ON CONFLICT (conversation_id) DO NOTHING
            RETURNING *
            """,
            (
                grievance_id, conversation_id, details["name"], details["contact"], details["recipient"], details["category"],
                details["subject"], details["description"], details.get("related_date"),
                details.get("supporting_information"), Jsonb([event]),
            ),
        ).fetchone()
        if inserted:
            return dict(inserted)
        existing = connection.execute(
            "SELECT * FROM grievances WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        if not existing:
            raise RuntimeError("The grievance could not be retrieved after a duplicate submission.")
        return dict(existing)


def get_grievance(grievance_id: str) -> dict[str, Any] | None:
    _ensure_initialized()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM grievances WHERE upper(grievance_id) = upper(%s)", (grievance_id.strip(),)
        ).fetchone()
    return dict(row) if row else None


def escalate_grievance(grievance_id: str, reason: str) -> dict[str, Any] | None:
    _ensure_initialized()
    with _connect() as connection:
        row = connection.execute(
            """
            UPDATE grievances
            SET status = 'escalated',
                history = history || %s::jsonb
            WHERE upper(grievance_id) = upper(%s) AND status NOT IN ('resolved', 'closed')
            RETURNING *
            """,
            (
                json.dumps([{"status": "escalated", "at": datetime.now().astimezone().isoformat(), "note": reason}]),
                grievance_id.strip(),
            ),
        ).fetchone()
    return dict(row) if row else get_grievance(grievance_id)
