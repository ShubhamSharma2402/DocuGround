"""Document upload, listing, and deletion endpoints."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import aiofiles
import fitz  # PyMuPDF
import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import FileTooLargeError, ForbiddenError, InvalidPDFError, NotFoundError
from app.core.metrics import DOCUGROUND_UPLOADS_TOTAL
from app.core.rate_limit import RateLimiter
from app.db.client import get_db
from app.db.models import DocumentDB, DocumentRead, DocumentStatus, UploadResponse, UserDB, utc_now
from app.db.redis import create_job_record, enqueue_ingest_job
from app.services.vector_store import get_vector_store

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_CHUNK_BUFFER_SIZE = 64 * 1024  # 64 KB stream buffer


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection."""
    name = Path(filename).name
    # Keep only alphanumeric, underscores, hyphens, dots
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return cleaned or "document.pdf"


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    _rate_limit: None = Depends(RateLimiter("upload", "rate_limit_upload_per_minute")),
) -> UploadResponse:
    """Stream-upload a PDF file, validate, enqueue background ingestion, and return 202."""
    DOCUGROUND_UPLOADS_TOTAL.inc()
    settings = get_settings()
    user_id = str(current_user.id)
    doc_id = str(ObjectId())
    job_id = str(uuid.uuid4())

    raw_filename = file.filename or "uploaded.pdf"
    safe_name = sanitize_filename(raw_filename)

    # Content-type pre-check
    if file.content_type and "pdf" not in file.content_type.lower() and not safe_name.lower().endswith(".pdf"):
        raise InvalidPDFError("Content-Type must be application/pdf")

    temp_path = settings.upload_dir / f"tmp_{user_id}_{doc_id}.pdf"
    total_bytes = 0
    first_chunk = True

    try:
        async with aiofiles.open(temp_path, "wb") as out_file:
            while True:
                chunk = await file.read(_CHUNK_BUFFER_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise FileTooLargeError(
                        f"File size exceeds limit of {settings.max_upload_mb} MB"
                    )

                if first_chunk:
                    if not chunk.startswith(b"%PDF-"):
                        raise InvalidPDFError("File is not a valid PDF (invalid magic bytes)")
                    first_chunk = False

                await out_file.write(chunk)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        temp_path.unlink(missing_ok=True)
        raise InvalidPDFError("Uploaded file is empty")

    # Cheap PDF structure and page count validation with PyMuPDF
    try:
        pdf_doc = fitz.open(str(temp_path))
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise InvalidPDFError(f"Corrupt or unreadable PDF: {exc}") from exc

    try:
        if pdf_doc.is_encrypted:
            raise InvalidPDFError("PDF is encrypted or password-protected")

        page_count = pdf_doc.page_count
        if page_count > settings.max_pages:
            raise InvalidPDFError(
                f"PDF has {page_count} pages, which exceeds maximum limit of {settings.max_pages}"
            )
    finally:
        pdf_doc.close()

    # Move to permanent upload storage path
    final_path = settings.upload_dir / f"{user_id}_{doc_id}.pdf"
    temp_path.rename(final_path)

    now = utc_now()
    doc_record = {
        "_id": doc_id,
        "user_id": user_id,
        "filename": safe_name,
        "size_bytes": total_bytes,
        "status": DocumentStatus.processing.value,
        "page_count": page_count,
        "chunk_count": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    await db.documents.insert_one(doc_record)

    # Redis job tracking & stream dispatch
    await create_job_record(
        job_id=job_id,
        doc_id=doc_id,
        user_id=user_id,
        filename=safe_name,
        file_path=str(final_path),
    )
    await enqueue_ingest_job(
        job_id=job_id,
        doc_id=doc_id,
        user_id=user_id,
        file_path=str(final_path),
    )

    log.info(
        "Document upload accepted",
        doc_id=doc_id,
        job_id=job_id,
        user_id=user_id,
        bytes=total_bytes,
        pages=page_count,
    )
    return UploadResponse(job_id=job_id, document_id=doc_id)


@router.get("", response_model=list[DocumentRead])
async def list_documents(
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> list[DocumentRead]:
    """List all documents owned by the current user."""
    user_id = str(current_user.id)
    cursor = db.documents.find({"user_id": user_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=500)

    return [
        DocumentRead(
            id=str(d["_id"]),
            user_id=d["user_id"],
            filename=d["filename"],
            size_bytes=d["size_bytes"],
            status=DocumentStatus(d["status"]),
            page_count=d.get("page_count"),
            chunk_count=d.get("chunk_count"),
            error=d.get("error"),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
        for d in docs
    ]


@router.get("/{doc_id}", response_model=DocumentRead)
async def get_document(
    doc_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DocumentRead:
    """Get metadata for a single document."""
    d = await db.documents.find_one({"_id": doc_id})
    if not d:
        raise NotFoundError("Document not found")
    if d["user_id"] != str(current_user.id):
        raise ForbiddenError("You do not have access to this document")

    return DocumentRead(
        id=str(d["_id"]),
        user_id=d["user_id"],
        filename=d["filename"],
        size_bytes=d["size_bytes"],
        status=DocumentStatus(d["status"]),
        page_count=d.get("page_count"),
        chunk_count=d.get("chunk_count"),
        error=d.get("error"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


@router.get("/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> FileResponse:
    """Return raw PDF binary stream for the document viewer."""
    user_id = str(current_user.id)
    doc = await db.documents.find_one({"_id": doc_id})
    if not doc:
        raise NotFoundError("Document not found")
    if doc["user_id"] != user_id:
        raise ForbiddenError("You do not have access to this document")

    settings = get_settings()
    disk_file = settings.upload_dir / f"{user_id}_{doc_id}.pdf"
    if not disk_file.exists():
        raise NotFoundError("Document file not found on disk")

    return FileResponse(
        path=disk_file,
        media_type="application/pdf",
        filename=doc["filename"],
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict[str, str]:
    """Delete a document, its Qdrant vectors, and local file."""
    user_id = str(current_user.id)
    doc = await db.documents.find_one({"_id": doc_id})
    if not doc:
        raise NotFoundError("Document not found")
    if doc["user_id"] != user_id:
        raise ForbiddenError("You do not have permission to delete this document")

    # 1. Delete points from Qdrant (best-effort if vector DB temporarily unreachable)
    try:
        vs = get_vector_store()
        await vs.delete_by_doc_id(doc_id, user_id)
    except Exception as exc:
        log.warning("Qdrant deletion failed or unreachable", doc_id=doc_id, error=str(exc))

    # 2. Delete MongoDB record
    await db.documents.delete_one({"_id": doc_id, "user_id": user_id})

    # 3. Delete file on disk if exists
    settings = get_settings()
    disk_file = settings.upload_dir / f"{user_id}_{doc_id}.pdf"
    disk_file.unlink(missing_ok=True)

    log.info("Document deleted", doc_id=doc_id, user_id=user_id)
    return {"message": "Document deleted successfully"}
