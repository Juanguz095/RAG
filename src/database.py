from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector

from src.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_size=5, max_overflow=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="assistant", nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


ROLE_VALUES = ("admin", "editor", "assistant", "viewer", "auditor")


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(100))
    file_size = Column(Integer)
    page_count = Column(Integer)
    status = Column(String(20), default="pending")
    total_chunks = Column(Integer, default=0)
    visibility = Column(String(10), default="public")  # public|restricted
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    chunks = relationship("Chunk", back_populates="document", cascade="all,delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("idx_chunks_embedding_hnsw", "embedding",
              postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_chunks_content_trgm", "content",
              postgresql_using="gin",
              postgresql_ops={"content": "gin_trgm_ops"}),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_numbers = Column(ARRAY(Integer), default=[])
    bbox = Column(JSONB)
    token_count = Column(Integer)
    chunk_metadata = Column(JSONB, default=dict)
    embedding = Column(Vector(1024))
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class MedicalEntity(Base):
    __tablename__ = "medical_entities"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    entity_text = Column(Text, nullable=False)
    entity_type = Column(String(50), nullable=False)
    icd10_code = Column(String(20))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class MedicalSynonym(Base):
    __tablename__ = "medical_synonyms"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical = Column(String(200), nullable=False, index=True)
    synonym = Column(String(200), nullable=False, index=True)
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Registro append-only de eventos sensibles (RAG-039/040, PLAN-004)."""
    __tablename__ = "audit_log"
    __table_args__ = (
        CheckConstraint(
            "action IN ('login','login_failed','logout','register','upload','update',"
            "'delete','search','keyword','query','export','admin_action','validation','error')",
            name="chk_audit_action",
        ),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    username = Column(String(50))
    action = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(String(64))
    detail = Column(JSONB, default=dict)
    ip = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class DocumentACL(Base):
    """Roles con acceso a documentos restricted (PLAN-004 C7)."""
    __tablename__ = "document_acl"
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    role = Column(String(20), primary_key=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
