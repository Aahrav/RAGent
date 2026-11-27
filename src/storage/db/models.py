import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from src.storage.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="public")  # e.g., 'admin', 'engineering', 'sales', 'public'

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class DocumentAccess(Base):
    __tablename__ = "document_access"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    allowed_role = Column(String, nullable=False)


class NarrativeFact(Base):
    __tablename__ = "narrative_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    subject = Column(String, nullable=False, index=True)
    predicate = Column(String, nullable=False, index=True)
    object = Column(String, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    source_segment = Column(String, nullable=True, index=True)
    scene_id = Column(String, nullable=True, index=True)
    source = Column(String, nullable=False, default="system1_deterministic")
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class NarrativeEventRecord(Base):
    __tablename__ = "narrative_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    description = Column(Text, nullable=False)
    characters = Column(JSON, nullable=False, default=list)
    location = Column(String, nullable=True)
    temporal_order = Column(String, nullable=True)
    source_segment = Column(String, nullable=True, index=True)
    scene_id = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class WorldRuleRecord(Base):
    __tablename__ = "world_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    rule = Column(Text, nullable=False)
    established_in = Column(String, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class NarrativeKnowledgeRecord(Base):
    __tablename__ = "narrative_knowledge"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    character = Column(String, nullable=False, index=True)
    fact = Column(Text, nullable=False)
    knows = Column(Boolean, nullable=False, default=True)
    source_segment = Column(String, nullable=True, index=True)
    scene_id = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
