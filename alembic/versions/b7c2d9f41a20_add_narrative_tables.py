"""Add narrative extraction tables (facts, events, world_rules, knowledge)

Revision ID: b7c2d9f41a20
Revises: 1a81013deea8
Create Date: 2026-11-28 00:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7c2d9f41a20'
down_revision: Union[str, Sequence[str], None] = '1a81013deea8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create narrative_facts, narrative_events, world_rules, and narrative_knowledge tables."""
    op.create_table(
        'narrative_facts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('subject', sa.String(), nullable=False),
        sa.Column('predicate', sa.String(), nullable=False),
        sa.Column('object', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('source_segment', sa.String(), nullable=True),
        sa.Column('scene_id', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_narrative_facts_id'), 'narrative_facts', ['id'], unique=False)
    op.create_index(op.f('ix_narrative_facts_document_id'), 'narrative_facts', ['document_id'], unique=False)
    op.create_index(op.f('ix_narrative_facts_subject'), 'narrative_facts', ['subject'], unique=False)
    op.create_index(op.f('ix_narrative_facts_predicate'), 'narrative_facts', ['predicate'], unique=False)
    op.create_index(op.f('ix_narrative_facts_source_segment'), 'narrative_facts', ['source_segment'], unique=False)
    op.create_index(op.f('ix_narrative_facts_scene_id'), 'narrative_facts', ['scene_id'], unique=False)

    op.create_table(
        'narrative_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('characters', sa.JSON(), nullable=False),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('temporal_order', sa.String(), nullable=True),
        sa.Column('source_segment', sa.String(), nullable=True),
        sa.Column('scene_id', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_narrative_events_id'), 'narrative_events', ['id'], unique=False)
    op.create_index(op.f('ix_narrative_events_document_id'), 'narrative_events', ['document_id'], unique=False)
    op.create_index(op.f('ix_narrative_events_source_segment'), 'narrative_events', ['source_segment'], unique=False)
    op.create_index(op.f('ix_narrative_events_scene_id'), 'narrative_events', ['scene_id'], unique=False)

    op.create_table(
        'world_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule', sa.Text(), nullable=False),
        sa.Column('established_in', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_world_rules_id'), 'world_rules', ['id'], unique=False)
    op.create_index(op.f('ix_world_rules_document_id'), 'world_rules', ['document_id'], unique=False)

    op.create_table(
        'narrative_knowledge',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('character', sa.String(), nullable=False),
        sa.Column('fact', sa.Text(), nullable=False),
        sa.Column('knows', sa.Boolean(), nullable=False),
        sa.Column('source_segment', sa.String(), nullable=True),
        sa.Column('scene_id', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_narrative_knowledge_id'), 'narrative_knowledge', ['id'], unique=False)
    op.create_index(op.f('ix_narrative_knowledge_document_id'), 'narrative_knowledge', ['document_id'], unique=False)
    op.create_index(op.f('ix_narrative_knowledge_character'), 'narrative_knowledge', ['character'], unique=False)
    op.create_index(op.f('ix_narrative_knowledge_source_segment'), 'narrative_knowledge', ['source_segment'], unique=False)
    op.create_index(op.f('ix_narrative_knowledge_scene_id'), 'narrative_knowledge', ['scene_id'], unique=False)


def downgrade() -> None:
    """Drop narrative extraction tables."""
    op.drop_table('narrative_knowledge')
    op.drop_table('world_rules')
    op.drop_table('narrative_events')
    op.drop_table('narrative_facts')
