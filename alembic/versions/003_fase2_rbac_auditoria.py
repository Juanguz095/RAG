"""003: RBAC (constraint de rol) + auditoría + visibilidad de documentos.

Solo ADD/CREATE (nada destructivo), según PLAN-004 §4.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


revision = "003_fase2_rbac_auditoria"
down_revision = "002_bge_m3_1024d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # usuarios: constraint de roles (5 del spec)
    op.create_check_constraint(
        "chk_user_role",
        "users",
        "role IN ('admin','editor','assistant','viewer','auditor')",
    )

    # auditoría append-only
    op.create_table(
        "audit_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("username", sa.String(50)),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("resource_type", sa.String(50)),
        sa.Column("resource_id", sa.String(64)),
        sa.Column("detail", JSONB, server_default="{}"),
        sa.Column("ip", sa.String(45)),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "action IN ('login','login_failed','logout','register','upload','update',"
            "'delete','search','keyword','query','export','admin_action','validation','error')",
            name="chk_audit_action",
        ),
    )
    op.create_index("idx_audit_created", "audit_log", ["created_at"])
    op.create_index("idx_audit_user", "audit_log", ["user_id"])
    op.create_index("idx_audit_action", "audit_log", ["action"])

    # visibilidad de documentos + ACL
    op.add_column("documents", sa.Column("visibility", sa.String(10), server_default="public"))
    op.create_table(
        "document_acl",
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(20), primary_key=True),
    )


def downgrade() -> None:
    op.drop_table("document_acl")
    op.drop_column("documents", "visibility")
    op.drop_index("idx_audit_action", table_name="audit_log")
    op.drop_index("idx_audit_user", table_name="audit_log")
    op.drop_index("idx_audit_created", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_constraint("chk_user_role", "users", type_="check")
