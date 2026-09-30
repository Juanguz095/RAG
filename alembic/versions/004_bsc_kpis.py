"""004: tablas BSC — kpis, kpi_snapshots, alerts, action_plans (PLAN-005).

Solo CREATE TABLE (nada destructivo), según PLAN-005.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


revision = "004_bsc_kpis"
down_revision = "003_fase2_rbac_auditoria"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kpis",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("code", sa.String(30), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "perspective", sa.String(20), nullable=False,
            server_default="conocimiento",
        ),
        sa.CheckConstraint(
            "perspective IN ('conocimiento','rag','usuario','seguridad')",
            name="chk_kpi_perspective",
        ),
        sa.Column("direction", sa.String(3), server_default="gte"),
        sa.Column("unit", sa.String(20)),
        sa.Column("target", sa.Float),
        sa.Column("thresholds", JSONB),
        sa.Column("query_type", sa.String(30), nullable=False),
        sa.Column("is_active", sa.Integer, server_default="1"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )

    op.create_table(
        "kpi_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("kpi_id", UUID(as_uuid=True), sa.ForeignKey("kpis.id", ondelete="CASCADE"), nullable=False),
        sa.Column("period_start", sa.DateTime, nullable=False),
        sa.Column("period_end", sa.DateTime, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("computed_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("kpi_id", "period_start", "period_end", name="uq_kpi_snapshot_period"),
    )
    op.create_index("idx_kpi_snapshots_kpi", "kpi_snapshots", ["kpi_id"])

    op.create_table(
        "alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("kpi_id", UUID(as_uuid=True), sa.ForeignKey("kpis.id", ondelete="SET NULL")),
        sa.Column("severity", sa.String(10), nullable=False, server_default="warnings"),
        sa.CheckConstraint("severity IN ('info','warning','critical')", name="chk_alert_sev"),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("status", sa.String(10), server_default="open"),
        sa.CheckConstraint("status IN ('open','ack','resolved')", name="chk_alert_status"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("resolved_at", sa.DateTime),
        sa.Column("resolved_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
    )
    op.create_index("idx_alerts_status", "alerts", ["status"])

    op.create_table(
        "action_plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("kpi_id", UUID(as_uuid=True), sa.ForeignKey("kpis.id", ondelete="SET NULL")),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("owner", sa.String(100)),
        sa.Column("due_date", sa.DateTime),
        sa.Column("status", sa.String(10), server_default="open"),
        sa.CheckConstraint("status IN ('open','in_progress','done','cancelled')", name="chk_plan_status"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_plans_status", "action_plans", ["status"])


def downgrade() -> None:
    op.drop_table("action_plans")
    op.drop_table("alerts")
    op.drop_table("kpi_snapshots")
    op.drop_table("kpis")
