"""CLI entry points using Click."""

from __future__ import annotations

import click

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger

logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """Brand Conscience — Autonomous Meta Advertisement System."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)


@cli.command()
def health() -> None:
    """Check connectivity to all services."""
    settings = load_settings()
    checks: dict[str, str] = {}

    # PostgreSQL
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgresql"] = "OK"
    except Exception as exc:
        checks["postgresql"] = f"FAIL: {exc}"

    # Redis
    try:
        import redis

        r = redis.from_url(settings.redis.url)
        r.ping()
        checks["redis"] = "OK"
    except Exception as exc:
        checks["redis"] = f"FAIL: {exc}"

    # Meta API
    if settings.meta.access_token:
        checks["meta_api"] = "configured"
    else:
        checks["meta_api"] = "not configured"

    # Gemini API
    if settings.gemini.api_key:
        checks["gemini_api"] = "configured"
    else:
        checks["gemini_api"] = "not configured"

    # Slack
    if settings.slack.bot_token:
        checks["slack"] = "configured"
    else:
        checks["slack"] = "not configured"

    click.echo("Brand Conscience Health Check")
    click.echo("=" * 40)
    all_ok = True
    for service, status in checks.items():
        icon = "+" if "OK" in status or "configured" in status else "-"
        click.echo(f"  [{icon}] {service}: {status}")
        if "FAIL" in status:
            all_ok = False

    if all_ok:
        click.echo("\nAll checks passed.")
    else:
        click.echo("\nSome checks failed. See above.")
        raise SystemExit(1)


@cli.command()
@click.option("--force", is_flag=True, help="Force immediate monitoring cycle")
def monitor(force: bool) -> None:
    """Run a monitoring cycle."""
    from brand_conscience.common.database import init_database
    from brand_conscience.layer0_awareness.tasks import run_monitoring_cycle

    settings = load_settings()
    init_database(settings)

    click.echo("Running monitoring cycle...")
    result = run_monitoring_cycle()
    click.echo(f"Urgency: {result['urgency_score']:.2f}")
    click.echo(f"Action: {result['recommended_action']}")
    click.echo(f"Summary: {result['context_summary']}")


@cli.command()
def pipeline() -> None:
    """Run the full pipeline (awareness → strategy → prompts → creative → deploy)."""
    from brand_conscience.app import create_app
    from brand_conscience.common.database import init_database
    from brand_conscience.common.tracing import init_opik

    settings = load_settings()
    init_database(settings)
    init_opik()

    click.echo("Running full pipeline...")
    app = create_app()
    result = app.invoke({"context": {}})

    click.echo(f"Pipeline complete. Status: {result.get('deployment_status', 'N/A')}")


@cli.group()
def campaigns() -> None:
    """Campaign management commands."""


@campaigns.command("list")
def campaigns_list() -> None:
    """List all campaigns."""
    from brand_conscience.common.database import init_database, get_session
    from brand_conscience.db.tables import Campaign

    settings = load_settings()
    init_database(settings)

    with get_session() as session:
        all_campaigns = session.query(Campaign).order_by(Campaign.created_at.desc()).all()

    if not all_campaigns:
        click.echo("No campaigns found.")
        return

    for c in all_campaigns:
        click.echo(f"  {c.id} | {c.name} | {c.status.value} | ${c.daily_budget:,.2f}")


@campaigns.command()
@click.argument("campaign_id")
def approve(campaign_id: str) -> None:
    """Approve a pending campaign."""
    from brand_conscience.common.database import init_database
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager

    settings = load_settings()
    init_database(settings)

    manager = CampaignManager()
    manager.approve(campaign_id)
    click.echo(f"Campaign {campaign_id} approved and set to LIVE.")


@campaigns.command()
@click.argument("campaign_id")
def pause(campaign_id: str) -> None:
    """Pause a live campaign."""
    from brand_conscience.common.database import init_database
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager

    settings = load_settings()
    init_database(settings)

    manager = CampaignManager()
    manager.pause(campaign_id)
    click.echo(f"Campaign {campaign_id} paused.")


if __name__ == "__main__":
    cli()
