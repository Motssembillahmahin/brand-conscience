"""Model retraining trigger and checkpoint management."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brand_conscience.common.database import get_session
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.db.tables import ModelCheckpoint

logger = get_logger(__name__)


class ModelUpdater:
    """Manage model retraining and checkpoint promotion."""

    @traced(name="trigger_retrain", tags=["layer5", "retrain"])
    def trigger_retrain(
        self,
        model_name: str,
        reason: str,
    ) -> dict:
        """Trigger model retraining.

        Args:
            model_name: Name of the model to retrain (e.g., 'prompt_scorer').
            reason: Why retraining was triggered.

        Returns:
            Dict with retrain job details.
        """
        logger.info(
            "retrain_triggered",
            model_name=model_name,
            reason=reason,
        )

        # Dispatch retrain job via Celery
        task_id = self._dispatch_retrain_task(model_name, reason)

        return {
            "model_name": model_name,
            "reason": reason,
            "status": "queued",
            "task_id": task_id,
            "triggered_at": datetime.now(UTC).isoformat(),
        }

    def _dispatch_retrain_task(self, model_name: str, reason: str) -> str:
        """Dispatch a retrain job as a Celery task.

        Returns the Celery task ID for tracking.
        """
        try:
            from brand_conscience.celery_app import app

            result = app.send_task(
                "brand_conscience.tasks.retrain_model",
                kwargs={"model_name": model_name, "reason": reason},
                queue="ml_training",
            )
            logger.info(
                "retrain_task_dispatched",
                model_name=model_name,
                task_id=result.id,
            )
            return str(result.id)
        except Exception as exc:
            logger.error(
                "retrain_dispatch_failed",
                model_name=model_name,
                error=str(exc),
            )
            return ""

    @traced(name="promote_checkpoint", tags=["layer5", "checkpoint"])
    def promote_checkpoint(
        self,
        model_name: str,
        checkpoint_path: str,
        metrics: dict,
    ) -> str:
        """Promote a new model checkpoint to active.

        Deactivates the current active checkpoint and activates the new one.

        Args:
            model_name: Model name.
            checkpoint_path: Path to the new checkpoint file.
            metrics: Evaluation metrics for the new checkpoint.

        Returns:
            Checkpoint ID.
        """
        checkpoint_id = str(uuid.uuid4())

        with get_session() as session:
            # Deactivate current active checkpoint
            current_active = (
                session.query(ModelCheckpoint)
                .filter_by(model_name=model_name, is_active=True)
                .all()
            )
            for cp in current_active:
                cp.is_active = False

            # Get next version number
            max_version = session.query(ModelCheckpoint).filter_by(model_name=model_name).count()

            # Create new checkpoint record
            new_cp = ModelCheckpoint(
                id=uuid.UUID(checkpoint_id),
                model_name=model_name,
                version=max_version + 1,
                checkpoint_path=checkpoint_path,
                metrics=metrics,
                is_active=True,
                created_at=datetime.now(UTC),
            )
            session.add(new_cp)

        logger.info(
            "checkpoint_promoted",
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            version=max_version + 1,
            metrics=metrics,
        )
        return checkpoint_id

    def get_active_checkpoint(self, model_name: str) -> ModelCheckpoint | None:
        """Get the currently active checkpoint for a model."""
        with get_session() as session:
            return (
                session.query(ModelCheckpoint)
                .filter_by(model_name=model_name, is_active=True)
                .first()
            )
