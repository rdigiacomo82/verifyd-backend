"""Read-only learning-based game-feature unlocks.

Games never award Trust Rank points. Existing games remain available on the
device; this endpoint only offers additional optional levels/features after
verified Homework milestones. No behavior, safety or identity score is created.
"""
import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/learning-unlocks", tags=["Circle Learning Unlocks"])

UNLOCKS = (
    {"id": "bonus-match-theme", "label": "Extra Circle Match theme", "required_homework": 1},
    {"id": "star-quest-bonus-world", "label": "Star Quest bonus world", "required_homework": 3},
    {"id": "creative-challenge-pack", "label": "Creative Challenge Pack", "required_homework": 6},
)


def enabled():
    return os.environ.get("VERIFYD_CIRCLE_LEARNING_UNLOCKS_ENABLED", "0").strip().lower() in ("1", "yes", "on", "true")


@router.get("/me")
def my_unlocks(authorization: Optional[str] = Header(default=None)):
    if not enabled():
        raise HTTPException(404, detail={"error": "learning_unlocks_unavailable"})
    identity = _identity_from_bearer(authorization)
    if identity.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    # Homework schema is managed by its own already-enabled service. A missing
    # schema is an error, not an excuse to invent achievements.
    from trust_voice_homework import _schema as ensure_homework_schema
    ensure_homework_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT COUNT(DISTINCT assignment_id) AS completed
                       FROM verifyd_circle_homework_work
                       WHERE identity_id=%s AND status='completed'""", (identity["id"],))
        completed = int(cur.fetchone()["completed"])
    return {
        "version": "0.1.0",
        "completed_homework_assignments": completed,
        "base_games_remain_available": True,
        "game_points_added_to_trust_rank": False,
        "optional_unlocks": [
            {**item, "unlocked": completed >= item["required_homework"],
             "remaining": max(0, item["required_homework"] - completed)}
            for item in UNLOCKS
        ],
        "note": "Optional unlocks recognize completed learning. Accessibility and core game access are never restricted.",
    }
