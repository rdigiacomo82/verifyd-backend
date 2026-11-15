"""VeriFYD Circle: server-verified Ranked Missions for four older-player games.

The browser may run local practice using public questions. Ranked missions use
DIFFERENT, server-held questions and independent correctness checks.
No raw client-reported score or point amount is trusted.

Flag: VERIFYD_CIRCLE_MISSION_ARCADE_ENABLED=1
Also requires VERIFYD_CIRCLE_RANKING_ENABLED=1.
Nothing here alters existing Arcade, Homework, Therapy, or Trust Rank tiers.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_ranks import _schema as ensure_rank_schema, enabled as rank_enabled

router = APIRouter(prefix="/trust-voice/mission-arcade", tags=["Circle Ranked Missions"])
_READY = False
_LOCK = threading.Lock()
_TRUE = ("1", "true", "yes", "on")
LEVELS = ("beginner", "explorer", "expert")
LEVEL_POINTS = (10, 25, 50)
LEVEL_MINIMUMS = (0, 120, 350)
GAMES = {
    "mars-rescue": {"title": "Mission Mars Rescue", "unlock": 0},
    "media-detective": {"title": "Media Detective", "unlock": 0},
    "debate-duel": {"title": "Debate Duel", "unlock": 150},
    "design-lab": {"title": "Design Lab", "unlock": 350},
}

# Private server bank: never put the answer field or correct option index
# into GET/catalog or POST/sessions response. The local practice bank is distinct.
# Row format: (prompt, answer, wrong-option-A, wrong-option-B, explanation).
BANK = {
"media-detective": [
 [
  ("An old photograph is presented as news from today. What checks it best?", "Find the earliest dated use", "Count the shares", "Read only the latest caption", "The original date tests whether a caption is misleading."),
  ("A post asks you to install a 'verification' extension. Safest choice?", "Verify directly with the official service", "Install the extension at once", "Send your password by message", "Unrequested extensions may be unsafe."),
  ("A headline uses a number with no source. What makes it checkable?", "A verifiable primary data source", "Lots of emojis", "An anonymous repost", "A checkable source supports numerical claims."),
  ("Two posts have exactly the same text. What must you investigate?", "Whether they originated independently", "Which has more reactions", "Their font colors", "Copies are not independent confirmations."),
 ],
 [
  ("A survey used 14 volunteers from one club. Which concern matters?", "Selection bias and limited sample", "Too many measurements", "The graph's font", "A convenience sample may not represent a population."),
  ("A chart starts at 99 on the vertical scale. What should you check?", "Whether axis truncation exaggerates change", "Whether the labels are bold", "Whether the graph uses blue", "Truncated scales can distort visual impressions."),
  ("A headline claims a treatment always works from one case. What is the flaw?", "Drawing a general claim from one example", "Using an exact date", "Showing the author's name", "Anecdotes do not justify universal conclusions."),
  ("An account has a verification checkmark. What does that NOT establish?", "That every claim it publishes is true", "That the account has a profile", "That its content can be read", "Account labels do not independently validate a claim."),
 ],
 [
  ("A study has a statistically significant result but tiny effect size. What matters?", "Both effect size and context", "Only the headline", "Only the color of the chart", "Statistical significance does not establish importance."),
  ("Two studies disagree and sampled different groups. Best next step?", "Compare their methods and populations", "Treat the larger headline as proof", "Assume both are false", "Differences in methods or samples can explain divergent results."),
  ("A video has believable audio but no provenance. The sound alone proves?", "Neither authenticity nor fabrication", "Definitely genuine", "Definitely synthetic", "Provenance and corroboration are necessary."),
  ("A claim cites a source that cites the original claim. What problem?", "Circular sourcing", "Too much independent evidence", "A randomized trial", "Circular citations add no independent verification."),
 ],
],
"debate-duel": [
 [
  ("A school proposes bike racks. Which argument is evidence-based?", "Surveyed demand plus available space", "Everyone I know would love it", "The racks would look cool", "A specific need and feasibility support the claim."),
  ("Someone challenges your position. Which question advances discussion?", "What evidence supports the alternative?", "Why are you so wrong?", "Can you just stop talking?", "Seek testable reasons and stay respectful."),
  ("Which is an observable statement?", "A library has 24 seats", "This library is the nicest", "Books are always boring", "A seat count is directly testable."),
  ("A proposal benefits some students but excludes others. Best response?", "Consider adjustments and accessibility", "Ignore the excluded group", "Cancel all discussion", "Inclusive design addresses trade-offs."),
 ],
 [
  ("'Every teenager dislikes math' is rebutted by what?", "One valid counterexample", "A louder speaker", "A popular opinion", "A counterexample challenges an absolute claim."),
  ("A speaker changes the topic instead of addressing evidence. This is?", "A diversion from the original claim", "A controlled experiment", "A careful concession", "Off-topic arguments avoid the question."),
  ("A new schedule saves time but increases costs. Strong analysis?", "Weigh benefits, costs and alternatives", "Only count time savings", "Only count costs", "Trade-offs require examining more than one factor."),
  ("Why does a sample of one rarely prove a universal rule?", "Individual cases may not generalize", "One always equals zero", "Personal stories are illegal", "Variation limits broad conclusions."),
 ],
 [
  ("Two groups improved after different interventions. What cannot follow?", "One intervention necessarily caused all gains", "Both groups changed", "The measurements can be compared", "Other factors may explain the results."),
  ("A speaker rejects an idea by attacking its author. This is?", "Ad hominem reasoning", "A valid causal inference", "A direct counterexample", "Judging the person does not address the evidence."),
  ("A chart omits the survey's missing responses. What matters?", "Potential nonresponse bias", "Which marker is larger", "Whether the chart is printed", "Unanswered responses can bias reported results."),
  ("What strengthens a nuanced claim about school policy?", "Stated assumptions and limits", "No possible exceptions", "A personal insult", "Acknowledged limits improve reasoned argument."),
 ],
],
"design-lab": [
 [
  ("A paper bridge fails under load. Best next step?", "Change one design variable and retest", "Use the same plan without testing", "Hide the test result", "Controlled iteration helps identify improvements."),
  ("Which shape often strengthens a simple truss?", "Triangle", "Open semicircle only", "Loose straight line", "Triangular bracing helps resist deformation."),
  ("A model uses two different adhesives. How compare fairly?", "Same material, load and curing conditions", "Different loads for each", "Only compare appearance", "Control other factors when testing one variable."),
  ("Why record a prototype's failures?", "They guide the next design iteration", "They prove effort was wasted", "They replace all measurements", "Failures provide useful evidence."),
 ],
 [
  ("A structure is light but unstable. What might improve it?", "Better bracing and load paths", "Removing every joint", "Painting it a darker color", "Bracing redistributes forces."),
  ("Two beams have the same material but different lengths. What should be controlled?", "Applied load and support conditions", "The engineer's clothing", "The title of the design", "Comparable tests need equivalent conditions."),
  ("An affordable material is less durable. Best decision method?", "Score cost, safety and durability", "Choose solely by price", "Choose by color", "Evaluate multiple design constraints."),
  ("A wheel slips on a ramp. What experiment targets the issue?", "Compare tire surface texture", "Move the starting line", "Change the poster text", "Surface grip influences traction."),
 ],
 [
  ("A critical component has a single point of failure. Best change?", "Introduce independent redundancy", "Add a decorative shell", "Hide the component", "Redundancy can reduce catastrophic failure risk."),
  ("A prototype meets strength goals but exceeds cost limits. What next?", "Explore lighter or cheaper alternatives and retest", "Ignore the cost target", "Claim all goals were met", "Engineering balances stated constraints."),
  ("When comparing 3 designs, which data supports selection?", "Consistent measured performance under equal conditions", "The team members' favorite color", "Which was drawn first", "Comparable measurement enables fair selection."),
  ("A design works in ideal weather but fails in rain. What was missing?", "Testing realistic environmental conditions", "A larger logo", "An earlier launch announcement", "Design verification should cover expected conditions."),
 ],
],
}

def enabled() -> bool:
    return (os.getenv("VERIFYD_CIRCLE_MISSION_ARCADE_ENABLED", "0").strip().lower() in _TRUE
            and rank_enabled())

def _identity(authorization):
    if not enabled():
        raise HTTPException(404, detail={"error": "ranked_missions_unavailable"})
    account = _identity_from_bearer(authorization)
    if account.get("status") != "active" or not account.get("id"):
        raise HTTPException(401, detail={"error": "session_required"})
    return str(account["id"])

def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        ensure_rank_schema()  # Avoid nested DB acquisition while holding a session row lock.
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_ranked_mission_sessions (
                session_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                game TEXT NOT NULL,
                level INTEGER NOT NULL CHECK(level BETWEEN 0 AND 2),
                questions_json JSONB NOT NULL,
                step INTEGER NOT NULL DEFAULT 0 CHECK(step BETWEEN 0 AND 4),
                correct_count INTEGER NOT NULL DEFAULT 0 CHECK(correct_count BETWEEN 0 AND 4),
                step_token TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '24 hours',
                completed_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_ranked_mission_sessions_identity
                ON verifyd_circle_ranked_mission_sessions(identity_id,created_at DESC)""")
        _READY = True

def _mars_bank(level):
    cells = 21 + level * 6
    crew = 3 + level
    each = 2 + level
    energy = 5 + 2 * level
    hours = 4 + level
    east = 11 + 3 * level
    west = 4 + level
    supply = 80 + 25 * level
    food = 17 + 6 * level
    repairs = 19 + 4 * level
    return [
      (f"The Mars habitat has {cells} oxygen units. {crew} crew members use {each} each. How many remain?",
       str(cells - crew * each), str(cells - each), str(cells + crew * each), "Subtract the crew's total oxygen from the supply."),
      (f"Solar panels gain {energy} battery units per hour for {hours} hours. How much is gained?",
       str(energy * hours), str(energy + hours), str(energy * (hours - 1)), "Multiply the hourly rate by elapsed hours."),
      (f"The rover travels {east} km east then {west} km west. How far east is it from the starting point?",
       str(east - west), str(east + west), str(west), "Find the net eastward distance."),
      (f"A rescue pod carries {supply} kg. It uses {food} kg food and {repairs} kg parts. How much capacity remains?",
       str(supply - food - repairs), str(supply - food), str(supply - repairs), "Subtract both items from available capacity."),
    ]

def _questions(game, level):
    bank = _mars_bank(level) if game == "mars-rescue" else BANK[game][level]
    if len(bank) != 4:
        raise RuntimeError("Each ranked mission must have four validated questions.")
    random = secrets.SystemRandom()
    order = list(bank)
    random.shuffle(order)
    result = []
    for prompt, correct, wrong_a, wrong_b, explanation in order:
        choices = [correct, wrong_a, wrong_b]
        if len(set(choices)) != 3:
            raise RuntimeError("Duplicate answer choices in mission bank.")
        random.shuffle(choices)
        result.append({"prompt": prompt, "choices": choices,
                       "correct": choices.index(correct), "explanation": explanation})
    return result

def _public_question(item, index):
    return {"step": index, "total_steps": 4, "prompt": item["prompt"],
            "options": item["choices"]}

class Start(BaseModel):
    game: Literal["mars-rescue", "media-detective", "debate-duel", "design-lab"]
    level: int = Field(ge=0, le=2)

class Answer(BaseModel):
    step: int = Field(ge=0, le=3)
    choice: int = Field(ge=0, le=2)
    step_token: str = Field(min_length=16, max_length=128)

@router.get("/catalog")
def catalog():
    return {"enabled": enabled(), "games": [
        {"id": key, "title": item["title"], "unlock": item["unlock"]}
        for key, item in GAMES.items()],
        "levels": list(LEVELS), "level_points": list(LEVEL_POINTS),
        "minimum_points": list(LEVEL_MINIMUMS), "rounds": 4,
        "pass_threshold": 3, "once_per_game_and_level": True,
        "method": "Server-held questions with session-verified answers; not an identity or clinical score."}

@router.post("/sessions", status_code=201)
def start(payload: Start, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _schema()
    required = max(GAMES[payload.game]["unlock"], LEVEL_MINIMUMS[payload.level])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT COALESCE(SUM(points),0) AS points FROM verifyd_circle_rank_awards
            WHERE identity_id=%s""", (user,))
        points = int(cur.fetchone()["points"])
        if points < required:
            raise HTTPException(403, detail={"error": "mission_locked", "required_points": required})
        cur.execute("""SELECT COUNT(*) AS n FROM verifyd_circle_ranked_mission_sessions
            WHERE identity_id=%s AND created_at > NOW()-INTERVAL '1 day'""", (user,))
        if int(cur.fetchone()["n"]) >= 30:
            raise HTTPException(429, detail={"error": "too_many_mission_sessions"})
        session_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        questions = _questions(payload.game, payload.level)
        cur.execute("""INSERT INTO verifyd_circle_ranked_mission_sessions
            (session_id,identity_id,game,level,questions_json,step_token)
            VALUES (%s,%s,%s,%s,%s::jsonb,%s)""",
            (session_id,user,payload.game,payload.level,json.dumps(questions),token))
    return {"session_id": session_id, "step_token": token,
            "question": _public_question(questions[0], 0),
            "ranked": True, "points_if_first_verified_win": LEVEL_POINTS[payload.level],
            "required_points": required}

@router.post("/sessions/{session_id}/answer")
def answer(session_id: str, payload: Answer, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT game,level,questions_json,step,correct_count,step_token,
                       expires_at,completed_at FROM verifyd_circle_ranked_mission_sessions
                       WHERE session_id=%s AND identity_id=%s FOR UPDATE""", (session_id,user))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(404, detail={"error": "mission_session_unavailable"})
        if row["completed_at"] is not None:
            raise HTTPException(409, detail={"error": "mission_already_completed"})
        if row["expires_at"] <= datetime.now(timezone.utc):
            raise HTTPException(410, detail={"error": "mission_session_expired"})
        index = int(row["step"])
        if index != payload.step or not secrets.compare_digest(row["step_token"],payload.step_token):
            raise HTTPException(409, detail={"error": "mission_step_mismatch"})
        questions = row["questions_json"]
        if isinstance(questions, str):
            questions = json.loads(questions)
        if index >= len(questions) or len(questions) != 4:
            raise HTTPException(409, detail={"error": "mission_question_mismatch"})
        current = questions[index]
        is_correct = payload.choice == current["correct"]
        score = int(row["correct_count"]) + int(is_correct)
        done = index + 1 == 4
        next_token = secrets.token_urlsafe(32)
        cur.execute("""UPDATE verifyd_circle_ranked_mission_sessions SET
                step=%s,correct_count=%s,step_token=%s,
                completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
                WHERE session_id=%s""",(index+1,score,next_token,done,session_id))
        awarded = 0
        already_earned = False
        passed = done and score >= 3
        if passed:
            challenge = f"mission:{row['game']}:{LEVELS[row['level']]}"
            cur.execute("""INSERT INTO verifyd_circle_rank_awards
                (award_id,identity_id,challenge_id,kind,points,evidence_ref)
                VALUES (%s,%s,%s,'game',%s,%s)
                ON CONFLICT DO NOTHING RETURNING award_id""",
                (str(uuid.uuid4()),user,challenge,LEVEL_POINTS[row["level"]],
                 f"ranked-mission-session:{session_id}"))
            awarded = LEVEL_POINTS[row["level"]] if cur.fetchone() else 0
            already_earned = awarded == 0
    return {"correct": is_correct, "explanation": current["explanation"],
            "completed": done, "passed": bool(passed), "correct_count": score,
            "total_steps": 4, "next_step_token": None if done else next_token,
            "next_question": None if done else _public_question(questions[index+1],index+1),
            "points_added": awarded, "already_earned": already_earned,
            "rank_updated": bool(awarded),
            "note": "Rank points reward an educational challenge, not medical or identity verification."}
