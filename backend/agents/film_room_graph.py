"""LangGraph state machine for the Film Room Q&A agent.

Node flow:
  classify_intent → retrieve_relevant_events → retrieve_video_context
                  → analyze_sequence → validate_answer → generate_final_response
"""
from __future__ import annotations

import os
import re
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.agents.langchain_tools import (
    get_video_events,
    get_player_stats,
    retrieve_similar_events,
    summarize_video,
    get_event_by_timestamp,
)
from backend.agents.prompts import (
    SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    ANALYSIS_PROMPT,
    VALIDATION_PROMPT,
    FINAL_RESPONSE_PROMPT,
)
from backend.config import get_settings

logger = structlog.get_logger()


class FilmRoomState(TypedDict):
    video_id: str
    question: str
    run_id: str
    intent: str | None
    relevant_events: str | None
    video_context: str | None
    sequence_analysis: str | None
    validated_answer: str | None
    final_response: str | None
    evidence: list[dict]
    error: str | None


def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )


# ── Node implementations ───────────────────────────────────────────────────────

def classify_intent(state: FilmRoomState) -> FilmRoomState:
    """Classify the user's question into an intent category."""
    llm = _get_llm()
    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=state["question"])
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    valid_intents = {"event_query", "player_query", "summary_query", "clip_query", "tactical_query"}
    if intent not in valid_intents:
        intent = "event_query"

    logger.info("agent.classify_intent", intent=intent, question=state["question"][:80])
    return {**state, "intent": intent}


def retrieve_relevant_events(state: FilmRoomState) -> FilmRoomState:
    """Use tools to pull relevant events based on intent and question."""
    video_id = state["video_id"]
    intent = state["intent"]
    question = state["question"].lower()

    event_types = None
    if "shot" in question or "goal" in question:
        event_types = ["shot", "save"]
    elif "pass" in question:
        event_types = ["pass"]
    elif "turnover" in question or "possession" in question or "lost" in question:
        event_types = ["turnover"]
    elif "dribble" in question:
        event_types = ["dribble"]

    if intent == "player_query":
        events_text = get_video_events.invoke({"video_id": video_id, "limit": 30})
        player_text = get_player_stats.invoke({"video_id": video_id})
        combined = f"{events_text}\n\n{player_text}"
    elif intent == "summary_query":
        combined = summarize_video.invoke({"video_id": video_id})
    else:
        # Semantic similarity search + type filter
        similar = retrieve_similar_events.invoke({
            "video_id": video_id,
            "query_text": state["question"],
            "top_k": 8,
        })
        typed = get_video_events.invoke({
            "video_id": video_id,
            "event_types": event_types,
            "min_confidence": 0.6,
            "limit": 15,
        })
        combined = f"Semantic matches:\n{similar}\n\nType-filtered events:\n{typed}"

    return {**state, "relevant_events": combined}


def retrieve_video_context(state: FilmRoomState) -> FilmRoomState:
    """Fetch the full video summary for context."""
    summary = summarize_video.invoke({"video_id": state["video_id"]})
    return {**state, "video_context": summary}


def analyze_sequence(state: FilmRoomState) -> FilmRoomState:
    """LLM reasons over retrieved events and video context."""
    llm = _get_llm()
    prompt = ANALYSIS_PROMPT.format(
        video_context=state["video_context"] or "No context available.",
        events_context=state["relevant_events"] or "No events retrieved.",
        question=state["question"],
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return {**state, "sequence_analysis": response.content}


def validate_answer_against_evidence(state: FilmRoomState) -> FilmRoomState:
    """Check that the analysis only cites events that exist in the retrieved evidence."""
    llm = _get_llm()
    prompt = VALIDATION_PROMPT.format(
        answer=state["sequence_analysis"] or "",
        evidence=state["relevant_events"] or "No events available.",
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    validated = response.content.strip()

    # Extract event IDs cited for the response evidence field, then hydrate
    # exact timestamps/confidence from Postgres so API evidence is DB-grounded.
    evidence = _extract_evidence_from_text(state["relevant_events"] or "", state["video_id"])
    return {**state, "validated_answer": validated, "evidence": evidence}


def generate_final_response(state: FilmRoomState) -> FilmRoomState:
    """Format the validated answer as a polished coach-style response."""
    llm = _get_llm()
    prompt = FINAL_RESPONSE_PROMPT.format(
        analysis=state["sequence_analysis"] or "",
        validated_answer=state["validated_answer"] or "",
        intent=state["intent"] or "unknown",
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    grounded_response = _remove_ungrounded_timestamp_sentences(response.content, state["video_id"])
    grounded_response = _remove_ungrounded_track_references(grounded_response, state["video_id"])
    return {**state, "final_response": grounded_response}


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(FilmRoomState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_relevant_events", retrieve_relevant_events)
    graph.add_node("retrieve_video_context", retrieve_video_context)
    graph.add_node("analyze_sequence", analyze_sequence)
    graph.add_node("validate_answer_against_evidence", validate_answer_against_evidence)
    graph.add_node("generate_final_response", generate_final_response)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve_relevant_events")
    graph.add_edge("retrieve_relevant_events", "retrieve_video_context")
    graph.add_edge("retrieve_video_context", "analyze_sequence")
    graph.add_edge("analyze_sequence", "validate_answer_against_evidence")
    graph.add_edge("validate_answer_against_evidence", "generate_final_response")
    graph.add_edge("generate_final_response", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_film_room_agent(video_id: str, question: str, run_id: str) -> dict:
    """Entry point called by the FastAPI /ask endpoint."""
    settings = get_settings()

    # Configure LangSmith tracing if enabled
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    graph = _get_graph()
    initial_state: FilmRoomState = {
        "video_id": video_id,
        "question": question,
        "run_id": run_id,
        "intent": None,
        "relevant_events": None,
        "video_context": None,
        "sequence_analysis": None,
        "validated_answer": None,
        "final_response": None,
        "evidence": [],
        "error": None,
    }

    # LangGraph invoke is sync; run in thread pool to not block event loop
    import asyncio
    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(None, graph.invoke, initial_state)

    return {
        "answer": final_state.get("final_response") or final_state.get("validated_answer") or "No answer generated.",
        "intent": final_state.get("intent"),
        "evidence": final_state.get("evidence", []),
    }


def _extract_evidence_from_text(events_text: str, video_id: str) -> list[dict]:
    """Parse event IDs from tool output, then hydrate exact DB rows."""
    event_ids = []
    for match in re.finditer(r"(?:event_id|id)=([0-9a-fA-F-]{36})", events_text):
        event_id = match.group(1)
        if event_id not in event_ids:
            event_ids.append(event_id)

    if not event_ids:
        return []

    from sqlalchemy import create_engine, select
    from backend.db.models import EventPrediction

    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                EventPrediction.id,
                EventPrediction.event_type,
                EventPrediction.start_time_s,
                EventPrediction.end_time_s,
                EventPrediction.confidence,
                EventPrediction.nearest_track_id,
            )
            .where(EventPrediction.video_id == video_id)
            .where(EventPrediction.id.in_(event_ids))
        ).all()

    by_id = {str(row.id): row for row in rows}
    evidence = []
    for event_id in event_ids:
        row = by_id.get(event_id)
        if not row:
            continue
        evidence.append(
            {
                "event_id": str(row.id),
                "event_type": row.event_type,
                "start_time_s": float(row.start_time_s),
                "end_time_s": float(row.end_time_s),
                "confidence": float(row.confidence),
                "nearest_track_id": row.nearest_track_id,
            }
        )
    return evidence


def _extract_timestamps(text: str) -> list[float]:
    timestamps = []
    for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text):
        timestamps.append(int(match.group(1)) * 60 + int(match.group(2)))
    return timestamps


def _timestamp_exists_in_db(video_id: str, timestamp_s: float, window_s: float = 3.0) -> bool:
    from sqlalchemy import create_engine, text

    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT 1
                FROM event_predictions
                WHERE video_id = :video_id
                  AND start_time_s BETWEEN :timestamp_s - :window_s AND :timestamp_s + :window_s
                LIMIT 1
                """
            ),
            {"video_id": video_id, "timestamp_s": timestamp_s, "window_s": window_s},
        ).fetchone()
    return row is not None


def _remove_ungrounded_timestamp_sentences(answer: str, video_id: str) -> str:
    """Drop sentences containing timestamps that are not backed by event rows."""
    timestamps = _extract_timestamps(answer)
    if not timestamps:
        return answer

    ungrounded = {ts for ts in timestamps if not _timestamp_exists_in_db(video_id, ts)}
    if not ungrounded:
        return answer

    sentence_pattern = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
    kept = []
    for match in sentence_pattern.finditer(answer):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        sentence_timestamps = set(_extract_timestamps(sentence))
        if sentence_timestamps & ungrounded:
            continue
        kept.append(sentence)

    if kept:
        return " ".join(kept)
    return "I could not produce a timestamped answer that passed DB evidence validation."


def _event_track_ids(video_id: str) -> set[int]:
    from sqlalchemy import create_engine, select
    from backend.db.models import EventPrediction

    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            select(EventPrediction.nearest_track_id)
            .where(EventPrediction.video_id == video_id)
            .where(EventPrediction.nearest_track_id.is_not(None))
        ).all()
    return {int(row.nearest_track_id) for row in rows}


def _remove_ungrounded_track_references(answer: str, video_id: str) -> str:
    valid_track_ids = _event_track_ids(video_id)

    def replace(match: re.Match) -> str:
        track_id = int(match.group(1))
        if track_id in valid_track_ids:
            return match.group(0)
        return "the involved player"

    return re.sub(r"\bTrack ID\s+(\d+)\b", replace, answer)
