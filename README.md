# AI Film Room

An end-to-end AI sports film analysis system for soccer match footage. Detects and tracks players and the ball using YOLO + ByteTrack, classifies tactical events with a PyTorch LSTM, stores structured evidence in PostgreSQL with pgvector, and answers natural-language coaching questions via a LangGraph reasoning agent.

**Stack:** Python · OpenCV · YOLOv8 · Supervision/ByteTrack · PyTorch · FastAPI · PostgreSQL + pgvector · Redis · Celery · LangChain · LangGraph · LangSmith · MLflow · DVC · Docker

---

## System Architecture

```
User → FastAPI Backend → Redis Queue → Celery Worker
                                          ├─ extract_frames (OpenCV)
                                          ├─ detect_objects (YOLOv8)
                                          ├─ track_objects (ByteTrack)
                                          ├─ generate_features (18-dim vectors)
                                          ├─ classify_events (PyTorch LSTM)
                                          ├─ embed_events (pgvector)
                                          └─ generate_clips (FFmpeg)

User question → FastAPI → LangGraph Agent
                           ├─ classify_intent
                           ├─ retrieve_relevant_events (LangChain tools)
                           ├─ retrieve_video_context
                           ├─ analyze_sequence (GPT-4o-mini)
                           ├─ validate_answer_against_evidence
                           └─ generate_final_response
```

## Quick Start

### Prerequisites
- Docker + Docker Compose
- FFmpeg (included in Docker image)
- OpenAI API key (for embeddings + LLM; local fallback available)

### 1. Clone and configure
```bash
git clone <repo>
cd ai-film-room
cp .env.example .env
# Edit .env: set OPENAI_API_KEY
```

### 2. Start services
```bash
docker compose up -d postgres redis
docker compose up backend worker
```

### 3. Run database migrations
```bash
docker compose exec backend alembic upgrade head
```

### 4. Upload and process a video
```bash
# Upload
curl -X POST http://localhost:8000/videos/upload \
  -F "file=@match_clip.mp4"
# → {"data": {"id": "video-uuid", "status": "uploaded"}, "error": null, "meta": {}}

# Start processing
curl -X POST http://localhost:8000/videos/{video_id}/process
# → {"video_id": "...", "job_id": "..."}

# Poll status
curl http://localhost:8000/videos/{video_id}/status
```

### 5. Query the agent
```bash
curl -X POST http://localhost:8000/videos/{video_id}/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When did the attack break down?"}'
```

Example response:
```json
{
  "data": {
    "answer": "The attack broke down at 00:24 when possession changed near the right sideline...",
    "intent": "event_query",
    "evidence": [{"event_id": "...", "event_type": "turnover", "start_time_s": 24.1, "confidence": 0.81}],
    "latency_ms": 1842
  },
  "error": null,
  "meta": {}
}
```

### Demo check
```bash
open http://localhost:8000
python scripts/demo_check.py
python scripts/demo_check.py --ask
```

### 6. Optional: Start dev tools
```bash
docker compose --profile dev up flower mlflow
# Celery Flower: http://localhost:5555
# MLflow UI:     http://localhost:5001
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/videos/upload` | Upload video file |
| `GET` | `/videos/{id}` | Get video metadata |
| `GET` | `/videos/{id}/status` | Poll processing status |
| `POST` | `/videos/{id}/process` | Start background processing |
| `GET` | `/videos/{id}/events` | List detected events |
| `GET` | `/videos/{id}/events/{event_id}` | Get single event |
| `GET` | `/videos/{id}/clips` | List generated clips |
| `GET` | `/clips/{clip_id}/stream` | Stream clip video |
| `POST` | `/videos/{id}/ask` | Natural language Q&A |
| `GET` | `/jobs/{job_id}` | Check job status |
| `GET` | `/evals/summary` | Model/eval artifact summary |

Interactive docs: `http://localhost:8000/docs`

---

## Current MVP Status

Implemented:

- FastAPI upload/status/process/events/clips/ask/job routes
- Docker Compose stack for FastAPI, Celery worker, Postgres/pgvector, Redis, Flower, and MLflow
- OpenCV frame extraction from real video
- YOLOv8 detection wrapper and Supervision ByteTrack tracking wrapper
- 18-dimensional feature generation and persisted feature artifacts
- PyTorch LSTM classifier, synthetic data generator, training loop, evaluation, and saved model artifact
- Celery video-processing orchestration through CV, features, event classification, embeddings, player stats, and clips
- LangGraph Q&A skeleton with structured retrieval tools and answer validation step
- Alembic initial schema for videos, jobs, frames, detections, tracks, events, clips, player stats, and agent runs

Known demo gaps:

- Full upload → process should be rerun before recording a final screencast so the generated artifacts are fresh from one continuous job.
- The LSTM model metrics are synthetic-label metrics, not SoccerNet/manual-label metrics.
- LangSmith tracing is optional; keep `LANGCHAIN_TRACING_V2=false` locally unless a valid LangSmith key is configured.
- Integration tests are mostly scaffolded and require a running Docker stack; local unit tests require the project Python dependencies installed.

Latest local demo artifacts:

- 10 event predictions stored for `match_clip.mp4`
- 10 event clips generated and streamable through `/clips/{clip_id}/stream`
- 10/10 event embeddings populated for pgvector retrieval
- Classifier eval written to `ml/artifacts/metrics.json`
- Retrieval eval written to `ml/artifacts/retrieval_eval.json` with top-3 score `0.556`
- Agent groundedness eval written to `ml/artifacts/agent_groundedness_eval.json` with score `1.0`

---

## ML Pipeline

### Portfolio Data Strategy

This MVP uses a pragmatic hybrid data setup:

- `data/features/synthetic/` contains 7,200 generated training windows, balanced across the six event classes. These simulate labeled spatiotemporal features that would normally come from SoccerNet/manual annotation.
- `ml/artifacts/best_model.pth` is a trained LSTM artifact produced from that feature dataset, with `ml/artifacts/confusion_matrix.png` saved as a training evidence artifact.
- `data/raw_videos/match_clip.mp4` and `data/raw_videos/<video_id>/match_clip.mp4` provide a real video input for the CV pipeline.
- `data/processed/<video_id>/frames/` and `data/features/<video_id>/` show the real-video path producing extracted frames and 18-dimensional feature rows for inference.

For a resume/demo build, the honest claim is: **the event classifier is trained on synthetic feature labels, while the upload, CV extraction, detection/tracking, feature generation, async processing, database storage, and agent surfaces are wired around real video inputs.** Replacing the synthetic windows with SoccerNet/manual labels is the main step toward production-grade model validity.

### Feature Engineering (18-dim per frame)
Ball position/velocity, nearest player position/velocity/distance, possession player encoding, possession change, players near ball, field zone indicators, goal proximity, time-in-window, frame detection density.

### LSTM Event Classifier
- 2-layer bidirectional LSTM, hidden_size=128
- Input: 30-frame windows × 18 features
- Output: 6 event classes (pass, shot, dribble, turnover, save, buildup)
- Training: AdamW + CosineAnnealingLR + class-weighted CrossEntropy + early stopping

### Training
```bash
# After running CV pipeline to generate features:
python ml/train.py --params params.yaml

# Evaluate:
python evals/eval_event_classifier.py
```

MLflow experiment tracking at `http://localhost:5001`. Logs: loss curves, per-class F1, confusion matrix, model artifact. The standalone eval writes `ml/artifacts/metrics.json`; for the current portfolio MVP this evaluates the synthetic labeled windows.

---

## DVC Pipeline
```bash
dvc init
dvc run  # runs full extract → detect → features → train → evaluate pipeline
dvc repro # reproduce from changed stage
dvc params diff  # compare params
```

Stages: `extract_frames → detect → generate_features → train → evaluate`

---

## LangGraph Agent Workflow

```
classify_intent → retrieve_relevant_events → retrieve_video_context
               → analyze_sequence → validate_answer_against_evidence
               → generate_final_response
```

**Validation node** checks every cited timestamp exists in the DB — reduces hallucinations.

**LangSmith** traces each run end-to-end. Set `LANGCHAIN_TRACING_V2=true` in `.env`.

### Available Tools
- `get_video_events` — filtered event list with timestamps
- `get_event_by_timestamp` — nearest event to a timestamp
- `get_player_stats` — possession%, territory zones
- `retrieve_similar_events` — pgvector cosine similarity search
- `get_clip_segment` — find clip covering a time range
- `summarize_video` — event breakdown + top players

---

## Testing
```bash
# Unit + smoke tests (no services needed)
pytest tests/unit/ tests/smoke/ -v

# With coverage
pytest tests/unit/ --cov=cv --cov=ml --cov-report=term-missing

# Integration tests (requires running Docker stack)
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

---

## Evaluation Scripts
```bash
# Event classifier: F1, precision, recall, confusion matrix
python evals/eval_event_classifier.py

# Retrieval quality: precision@K for semantic search
python evals/eval_retrieval.py --video-id <uuid>

# Agent groundedness: % answers with all timestamps verified in DB
python evals/eval_agent_groundedness.py --video-id <uuid>
```

Current local eval outputs:

- `ml/artifacts/metrics.json`: synthetic-label classifier metrics and caveat
- `ml/artifacts/retrieval_eval.json`: semantic retrieval precision checks
- `ml/artifacts/agent_groundedness_eval.json`: timestamp groundedness checks

---

## Getting Real Data

For meaningful model metrics, use [SoccerNet](https://www.soccer-net.org/):
1. Register at soccer-net.org for free video access
2. Download 2-3 match clips to `data/raw_videos/`
3. Use their event annotations to populate `data/annotations/{video_id}/labels.json`
4. Run `dvc repro` to regenerate features and retrain

---

## Resume Bullets This Project Demonstrates

- Built end-to-end AI sports film analysis system using PyTorch, YOLOv8, LangGraph, FastAPI, PostgreSQL/pgvector, Celery, MLflow, and DVC to detect soccer events, classify play sequences, and generate coach-style tactical analysis from match footage
- Trained a bidirectional PyTorch LSTM on 18-dimensional spatiotemporal player/ball movement features to classify soccer events across 6 classes, tracking precision, recall, F1, and confusion matrices via MLflow experiment tracking
- Designed a 6-node LangGraph reasoning workflow with intent classification, semantic event retrieval (pgvector), tactical sequence analysis, answer validation against DB evidence, and final response generation to reduce hallucinations in video Q&A
- Implemented ByteTrack object tracking via Supervision over YOLOv8 detections to maintain stable player/ball IDs across frames, enabling per-player possession estimation and spatiotemporal feature engineering
- Built DVC pipeline versioning raw videos, frame features, model artifacts, and evaluation metrics end-to-end with reproducible stage definitions and parameter management
- Containerized full system with Docker Compose (FastAPI, Celery workers, PostgreSQL+pgvector, Redis, MLflow) with GitHub Actions CI running lint, type checks, unit/smoke tests, and Docker build validation

---

## Project Structure

```
ai-film-room/
├── backend/         FastAPI app, API routes, services, LangGraph agent
├── cv/              OpenCV frame extraction, YOLO detection, ByteTrack, features
├── ml/              PyTorch LSTM, training loop, evaluation, inference
├── evals/           Classifier, retrieval, and agent groundedness evaluations
├── tests/           Unit, smoke, and integration tests
├── data/            DVC-tracked: raw_videos/, features/, annotations/, clips/
├── docker/          Dockerfiles and DB init script
├── dvc.yaml         DVC pipeline stages
├── params.yaml      All tunable parameters
└── docker-compose.yml
```
# AI-Sports-Intelligence
