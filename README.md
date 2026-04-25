# AI Film Room

AI Film Room is an end-to-end soccer film analysis system that combines computer vision, temporal modeling, asynchronous processing, vector search, and agentic Q&A into one coherent film-room workflow.

The system takes a soccer video, extracts frames, detects players and the ball, tracks objects across time, engineers movement features, predicts event windows, stores structured evidence, generates short clips around important moments, and answers tactical questions with timestamped evidence from the database.

This project is designed as a backend-first AI systems portfolio piece. It is not presented as a production scouting product. The important engineering work is the full pipeline: video ingestion, CV processing, tracking, feature generation, model inference, evidence storage, clip generation, retrieval, and grounded natural-language analysis.

## What We Built

We built a working AI film-room pipeline with these major parts:

- A FastAPI backend for video upload, processing jobs, events, clips, Q&A, eval summaries, and dashboard delivery
- A Redis-backed Celery worker for long-running video analysis outside the request cycle
- A PostgreSQL database with pgvector for structured event storage and semantic retrieval
- An OpenCV frame extraction stage
- A YOLOv8 detection stage for players and the ball
- A Supervision/ByteTrack tracking stage for assigning stable track IDs over time
- A feature engineering stage that turns detections/tracks into 18-dimensional frame features
- A PyTorch LSTM classifier for soccer event windows
- An embedding stage that converts event descriptions into vectors for similarity search
- An FFmpeg clip generation stage that cuts short videos around detected events
- A LangGraph agent that retrieves evidence and answers tactical questions
- A lightweight FastAPI-served dashboard for inspecting the system visually
- Evaluation scripts for classifier behavior, retrieval quality, and answer groundedness

The result is not just a model or a demo UI. It is a complete applied AI system where each layer produces artifacts used by the next layer.

## System Walkthrough

### 1. Video Ingestion

A user uploads a soccer clip through the backend. The video is stored under the raw video directory and registered in the `videos` table with metadata such as filename, status, duration, FPS, frame size, and timestamps.

The backend does not process the video synchronously. Instead, it creates a `processing_jobs` row and enqueues a Celery task. This lets the API respond quickly while the worker handles the expensive CV and ML work in the background.

The video lifecycle is tracked through statuses such as:

- `uploaded`
- `processing`
- `done`
- `failed`

The job record tracks progress, current pipeline step, errors, start time, and finish time.

### 2. Asynchronous Worker Pipeline

The Celery worker owns the full analysis pipeline. The main worker task is `process_video`, which coordinates each stage in order:

```text
extract_frames
  -> detect_objects
  -> track_objects
  -> generate_features
  -> classify_events
  -> embed_events
  -> aggregate_player_stats
  -> generate_clips
```

The worker updates the job row throughout the run so the UI and API can show meaningful progress. Detection progress is updated at the frame-batch level, so long YOLO runs do not look frozen while processing.

Reprocessing is idempotent. Before a video is reprocessed, derived rows and generated artifacts are cleared so the system does not append duplicate frames, detections, events, stats, or clips.

### 3. Frame Extraction

The first CV stage uses OpenCV to read the uploaded video and extract sampled frames. The sample rate is configurable. Each extracted frame is saved under the processed data directory and inserted into the `frames` table with:

- video ID
- frame index
- timestamp in seconds
- file path

This gives every later detection and event a timestamped frame reference.

### 4. Object Detection With YOLOv8

The detection stage runs YOLOv8 over extracted frames in batches. The system uses COCO classes relevant to soccer analysis:

- `person` mapped to `player`
- `sports ball` mapped to `ball`

Each detection is represented with:

- frame index
- timestamp
- class name
- confidence
- bounding box coordinates

The pipeline stores detections in the database after tracking assigns IDs.

### 5. Object Tracking With ByteTrack

Detection alone only tells us what appears in a single frame. To reason over plays, we need continuity across time.

The tracking stage uses Supervision's ByteTrack integration to assign stable `track_id` values to detected players and the ball. Tracking is run over ordered frame detections, producing both updated detections and track summaries.

The system stores:

- per-frame detections with optional `track_id`
- summarized tracks with first seen time, last seen time, class name, and frame count

These track IDs make later possession estimates, player stats, and agent answers more useful.

### 6. Possession Estimation

The system uses a simple nearest-player heuristic to estimate possession. For each frame, it compares the ball center with nearby player centers and assigns possession to the closest player when the distance is within a threshold.

This is not meant to be a perfect soccer possession model. It is an interpretable MVP heuristic that provides useful features for event classification and player summaries.

The possession estimator produces:

- possessing track ID
- distance from ball to nearest player
- count of nearby players
- possession change signal

### 7. Feature Engineering

The feature pipeline converts detections, tracks, and possession estimates into an 18-dimensional vector per frame.

The feature vector includes:

- normalized ball position
- ball velocity
- nearest player position
- nearest player distance
- nearest player velocity estimate
- encoded possession player
- possession changed flag
- number of players near the ball
- field-zone indicators
- ball-near-goal indicator
- normalized time within window
- detection-confidence density

These features are intentionally compact and explainable. They are designed for a CPU-friendly temporal model and for clear interview discussion.

Feature arrays are saved to disk as `.npy` files with frame metadata JSON. This gives the ML layer reproducible inputs and gives DVC/MLflow something concrete to track.

### 8. Event Classification With PyTorch

The event classifier is a PyTorch LSTM that predicts soccer event types from sliding windows of frame features.

The model takes:

```text
batch x 30 frames x 18 features
```

It predicts one of six event classes:

- `pass`
- `shot`
- `dribble`
- `turnover`
- `save`
- `buildup`

The model architecture is:

```text
Bidirectional LSTM
  -> LayerNorm
  -> Dropout
  -> Linear classifier
```

The training setup includes:

- class-weighted cross entropy
- AdamW optimizer
- cosine learning-rate scheduling
- early stopping
- MLflow-compatible logging
- confusion matrix artifact generation

During inference, the event service slides a window over the feature matrix, scores each window, filters by confidence threshold, merges overlapping windows, and stores event predictions in the database.

Each event row includes:

- event type
- confidence
- start/end timestamps
- start/end frames
- nearest track ID when available
- metadata with top predicted classes
- vector embedding field

### 9. Synthetic Training Data Strategy

The classifier is trained on synthetic labeled feature windows. This is an intentional resume-MVP strategy.

The synthetic generator creates class-specific feature patterns for the six soccer event types. For example:

- shots have high ball velocity near goal zones
- turnovers emphasize possession changes
- dribbles keep the ball close to the same player
- buildup has slower movement and spread-out spacing

This gives us a real PyTorch training/evaluation loop without requiring a full manually annotated soccer dataset.

The honest interpretation is:

> The event classifier demonstrates the ML training, inference, artifact, and evaluation pipeline. Its current metrics are synthetic-label metrics, not proof of real-world soccer accuracy.

The rest of the application is wired around real video processing: frames, detections, tracking, features, events, embeddings, clips, and Q&A all operate through the same production-shaped pipeline.

The natural next step would be replacing the synthetic windows with SoccerNet or manually labeled match annotations.

### 10. Event Storage And pgvector

Structured event predictions are stored in Postgres. The database is the central evidence layer for the system.

Important tables include:

- `videos`
- `processing_jobs`
- `frames`
- `detections`
- `tracks`
- `event_predictions`
- `clips`
- `player_stats`
- `agent_runs`

The `event_predictions` table includes a `vector(1536)` embedding column. Each event can be converted into a text description and embedded for semantic search.

This enables queries like:

- find moments similar to a shot toward goal
- retrieve turnovers under pressure
- find buildup sequences
- rank semantically relevant events for a natural-language question

### 11. Clip Generation

The clip service uses FFmpeg to create short MP4 clips around detected event windows.

For each selected event, the service pads the event start/end time, cuts the corresponding segment from the original video, stores the clip on disk, and inserts a row into the `clips` table.

The backend exposes clip metadata through the API and streams raw video bytes through the clip stream endpoint.

This turns model predictions into reviewable media artifacts, which is what makes the project feel like a film-room tool rather than just an event table.

### 12. Player Stats

After tracking and possession estimation, the worker aggregates simple per-track stats.

The system computes:

- total tracked frames
- possession frames
- rough possession percentage
- territory zone heuristic
- event involvement structure

These stats are intentionally lightweight, but they give the agent and dashboard another structured context source.

### 13. LangGraph Agent

The Q&A layer is built as a LangGraph state machine rather than a single prompt call.

The graph follows this flow:

```text
classify_intent
  -> retrieve_relevant_events
  -> retrieve_video_context
  -> analyze_sequence
  -> validate_answer_against_evidence
  -> generate_final_response
```

The agent classifies questions into intents such as:

- event query
- player query
- summary query
- clip query
- tactical query

It then retrieves relevant database evidence using structured tools.

Available tools include:

- `get_video_events`
- `get_event_by_timestamp`
- `get_player_stats`
- `retrieve_similar_events`
- `get_clip_segment`
- `summarize_video`

The validation step checks that timestamped claims are grounded in real event rows. The API response also returns structured evidence objects so an answer can be traced back to event IDs, timestamps, confidence values, and track IDs.

This is the main agentic design choice: the LLM is used for tactical explanation, but the evidence comes from the database.

### 14. Dashboard

The dashboard is a thin inspection UI served directly by FastAPI. It is not a separate frontend application.

The dashboard exists to make the backend system easy to demonstrate. It supports:

- video upload
- process start
- live job progress
- video list
- event table
- clip playback
- model/eval status
- natural-language question box
- evidence panel

The design choice was intentional: keep the project backend/AI-system focused while still making the pipeline visible and usable.

### 15. Evaluation Layer

The project includes three evaluation surfaces.

#### Event Classifier Evaluation

The classifier eval computes precision, recall, F1, accuracy, and a confusion matrix over the available labeled windows.

For the current MVP, those labels are synthetic. The eval still matters because it proves the model training/evaluation machinery exists and produces artifacts.

#### Retrieval Evaluation

The retrieval eval uses curated query/event-type pairs and checks whether pgvector similarity search returns matching event types in the top results.

This tests the semantic event retrieval layer used by the agent.

#### Agent Groundedness Evaluation

The groundedness eval asks a set of tactical questions, extracts timestamps from the answers, and verifies that those timestamps exist in the database for the selected video.

This tests whether the final Q&A layer is citing real evidence rather than inventing moments.

## API Design

JSON endpoints use a consistent response envelope:

```json
{
  "data": {},
  "error": null,
  "meta": {}
}
```

Clip streaming is the exception because it returns raw `video/mp4` bytes.

The core API surface includes:

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard |
| `POST /videos/upload` | Store a new source video |
| `GET /videos` | List uploaded videos |
| `GET /videos/{video_id}` | Read video metadata |
| `GET /videos/{video_id}/status` | Poll video and job status |
| `POST /videos/{video_id}/process` | Enqueue background processing |
| `GET /videos/{video_id}/events` | List predicted events |
| `GET /videos/{video_id}/events/{event_id}` | Read a single event |
| `GET /videos/{video_id}/clips` | List generated clips |
| `GET /clips/{clip_id}/stream` | Stream a generated clip |
| `POST /videos/{video_id}/ask` | Ask a tactical question |
| `GET /jobs/{job_id}` | Inspect a background job |
| `GET /evals/summary` | Summarize model/eval artifacts |

## Data Model

The database schema is built around evidence and traceability.

`videos` stores source video metadata and lifecycle state.

`processing_jobs` tracks asynchronous work and user-visible progress.

`frames` maps extracted frame files to timestamps.

`detections` stores YOLO outputs and track IDs.

`tracks` summarizes object trajectories.

`event_predictions` stores model outputs, timestamps, confidence, metadata, and embeddings.

`clips` links generated media segments back to events.

`player_stats` stores per-track aggregate summaries.

`agent_runs` stores questions, answers, evidence references, latency, and tracing metadata.

This schema gives the system a real audit trail from answer back to event, event back to frames, and frames back to source video.

## Project Structure

```text
backend/
  main.py                       FastAPI app and dashboard route
  api/                          route modules for videos, events, clips, ask, jobs, evals
  agents/                       LangGraph workflow, prompts, and tools
  db/                           SQLAlchemy models and Alembic migrations
  services/                     clip generation, embeddings, event inference, stats
  ui/index.html                 dashboard
  workers/process_video_job.py  Celery video-processing pipeline

cv/
  extract_frames.py             OpenCV frame extraction
  detect_objects.py             YOLOv8 detection
  track_objects.py              ByteTrack integration
  estimate_possession.py        nearest-player possession heuristic
  generate_features.py          18-dimensional feature generation

ml/
  models/lstm_classifier.py     PyTorch LSTM model
  train.py                      training loop and artifact logging
  evaluate.py                   metrics and confusion matrix helpers
  inference.py                  model loading and sliding-window inference
  datasets/                     synthetic and feature-window datasets

evals/
  eval_event_classifier.py      classifier evaluation
  eval_retrieval.py             pgvector retrieval evaluation
  eval_agent_groundedness.py    answer timestamp groundedness evaluation

scripts/
  demo_check.py                 local system verification script

data/
  raw_videos/                   source videos
  processed/                    extracted frames
  features/                     real and synthetic feature arrays
  clips/                        generated event clips
```

## Engineering Decisions

### Celery Instead Of Inline Processing

Video processing is too expensive for a normal HTTP request. Celery lets the API enqueue work, return quickly, and expose job progress while the worker performs long-running CV and ML tasks.

### LSTM Before Transformer

The LSTM is simple, CPU-friendly, fast to train, and easy to explain. For this MVP, interpretability and iteration speed matter more than chasing a larger architecture.

### pgvector For Evidence Retrieval

Vector search lets the system answer fuzzy tactical questions like “where did we lose the ball under pressure?” instead of only exact event-type filters.

### LangGraph Instead Of A Single Chain

The agent has explicit stages for intent classification, retrieval, analysis, validation, and final response. This makes the workflow easier to debug and gives a clear place to add groundedness checks.

### Thin Dashboard Instead Of A Full Frontend App

The project is about AI/backend systems. A FastAPI-served dashboard gives enough visual polish for a portfolio demo without turning the project into a frontend build.

### Synthetic Labels With Clear Boundaries

Synthetic labels make the ML pipeline demonstrable without pretending to have a production-quality annotated sports dataset. The README and eval artifacts explicitly mark the limits of those metrics.

## What This Demonstrates

This project demonstrates practical AI systems engineering across the full stack:

- computer vision ingestion and detection
- multi-object tracking
- spatiotemporal feature design
- PyTorch sequence modeling
- asynchronous job orchestration
- relational evidence modeling
- vector similarity search
- generated media artifacts
- agentic Q&A with validation
- API design and dashboard delivery
- honest evaluation boundaries

The main point is not that the model is production-accurate today. The main point is that the system architecture is complete enough to show how a real AI film-room product would be assembled and where real labeled data would improve it.

## Path To A Stronger Version

The clearest upgrade is replacing synthetic labels with real annotations.

A stronger next version would:

- use SoccerNet or manually labeled match clips
- train/evaluate on real event windows
- report real per-class precision, recall, and F1
- add visual overlays for detections and tracks
- improve possession modeling beyond nearest-player distance
- add team assignment and formation context
- add clip-level evidence links directly inside agent answers

The foundation is already built for those upgrades: video ingestion, CV processing, features, model inference, event storage, retrieval, clips, and Q&A are all connected.
