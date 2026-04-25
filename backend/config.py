from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    environment: str = "development"
    data_dir: Path = Path("./data")
    max_video_size_mb: int = 500
    frame_sample_rate: int = 3  # extract every Nth frame
    event_confidence_threshold: float = 0.65

    # Database
    database_url: str = "postgresql+asyncpg://filmroom:filmroom@localhost:5432/filmroom"
    sync_database_url: str = "postgresql://filmroom:filmroom@localhost:5432/filmroom"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # OpenAI
    openai_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-film-room"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    # Models
    yolo_model: str = "yolov8n.pt"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Derived paths
    @property
    def raw_videos_dir(self) -> Path:
        return self.data_dir / "raw_videos"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def models_dir(self) -> Path:
        return Path("./ml/artifacts")

    def ensure_dirs(self) -> None:
        for d in [self.raw_videos_dir, self.processed_dir, self.features_dir, self.clips_dir]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
