from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        extra="ignore",
    )

    database_url: str = Field(validation_alias="DATABASE_URL")
    artifacts_dir: str = "artifacts"
    artifact_version: str = "v1"
    chroma_dir: str = "chroma_data"
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server


settings = Settings()