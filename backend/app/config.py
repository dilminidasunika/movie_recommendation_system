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
    app_env: str = Field("dev", validation_alias="APP_ENV")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"], validation_alias="ALLOWED_HOSTS")
    artifacts_dir: str = "artifacts"
    artifact_version: str = "v1"
    chroma_dir: str = "chroma_data"
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server

    # Chroma collections (swap tags_collection after a rebuild)
    plots_collection: str = "plots_embeddings"
    tags_collection: str = "tags_vectors"

    # Retrieval tunables (tuned in step 53)
    chip_weight: float = 1.0
    exact_limit: int = 300_000
    ann_k: int = 2000
    pool_cap: int = 2000
    use_uncapped_pool: bool = True
    use_matvec_tag_candidates: bool = True
    k_tag: int = 10_000
    k_plot: int = 5_000
    max_results: int = 200
    min_survivors: int = 1000
    starvation_hard_cap: int = 300_000
    tie_break_plot: bool = True
    tag_dense_matvec_threshold: int = 200_000
    suppress_boilerplate_plots: bool = True

    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"   # check AI Studio for the current model id
    explain_timeout_s: float = 20.0

    @staticmethod
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def artifact_path(self) -> Path:
        return self._resolve(self.artifacts_dir) / self.artifact_version

    @property
    def chroma_path(self) -> Path:
        return self._resolve(self.chroma_dir)


settings = Settings()
