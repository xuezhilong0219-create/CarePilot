from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_api_key: str
    llm_base_url: str
    # 分类 / 抽取用快模型，生成回复用强模型
    llm_fast_model: str = "deepseek-v4.1-flash-1017"
    llm_main_model: str = "deepseek-v4.1-flash-1017"
    llm_timeout: float = 60
    llm_max_retries: int = 2

    kb_dir: Path = ROOT / "data" / "kb"
    checkpoint_db: Path = ROOT / "data" / "checkpoints.sqlite"
    retrieval_top_k: int = 4


def get_settings() -> Settings:
    return Settings()
