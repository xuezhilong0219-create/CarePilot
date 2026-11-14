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
    # bm25（默认）| vector | hybrid | hybrid_rerank_ce | hybrid_rerank_llm。
    # 默认 bm25：评估里客户习惯贴报错原文、平台名搜，BM25 本来就强；混合检索的提升在 test 半边不显著，
    # 却要多装 1 个本地 embedding 模型、预建 20 分钟的索引。要启用见 README 的「检索策略」一节。
    retriever_kind: str = "bm25"
    index_dir: Path | None = None
    # 治理模式（工单存储 + 工具网关 + 沙箱）用到的数据文件
    tickets_db: Path = ROOT / "data" / "tickets.sqlite"
    gateway_db: Path = ROOT / "data" / "gateway.sqlite"
    trace_dir: Path = ROOT / "data" / "traces"


def get_settings() -> Settings:
    return Settings()
