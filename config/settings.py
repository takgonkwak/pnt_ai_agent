from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM (OpenAI)
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 4096

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "mes-ai-agent"

    # MES Database
    mes_db_host: str = "localhost"
    mes_db_port: int = 1521
    mes_db_name: str = "MESDB"
    mes_db_user: str = ""
    mes_db_password: str = ""
    mes_db_type: str = "oracle"

    # Log System (Elasticsearch)
    es_enabled: bool = False
    es_host: str = "localhost"
    es_port: int = 9200
    es_scheme: str = "http"
    es_user: str = ""
    es_password: str = ""
    es_index_pattern: str = "mes-logs-*"
    log_max_search_results: int = 50

    # Source Code Repository (GitHub)
    source_enabled: bool = False
    github_token: str = ""
    github_owner: str = ""
    github_repo: str = ""
    github_branch: str = "main"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "mes_knowledge"

    # Slack
    slack_bot_token: str = ""
    slack_channel_mes_support: str = "#mes-support"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "dev-secret-key"

    # Agent Behavior
    max_clarification_turns: int = 2
    answer_confidence_threshold: float = 0.4
    log_search_window_minutes: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
