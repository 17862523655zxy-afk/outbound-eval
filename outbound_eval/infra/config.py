"""Configuration management."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")

    # Database
    database_url: str = Field(default="sqlite:///./data/eval.db", description="Database URL")

    # LLM
    llm_model: str = Field(default="deepseek-chat", description="LLM model for Agent")
    judge_llm_model: str = Field(
        default="deepseek-chat", description="LLM model for Judge"
    )
    base_url: str = Field(
        default="https://api.deepseek.com", description="API base URL (without /v1 path)"
    )
    token_price_per_1k: float = Field(default=0.0001, description="Token price per 1K tokens")

    # Dashboard
    dashboard_host: str = Field(default="0.0.0.0", description="Dashboard host")
    dashboard_port: int = Field(default=8000, description="Dashboard port")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    # Evaluation
    max_turns_per_call: int = Field(default=20, description="Max turns per call")
    target_turns: float = Field(default=8.0, description="Target turns per call")
    target_cost_per_success: float = Field(default=0.1, description="Target cost per success")
    bootstrap_n: int = Field(default=10000, description="Bootstrap iterations")


settings = Settings()