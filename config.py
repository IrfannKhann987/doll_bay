from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Unhabit backend.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,   # env var names are case-sensitive
        extra="ignore",
    )

    # these will read from ENV and DEBUG in env/system
    env: str = Field(default="local", alias="ENV")
    debug: bool = Field(default=False, alias="DEBUG")

    # this will read from OPENAI_API_KEY in env/system
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key",
        alias="OPENAI_API_KEY",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
