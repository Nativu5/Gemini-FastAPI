import os
import sys
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_PATH = "config/config.yaml"


class HTTPSConfig(BaseModel):
    """HTTPS configuration"""

    enabled: bool = Field(default=False, description="Enable HTTPS")
    key_file: str = Field(default="certs/privkey.pem", description="SSL private key file path")
    cert_file: str = Field(default="certs/fullchain.pem", description="SSL certificate file path")


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication, if set, will enable API key validation",
    )
    https: HTTPSConfig = Field(default=HTTPSConfig(), description="HTTPS configuration")


class GeminiClientSettings(BaseModel):
    """Credential set for one Gemini client."""

    id: str = Field(..., description="Unique identifier for the client")
    secure_1psid: str = Field(..., description="Gemini Secure 1PSID")
    secure_1psidts: str = Field(..., description="Gemini Secure 1PSIDTS")


class GeminiConfig(BaseModel):
    """Gemini API configuration"""

    clients: list[GeminiClientSettings] = Field(
        ..., description="List of Gemini client credential pairs"
    )
    timeout: int = Field(default=60, ge=1, description="Init timeout")
    auto_refresh: bool = Field(True, description="Enable auto-refresh for Gemini cookies")
    refresh_interval: int = Field(
        default=540, ge=1, description="Interval in seconds to refresh Gemini cookies"
    )
    verbose: bool = Field(False, description="Enable verbose logging for Gemini API requests")
    max_chars_per_request: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum characters Gemini Web can accept per request",
    )


class CORSConfig(BaseModel):
    """CORS configuration"""

    enabled: bool = Field(default=True, description="Enable CORS support")
    allow_origins: list[str] = Field(
        default=["*"], description="List of allowed origins for CORS requests"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials in CORS requests")
    allow_methods: list[str] = Field(
        default=["*"], description="List of allowed HTTP methods for CORS requests"
    )
    allow_headers: list[str] = Field(
        default=["*"], description="List of allowed headers for CORS requests"
    )


class StorageConfig(BaseModel):
    """LMDB Storage configuration"""

    path: str = Field(
        default="data/lmdb",
        description="Path to the storage directory where data will be saved",
    )
    max_size: int = Field(
        default=1024**2 * 256,  # 256 MB
        ge=1,
        description="Maximum size of the storage in bytes",
    )
    retention_days: int = Field(
        default=14,
        ge=0,
        description="Number of days to retain conversations before automatic cleanup (0 disables cleanup)",
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
        description="Logging level",
    )


class Config(BaseSettings):
    """Application configuration"""

    # Server configuration
    server: ServerConfig = Field(
        default=ServerConfig(),
        description="Server configuration, including host, port, and API key",
    )

    # CORS configuration
    cors: CORSConfig = Field(
        default=CORSConfig(),
        description="CORS configuration, allows cross-origin requests",
    )

    # Gemini API configuration
    gemini: GeminiConfig = Field(..., description="Gemini API configuration, must be set")

    storage: StorageConfig = Field(
        default=StorageConfig(),
        description="Storage configuration, defines where and how data will be stored",
    )

    # Logging configuration
    logging: LoggingConfig = Field(
        default=LoggingConfig(),
        description="Logging configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="CONFIG_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        yaml_file=os.getenv("CONFIG_PATH", CONFIG_PATH),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Read settings: env -> yaml -> default"""
        return (
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def extract_gemini_clients_env() -> dict[int, dict[str, str]]:
    """Extract and remove all Gemini clients related environment variables, return a mapping from index to field dict."""
    prefix = "CONFIG_GEMINI__CLIENTS__"
    env_overrides: dict[int, dict[str, str]] = {}
    to_delete = []
    for k, v in os.environ.items():
        if k.startswith(prefix):
            parts = k.split("__")
            if len(parts) < 4:
                continue
            index_str, field = parts[2], parts[3].lower()
            if not index_str.isdigit():
                continue
            idx = int(index_str)
            env_overrides.setdefault(idx, {})[field] = v
            to_delete.append(k)
    # Remove these environment variables to avoid Pydantic parsing errors
    for k in to_delete:
        del os.environ[k]
    return env_overrides


def _merge_clients_with_env(
    base_clients: list[GeminiClientSettings] | None,
    env_overrides: dict[int, dict[str, str]],
):
    """Override base_clients with env_overrides, return the new clients list."""
    if not env_overrides:
        return base_clients
    result_clients: list[GeminiClientSettings] = []
    if base_clients:
        result_clients = [client.model_copy() for client in base_clients]
    for idx in sorted(env_overrides):
        overrides = env_overrides[idx]
        if idx < len(result_clients):
            client_dict = result_clients[idx].model_dump()
            client_dict.update(overrides)
            result_clients[idx] = GeminiClientSettings(**client_dict)
        elif idx == len(result_clients):
            new_client = GeminiClientSettings(**overrides)
            result_clients.append(new_client)
        else:
            raise IndexError(f"Client index {idx} in env is out of range.")
    return result_clients if result_clients else base_clients


def initialize_config() -> Config:
    """
    Initialize the configuration.

    Returns:
        Config: Configuration object
    """
    try:
        # First, extract and remove Gemini clients related environment variables
        env_clients_overrides = extract_gemini_clients_env()

        # Then, initialize Config with pydantic_settings
        config = Config()  # type: ignore

        # Synthesize clients
        config.gemini.clients = _merge_clients_with_env(
            config.gemini.clients, env_clients_overrides
        )  # type: ignore

        return config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e!s}")
        sys.exit(1)
