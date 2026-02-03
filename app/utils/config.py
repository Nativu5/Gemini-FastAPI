import ast
import os
import sys
from typing import Any, Literal, Optional

import orjson
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator
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
    proxy: Optional[str] = Field(default=None, description="Proxy URL for this Gemini client")

    @field_validator("proxy", mode="before")
    @classmethod
    def _blank_proxy_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class GemDefinition(BaseModel):
    """Preset configuration applied when a request selects `model=gem:<id>`."""

    id: str = Field(..., description="Unique identifier for the gem")
    model: str = Field(..., description="Model name to use when this gem is selected")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt injected before user messages",
    )
    tool_policy: Literal["allow", "disallow", "auto"] = Field(
        default="allow",
        description="Tool handling policy for this gem",
    )
    default_temperature: float = Field(
        ..., ge=0.0, le=2.0, description="Default temperature override"
    )
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p override")
    max_output_tokens: int = Field(..., ge=1, description="Maximum output tokens override")

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _blank_system_prompt_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class GeminiModelConfig(BaseModel):
    """Configuration for a custom Gemini model."""

    model_name: Optional[str] = Field(default=None, description="Name of the model")
    model_header: Optional[dict[str, Optional[str]]] = Field(
        default=None, description="Header for the model"
    )

    @field_validator("model_header", mode="before")
    @classmethod
    def _parse_json_string(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                return orjson.loads(v)
            except orjson.JSONDecodeError:
                # Return the original value to let Pydantic handle the error or type mismatch
                return v
        return v


class GeminiConfig(BaseModel):
    """Gemini API configuration"""

    clients: list[GeminiClientSettings] = Field(
        ..., description="List of Gemini client credential pairs"
    )
    models: list[GeminiModelConfig] = Field(
        default_factory=list,
        description="List of custom Gemini models",
    )
    model_strategy: Literal["append", "overwrite"] = Field(
        default="append",
        description="Strategy for loading models: 'append' merges custom with default, 'overwrite' uses only custom",
    )
    gems: list[GemDefinition] = Field(
        default_factory=list,
        description="Gem presets selectable via request model alias (model=gem:<id>)",
    )
    timeout: int = Field(default=120, ge=1, description="Init timeout")
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

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models_json(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return orjson.loads(v)
            except orjson.JSONDecodeError as e:
                logger.warning(f"Failed to parse models JSON string: {e}")
                return v
        return v

    @field_validator("models")
    @classmethod
    def _filter_valid_models(cls, v: list[GeminiModelConfig]) -> list[GeminiModelConfig]:
        """Filter out models that don't have all required fields set."""
        valid_models = []
        for model in v:
            if model.model_name and model.model_header:
                valid_models.append(model)
            else:
                missing = []
                if not model.model_name:
                    missing.append("model_name")
                if not model.model_header:
                    missing.append("model_header")
                logger.warning(
                    f"Discarding custom model due to missing {', '.join(missing)}: {model}"
                )
        return valid_models

    @field_validator("gems")
    @classmethod
    def _validate_unique_gems(cls, value: list[GemDefinition]) -> list[GemDefinition]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for gem in value:
            if gem.id in seen:
                duplicates.add(gem.id)
            seen.add(gem.id)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate gem id(s) in config: {joined}")
        return value


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

    def get_gem(self, gem_id: str | None) -> GemDefinition | None:
        """Return the configured gem preset by id (used by `model=gem:<id>`)."""
        if not gem_id:
            return None
        for gem in self.gemini.gems:
            if gem.id == gem_id:
                return gem
        return None

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
            raise IndexError(
                f"Client index {idx} in env is out of range (current count: {len(result_clients)}). "
                "Client indices must be contiguous starting from 0."
            )
    return result_clients if result_clients else base_clients


def extract_gemini_models_env() -> dict[int, dict[str, Any]]:
    """Extract and remove all Gemini models related environment variables, supporting nested fields."""
    root_key = "CONFIG_GEMINI__MODELS"
    env_overrides: dict[int, dict[str, Any]] = {}

    if root_key in os.environ:
        val = os.environ[root_key]
        models_list = None
        parsed_successfully = False

        try:
            models_list = orjson.loads(val)
            parsed_successfully = True
        except orjson.JSONDecodeError:
            try:
                models_list = ast.literal_eval(val)
                parsed_successfully = True
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse {root_key} as JSON or Python literal: {e}")

        if parsed_successfully and isinstance(models_list, list):
            for idx, model_data in enumerate(models_list):
                if isinstance(model_data, dict):
                    env_overrides[idx] = model_data

            # Remove the environment variable to avoid Pydantic parsing errors
            del os.environ[root_key]

    return env_overrides


def _merge_models_with_env(
    base_models: list[GeminiModelConfig] | None,
    env_overrides: dict[int, dict[str, Any]],
):
    """Override base_models with env_overrides using standard update (replace whole fields)."""
    if not env_overrides:
        return base_models or []
    result_models: list[GeminiModelConfig] = []
    if base_models:
        result_models = [model.model_copy() for model in base_models]

    for idx in sorted(env_overrides):
        overrides = env_overrides[idx]
        if idx < len(result_models):
            # Update existing model: overwrite fields found in env
            model_dict = result_models[idx].model_dump()
            model_dict.update(overrides)
            result_models[idx] = GeminiModelConfig(**model_dict)
        elif idx == len(result_models):
            # Append new models
            new_model = GeminiModelConfig(**overrides)
            result_models.append(new_model)
        else:
            raise IndexError(
                f"Model index {idx} in env is out of range (current count: {len(result_models)}). "
                "Model indices must be contiguous starting from 0."
            )
    return result_models


def initialize_config() -> Config:
    """
    Initialize the configuration.

    Returns:
        Config: Configuration object
    """
    try:
        # First, extract and remove Gemini clients related environment variables
        env_clients_overrides = extract_gemini_clients_env()
        # Extract and remove Gemini models related environment variables
        env_models_overrides = extract_gemini_models_env()

        # Then, initialize Config with pydantic_settings
        config = Config()  # type: ignore

        # Synthesize clients
        config.gemini.clients = _merge_clients_with_env(
            config.gemini.clients, env_clients_overrides
        )

        # Synthesize models
        config.gemini.models = _merge_models_with_env(config.gemini.models, env_models_overrides)

        return config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e!s}")
        sys.exit(1)
