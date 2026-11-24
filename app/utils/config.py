import os
import sys
from enum import StrEnum
from typing import Any, Literal, cast

from curl_cffi import BrowserTypeLiteral
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_PATH = "config/config.yaml"


class HTTPSConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable HTTPS")
    key_file: str = Field(default="certs/privkey.pem", description="SSL private key file path")
    cert_file: str = Field(default="certs/fullchain.pem", description="SSL certificate file path")


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    api_key: str | None = Field(
        default=None,
        description="API key for authentication, if set, will enable API key validation",
    )
    max_request_body_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=0,
        description=(
            "Local HTTP body safety ceiling in bytes (0 disables it). This protects wrapper "
            "resources and does not define Gemini Web's upstream acceptance limit"
        ),
    )
    schema_validation_budget_seconds: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Wall-clock budget for evaluating the regex keywords of a client-supplied JSON "
            "Schema against one response. Guards against catastrophic backtracking; exhausting "
            "it leaves the reply unverified rather than treating it as a schema violation"
        ),
    )
    https: HTTPSConfig = Field(default=HTTPSConfig(), description="HTTPS configuration")


class GeminiClientSettings(BaseModel):
    """Credential set for one Gemini client."""

    # A key it does not know is most likely one of the removed `secure_1psid` and
    # `secure_1psidts`. Ignored, it would leave the client without cookies: a guest.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier for the client")
    cookies: str | None = Field(
        default=None,
        description=(
            "Cookie header of a request to gemini.google.com from a signed-in browser. "
            "Without it the client runs as a guest"
        ),
    )
    proxy: str | None = Field(default=None, description="Proxy URL for this Gemini client")
    impersonate: BrowserTypeLiteral | None = Field(
        default=None,
        description="Browser impersonation target for curl_cffi. None uses library default",
    )

    @field_validator("cookies")
    @classmethod
    def _signed_in_cookies(cls, value: str | None) -> str | None:
        # Without it no one is signed in, and startup would fail with Google's refusal instead of
        # this. A header copied with its name, "Cookie: ...", lands here too.
        names = {pair.partition("=")[0].strip() for pair in (value or "").split(";")}
        if value is not None and "__Secure-1PSID" not in names:
            raise ValueError("must be the value of a Cookie header that includes __Secure-1PSID")
        return value

    @field_validator("proxy", "impersonate", mode="before")
    @classmethod
    def _blank_string_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ChatMode(StrEnum):
    NORMAL = "normal"
    TEMPORARY = "temporary"
    AUTO_DELETE = "auto_delete"


class GuestMode(StrEnum):
    """Health-check policy for unhealthy Gemini clients."""

    STRICT = "strict"
    ADAPTIVE = "adaptive"
    PERMISSIVE = "permissive"


class GeminiGemPolicyMode(StrEnum):
    """Whether a leading system prompt is sent as a gem the server made earlier, when one exists."""

    OFF = "off"
    FETCH_ONLY = "fetch_only"


class GeminiGemPoliciesConfig(BaseModel):
    """How the gems managed by the server are named."""

    prefix: str = Field(
        default="fastapi_policy_",
        description="Name prefix used to identify gems made by this server",
    )


class GeminiGemCleanupConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable deletion of managed gems that have not been used recently",
    )
    unused_days: int = Field(
        default=7,
        ge=1,
        description="Delete managed gems that were not used for this many days",
    )
    touch_interval_minutes: int = Field(
        default=60,
        ge=1,
        description="Minimum minutes between usage-touch metadata updates for the same gem",
    )
    dry_run: bool = Field(
        default=False,
        description="Log cleanup candidates without deleting them",
    )
    max_deletes_per_run: int = Field(
        default=5,
        ge=1,
        description="Maximum managed-gem deletions per account in one cleanup",
    )


class GeminiGemsConfig(BaseModel):
    policy: GeminiGemPolicyMode = Field(
        default=GeminiGemPolicyMode.OFF,
        description=(
            "'off' sends system prompts inline; 'fetch_only' uses a gem that already exists "
            "for a system prompt. Google no longer lets personal accounts create gems"
        ),
    )
    policies: GeminiGemPoliciesConfig = Field(
        default=GeminiGemPoliciesConfig(),
        description="Naming of managed gems",
    )
    cleanup: GeminiGemCleanupConfig = Field(
        default=GeminiGemCleanupConfig(),
        description="Cleanup policy for managed gems",
    )


class GeminiConfig(BaseModel):
    clients: list[GeminiClientSettings] = Field(
        ..., description="List of Gemini client credential pairs"
    )
    timeout: int = Field(default=450, ge=30, description="Request timeout in seconds")
    watchdog_timeout: int = Field(default=120, ge=30, description="Watchdog timeout in seconds")
    auto_refresh: bool = Field(True, description="Enable auto-refresh for Gemini sessions")
    refresh_interval: int = Field(
        default=600,
        ge=60,
        description="Interval in seconds to refresh Gemini sessions (Not less than 60s)",
    )
    auto_close: bool = Field(
        default=False, description="Enable auto-close for Gemini sessions after inactivity"
    )
    close_delay: int = Field(
        default=900, ge=30, description="Inactivity delay in seconds before auto-closing"
    )
    verbose: bool = Field(True, description="Enable verbose logging for Gemini API requests")
    extended_thinking: bool = Field(
        default=False,
        description="Enable Gemini extended thinking mode for message generation",
    )
    max_chars_per_request: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum characters Gemini Web can accept per request",
    )
    chat_mode: ChatMode = Field(
        default=ChatMode.NORMAL,
        description=(
            "Chat mode: 'normal' uses standard chats (saved to account history); "
            "'temporary' sends with Google's temporary mode (not saved to account, tighter input limit); "
            "'auto_delete' uses standard chats for full reliability and context, then automatically "
            "deletes them from both Google and local storage once expired. "
            "Warning for 'temporary': Google may close a temporary window at any time mid-conversation, "
            "and the reply can then come back without the earlier context instead of erroring"
        ),
    )
    auto_delete_ttl: int = Field(
        default=86400,
        ge=60,
        description=(
            "Inactivity TTL in seconds applicable ONLY to conversations created with "
            "chat_mode='auto_delete'. When expired, the conversation is deleted from "
            "both Google account history and local LMDB. Does not affect 'normal' conversations "
            "(which are retained locally according to storage.retention_days and never deleted from Google; "
            "default: 86400s / 24h)"
        ),
    )
    guest_mode: GuestMode = Field(
        default=GuestMode.ADAPTIVE,
        description=(
            "Health policy for unhealthy clients: 'strict' fails health checks when any client is "
            "unhealthy; 'adaptive' fails only when all clients are unhealthy; 'permissive' "
            "only logs a warning"
        ),
    )
    allow_private_url_fetch: bool = Field(
        default=False,
        description="Allow server-side fetching of private/loopback URLs (SSRF risk; default blocks them)",
    )
    url_fetch_timeout: int = Field(
        default=15,
        ge=1,
        le=120,
        description="Timeout in seconds for server-side URL fetches",
    )
    gems: GeminiGemsConfig = Field(
        default=GeminiGemsConfig(),
        description="System-prompt gem policy, limits and cleanup",
    )


class CORSConfig(BaseModel):
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
    path: str = Field(
        default="data/lmdb",
        description="Path to the storage directory where data will be saved",
    )
    media_path: str = Field(
        default="data/media",
        description="Path to the directory where generated media will be stored",
    )
    max_size: int = Field(
        default=1024**2 * 256,  # 256 MB
        ge=1,
        description="Maximum size of the storage in bytes",
    )
    retention_days: int = Field(
        default=14,
        ge=0,
        description=(
            "Number of days to retain standard conversations in local LMDB before cleanup "
            "(0 disables cleanup). Only purges local database and media storage; does not delete "
            "conversations from Google accounts (for remote Google deletion, see gemini.auto_delete_ttl)"
        ),
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
        description="Logging level",
    )


class Config(BaseSettings):
    server: ServerConfig = Field(
        default=ServerConfig(),
        description="Server configuration, including host, port, and API key",
    )

    cors: CORSConfig = Field(
        default=CORSConfig(),
        description="CORS configuration, allows cross-origin requests",
    )

    gemini: GeminiConfig = Field(..., description="Gemini API configuration, must be set")

    storage: StorageConfig = Field(
        default=StorageConfig(),
        description="Storage configuration, defines where and how data will be stored",
    )

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


def extract_gemini_clients_env() -> dict[int, dict[str, Any]]:
    """Extract and remove all Gemini clients related environment variables, return a mapping from index to field dict."""
    prefix = "CONFIG_GEMINI__CLIENTS__"
    env_overrides: dict[int, dict[str, Any]] = {}
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

    for k in to_delete:
        del os.environ[k]
    return env_overrides


def _merge_clients_with_env(
    base_clients: list[GeminiClientSettings], env_overrides: dict[int, dict[str, Any]]
) -> list[GeminiClientSettings]:
    """Return Gemini clients with environment overrides applied to the base list."""
    result_clients = list(base_clients)
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
    return result_clients


def initialize_config() -> Config:
    """Load the configuration from the environment and YAML, merging per-client overrides."""
    try:
        env_clients_overrides = extract_gemini_clients_env()
        settings_cls: type[Any] = Config
        config = cast(Config, settings_cls())

        config.gemini.clients = _merge_clients_with_env(
            config.gemini.clients, env_clients_overrides
        )

        return config
    except ValidationError as e:
        # Not `str(e)`: it quotes each rejected value, and here those are cookies.
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )
        logger.error(f"Configuration validation failed: {problems}")
        sys.exit(1)
