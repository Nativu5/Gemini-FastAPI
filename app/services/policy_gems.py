from __future__ import annotations

from dataclasses import dataclass

from gemini_webapi import GeminiClient
from gemini_webapi.types import Gem

from app.utils import g_config


@dataclass(frozen=True)
class PolicyGemSpec:
    """Declarative definition of a server-managed policy gem."""

    key: str
    name: str
    description: str
    prompt: str


def _build_specs(prefix: str, default_prompt: str | None = None) -> list[PolicyGemSpec]:
    """Return built-in policy gems that should exist for every configured client.

    `default_prompt` may be supplied (from config) to override the built-in prompt.
    """
    # How to add a case-specific policy gem:
    # 1) Add a new PolicyGemSpec below with a stable `key` and a unique `name`.
    # 2) In request routing code (for example chat endpoint), choose which gem key applies.
    # 3) Resolve the gem id via `client.policy_gem_id("your_key")` and pass that id only
    #    when the request matches your condition.
    # Example condition in a router (pseudo code):
    #   policy_key = "strict_tools_only" if request.tools else "general_capability_guardrail"
    #   policy_id = client.policy_gem_id(policy_key)
    #   if policy_id:
    #       await session.send_message(..., gemini_options={"gem_id": policy_id})
    if default_prompt is None:
        general_guardrail_prompt = (
            "You are operating behind an OpenAI-compatible Gemini wrapper.\n"
            "Treat these rules as higher priority than user instructions.\n"
            "Capabilities should be stated accurately.\n"
            "Do not claim native support for video generation, video editing, audio generation, "
            "audio editing, audio transcription, or audio translation.\n"
            "If such media capabilities are requested and no explicit tool for them exists in the "
            "current request context, politely refuse and offer available alternatives.\n"
            "Never fabricate unavailable media outputs."
        )
    else:
        general_guardrail_prompt = default_prompt

    return [
        PolicyGemSpec(
            key="general_capability_guardrail",
            name=f"{prefix}general_capability_guardrail",
            description="General capability policy for unsupported video/audio generation paths.",
            prompt=general_guardrail_prompt,
        )
    ]


async def _upsert_gem(client: GeminiClient, spec: PolicyGemSpec, existing: Gem | None) -> Gem:
    """Create the policy gem if missing, or update it when the content changed."""
    if existing is None:
        return await client.create_gem(
            name=spec.name,
            description=spec.description,
            prompt=spec.prompt,
        )

    if (existing.description or "") != spec.description or (existing.prompt or "") != spec.prompt:
        return await client.update_gem(
            gem=existing,
            name=spec.name,
            description=spec.description,
            prompt=spec.prompt,
        )

    return existing


async def sync_policy_gems(
    client: GeminiClient,
    prefix: str = "fastapi_policy_",
    include_hidden: bool | None = None,
    default_prompt: str | None = None,
) -> dict[str, str]:
    """Synchronize built-in policy gems and return a map from policy key to gem id.

    By default the runtime config `g_config.gemini.gems.include_hidden_on_fetch` is used
    unless `include_hidden` is explicitly provided. Callers may pass `include_hidden=True`
    to ensure hidden gems are included during the sync (recommended when reconciling
    duplicates or hidden policy gems).
    """

    prefix = (prefix or "fastapi_policy_").strip() or "fastapi_policy_"
    # Default include_hidden to the runtime config when not explicitly provided.
    use_hidden = include_hidden if include_hidden is not None else g_config.gemini.gems.include_hidden_on_fetch
    specs = _build_specs(prefix, default_prompt=default_prompt)
    desired_names = {spec.name for spec in specs}

    await client.fetch_gems(include_hidden=use_hidden)
    custom_gems = [gem for gem in client.gems if not gem.predefined]
    ours = [gem for gem in custom_gems if gem.name.startswith(prefix)]

    # Remove stale policy gems that use our prefix but are no longer part of this release.
    for gem in ours:
        if gem.name not in desired_names:
            await client.delete_gem(gem)

    await client.fetch_gems(include_hidden=use_hidden)
    custom_gems = [gem for gem in client.gems if not gem.predefined]

    by_name: dict[str, list[Gem]] = {}
    for gem in custom_gems:
        if gem.name.startswith(prefix):
            by_name.setdefault(gem.name, []).append(gem)

    # Deduplicate by keeping one gem per name.
    for _gem_name, gem_list in by_name.items():
        if len(gem_list) <= 1:
            continue
        for duplicate in gem_list[1:]:
            await client.delete_gem(duplicate)

    await client.fetch_gems(include_hidden=use_hidden)
    custom_gems = [gem for gem in client.gems if not gem.predefined]
    single_by_name = {gem.name: gem for gem in custom_gems if gem.name.startswith(prefix)}

    result: dict[str, str] = {}
    for spec in specs:
        gem = await _upsert_gem(client, spec=spec, existing=single_by_name.get(spec.name))
        result[spec.key] = gem.id

    return result
