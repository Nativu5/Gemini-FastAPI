# Configuration

[ English | [中文](CONFIG.zh.md) ]

Every option, with its default, is listed in [`config/config.yaml`](config/config.yaml). This page covers what its comments cannot.

## Config file and environment variables

The server reads `config/config.yaml`, or the file named by `CONFIG_PATH`. An environment variable named `CONFIG_` plus the path to a key, with `__` between levels, overrides that key:

```bash
export CONFIG_SERVER__API_KEY="your-secure-api-key"
export CONFIG_GEMINI__CLIENTS__0__COOKIES='your-cookie-header'
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"
export CONFIG_GEMINI__CHAT_MODE="auto_delete"
```

- `CONFIG_GEMINI__CLIENTS__<n>__<key>` changes account `n` of the file, or adds an account when `n` is the next free index. An index past that stops the server.
- `GEMINI_COOKIE_PATH` sets the directory where the cookies the server refreshes are cached.
- A value the server does not accept stops it at startup, and the log names the key.

## Accounts

Each entry under `gemini.clients` is one Google account.

- `id`: stored with every conversation the account opens. Keep it stable when you update cookies, or stored conversations can no longer be continued.
- `cookies`: the Cookie header of a signed-in request to Gemini ([how to get it](README.md#getting-your-cookies)). Leave it out to run the client as a guest.
- `proxy`: an optional proxy URL for this account only.
- `impersonate`: an optional browser fingerprint for [`curl_cffi`](https://github.com/lexiforest/curl_cffi), such as `chrome`.

## Chat modes

`gemini.chat_mode` decides what happens to the chats the server opens in your Google account.

| Mode | Kept in the Google account | Context across turns |
| --- | --- | --- |
| `normal` (default) | Yes, until you delete it | Reliable |
| `temporary` | Never saved | Best effort |
| `auto_delete` | Yes, then deleted once idle | Reliable |

**`temporary`** uses Google's temporary chats. Google keeps one temporary chat open per account and may close it at any time without an error, so the reply can come back without the earlier context. The server continues only the most recent temporary chat and replays the full history into a new one otherwise. The input limit is also lower. Prefer another mode for long conversations.

**`auto_delete`** uses normal chats and removes them from both the Google account and the local store after `gemini.auto_delete_ttl` seconds without a request (default `86400`, minimum `60`).

- Only chats the server opened in this mode are deleted. Chats from other modes, and everything else in the account, are left alone.
- Deletion runs on a periodic pass, every half TTL (between one minute and six hours), so a chat can outlive its TTL by up to one interval.
- A deletion that fails, for example while the account is signed out, is retried on later passes and abandoned with an error in the log after five attempts.
- Generated media expires on the same TTL.

After `chat_mode` changes, stored chats from the other side of the `auto_delete` boundary are not continued. Their history is replayed into a new chat, and chats opened under `auto_delete` are still deleted on schedule.

## Guests and rejected cookies

A client configured without `cookies` is a guest. A guest can answer text prompts only: it cannot upload files, cannot choose a model, and keeps no history.

- Clients with an account are preferred. A guest receives requests only when none of them is available, and a request that needs an upload fails with `503` if only guests remain.
- A client configured with cookies never runs as a guest. When Google no longer accepts its cookies, at startup or later, because they expired or could not be refreshed, the error is logged and the client stops serving. A request it was handling gets `503`. Each later request that would use it first tries to restart it; while Google rejects its cookies, the request goes to another client, or gets `503` if none is left. New cookies take effect after a restart of the server.
- At startup, the server stops if no client starts.
- `/health` reports a client as unhealthy when it is not running, or when it was configured with cookies and Google no longer accepts them. A guest is healthy while it runs. `gemini.guest_mode` sets when unhealthy clients turn into HTTP 503:

| `guest_mode` | `/health` returns 503 when |
| --- | --- |
| `strict` | any client is unhealthy |
| `adaptive` (default) | every client is unhealthy |
| `permissive` | never because of clients |

A storage failure always returns 503.

## Models

Models are discovered from each account at startup, so there is nothing to configure. Which models you get, and what they are called, depends on the account's plan.

- A request can name a model by its listed name or its display name.
- Matching is loose. Version numbers are ignored and one word of a listed name is enough, so `gemini-2.5-pro` and `pro` both resolve to `gemini-pro`.
- A name that matches nothing is rejected with `400`. The server never falls back to another model.

## Gems

> [!WARNING]
> Google has announced the [retirement of Gems](https://support.google.com/gemini/answer/18560919): for personal accounts they became read-only on 2026-11-17 and are removed on 2026-12-17; Workspace accounts follow in March 2027 and education accounts in June 2027. The server no longer creates gems, and this feature only uses the ones an earlier version created while the account still has them. Once it does not, set the policy back to `off`: a gem request that Google rejects is retried for up to about 15 seconds, restarting the client each time, before the prompt is sent inline. That happens on every request.

Off by default. With `gemini.gems.policy: fetch_only`, the system prompt at the start of a `/v1/chat/completions` or `/v1/responses` request is referenced from the gem (Gemini's saved custom instructions) that an earlier version of the server stored for it in the Google account, named `fastapi_policy_...`, instead of being sent inside the message. A prompt without such a gem, or whose gem cannot be used, is sent inline.

- Gems stay in the account until you delete them. With `cleanup.enabled: true` as well, gems unused for `cleanup.unused_days` (default `7`) are deleted by a cleanup that runs at startup and every 6 hours after that, and each gem's description is updated to record when it was last used.
- Guest sessions and Gemini-native requests always send the prompt inline.

## Input size

Input longer than 90% of `gemini.max_chars_per_request` (81% in `temporary` mode) is sent to Gemini as a `message.txt` attachment. `server.max_request_body_bytes` is a separate local guard on the HTTP body and says nothing about what Gemini accepts.
