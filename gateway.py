"""
AI Gateway - Multi-Provider Fallback System (FIXED v2.0)
=========================================================
FIXES APPLIED:
1. ChatRequest now accepts 'messages' array from oura_server.py
2. ChatRequest accepts 'system_prompt' override field
3. Stateless sessions use passed messages array (not just user_message)
4. Non-stateless sessions still use gateway's own session history
5. All other logic unchanged
"""

import os
import time
import uuid
import json
import asyncio
import logging
import aiohttp
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gateway.log")],
)
log = logging.getLogger("ai_gateway")

KEYS = {
    "groq":        os.getenv("GROQ_API_KEY", ""),
    "cerebras":    os.getenv("CEREBRAS_API_KEY", ""),
    "gemini":      os.getenv("GEMINI_API_KEY", ""),
    "mistral":     os.getenv("MISTRAL_API_KEY", ""),
    "cloudflare":  os.getenv("CLOUDFLARE_API_KEY", ""),
    "nvidia":      os.getenv("NVIDIA_API_KEY", ""),
    "github":      os.getenv("GITHUB_API_KEY", ""),
    "openrouter":  os.getenv("OPENROUTER_API_KEY", ""),
    "kimi":        os.getenv("KIMI_API_KEY", ""),
}
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    auth_header: str
    auth_prefix: str
    daily_limit: int
    speed_rank: int
    expires_days: Optional[int] = None


PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        "groq",
        "https://api.groq.com/openai/v1/chat/completions",
        "llama-3.1-70b-versatile",
        "Authorization", "Bearer", 14400, 1
    ),
    ProviderConfig(
        "cerebras",
        "https://api.cerebras.ai/v1/chat/completions",
        "llama3.1-70b",
        "Authorization", "Bearer", 14400, 2, 30
    ),
    ProviderConfig(
        "mistral",
        "https://api.mistral.ai/v1/chat/completions",
        "mistral-large-latest",
        "Authorization", "Bearer", 99999, 3
    ),
    ProviderConfig(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "gemini-2.0-flash-exp",
        "Authorization", "Bearer", 1500, 4, 90
    ),
    ProviderConfig(
        "nvidia",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "nvidia/llama-3.1-70b-instruct",
        "Authorization", "Bearer", 9999, 5, 180
    ),
    ProviderConfig(
        "github",
        "https://models.inference.ai.azure.com/chat/completions",
        "gpt-4o",
        "Authorization", "Bearer", 150, 6
    ),
    ProviderConfig(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "meta-llama/llama-3.3-70b-instruct:free",
        "Authorization", "Bearer", 50, 7
    ),
    ProviderConfig(
        "kimi",
        "https://api.moonshot.cn/v1/chat/completions",
        "moonshot-v1-8k",
        "Authorization", "Bearer", 9999, 8
    ),
    ProviderConfig(
        "cloudflare",
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}"
        f"/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "Authorization", "Bearer", 10000, 9
    ),
]


@dataclass
class ProviderState:
    config: ProviderConfig
    alive: bool = True
    expired: bool = False
    requests_today: int = 0
    last_used: float = 0.0
    last_health_check: float = 0.0
    cooldown_until: float = 0.0
    error_streak: int = 0
    day_reset: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d")
    )

    def reset_daily_if_needed(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.day_reset != today:
            self.requests_today = 0
            self.day_reset = today

    @property
    def available(self) -> bool:
        self.reset_daily_if_needed()
        return (
            not self.expired
            and self.alive
            and time.time() >= self.cooldown_until
            and self.requests_today < self.config.daily_limit
        )


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "id": session_id,
                "history": [],
                "created_at": time.time(),
                "last_active": time.time(),
                "request_count": 0,
                "user_id": session_id,
            }
        return self._sessions[session_id]

    def append_message(self, session_id: str, role: str, content: str):
        # FIX: never accumulate history for stateless sessions —
        # oura_server.py owns all state for stateless sessions
        if session_id.startswith("stateless-"):
            return
        session = self.get_or_create(session_id)
        session["history"].append({"role": role, "content": content})
        session["last_active"] = time.time()
        if role == "user":
            session["request_count"] += 1

    def get_history(self, session_id: str) -> list:
        # FIX: stateless sessions always return empty — context
        # comes from oura_server via the messages array
        if session_id.startswith("stateless-"):
            return []
        return self.get_or_create(session_id).get("history", [])

    def prune_history(self, session_id: str, max_messages: int = 40):
        if session_id.startswith("stateless-"):
            return
        session = self.get_or_create(session_id)
        if len(session["history"]) > max_messages:
            history = session["history"]
            system_msgs = [m for m in history if m["role"] == "system"]
            rest = [m for m in history if m["role"] != "system"]
            session["history"] = system_msgs + rest[-max_messages:]

    def all_sessions(self) -> list:
        return list(self._sessions.values())


class HealthChecker:
    HEALTH_INTERVAL = 30
    FAST_RETRY_INTERVAL = 2

    def __init__(self, states, session):
        self._states = states
        self._http = session
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._health_loop())

    async def _health_loop(self):
        while self._running:
            await self._check_all()
            if not any(s.available for s in self._states.values()):
                await asyncio.sleep(self.FAST_RETRY_INTERVAL)
            else:
                await asyncio.sleep(self.HEALTH_INTERVAL)

    async def _check_all(self):
        await asyncio.gather(
            *[self._check_one(s) for s in self._states.values()],
            return_exceptions=True,
        )

    async def _check_one(self, state):
        name = state.config.name
        if state.expired:
            return
        try:
            alive = await self._ping(state)
            prev = state.alive
            state.alive = alive
            state.last_health_check = time.time()
            if alive:
                state.error_streak = 0
            if alive and not prev:
                log.info(f"[{name}] ✅ Back alive.")
            elif not alive and prev:
                log.warning(f"[{name}] ❌ Went down.")
        except Exception:
            state.alive = False

    async def _ping(self, state) -> bool:
        name = state.config.name
        key = KEYS.get(name, "")
        if not key:
            state.expired = True
            return False
        headers = {
            state.config.auth_header: f"{state.config.auth_prefix} {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": state.config.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        if name == "cloudflare":
            payload = {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
        try:
            async with self._http.post(
                state.config.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 401:
                    state.expired = True
                    return False
                if resp.status == 429:
                    state.cooldown_until = time.time() + 60
                    return False
                return resp.status < 500
        except Exception:
            return False

    def force_refresh(self):
        asyncio.create_task(self._check_all())


class RateLimitError(Exception):
    pass


class ExpiredKeyError(Exception):
    pass


class ProviderError(Exception):
    pass


def _validate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and sanitize a messages array.
    - Each entry must have 'role' and 'content' keys.
    - Role must be one of: system, user, assistant.
    - Content must be a non-empty string.
    - Strips any unknown keys.
    - Returns a cleaned list, discarding malformed entries with a warning.
    """
    valid_roles = {"system", "user", "assistant"}
    cleaned = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            log.warning(f"[messages] Entry {i} is not a dict, skipping: {msg!r}")
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in valid_roles:
            log.warning(
                f"[messages] Entry {i} has invalid role {role!r}, skipping."
            )
            continue
        if not isinstance(content, str) or not content.strip():
            log.warning(
                f"[messages] Entry {i} has empty/non-string content, skipping."
            )
            continue
        cleaned.append({"role": role, "content": content.strip()})
    return cleaned


def _ensure_system_message(
    messages: List[Dict[str, Any]],
    default_system_prompt: str,
) -> List[Dict[str, Any]]:
    """
    Ensure the messages array starts with exactly one system message.
    - If no system message exists, prepend default_system_prompt.
    - If multiple system messages exist, keep only the first.
    - Moves system message to position 0 if it is not already there.
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if not system_msgs:
        # No system message — prepend the default
        return [{"role": "system", "content": default_system_prompt}] + non_system

    # Keep only the first system message, discard duplicates
    if len(system_msgs) > 1:
        log.warning(
            f"[messages] {len(system_msgs)} system messages found; "
            f"keeping only the first."
        )
    return [system_msgs[0]] + non_system


class AIGateway:
    # Default system prompt used when no system message is provided
    # by the caller and no messages array is passed.
    SYSTEM_PROMPT = (
        "You are a helpful, honest, and knowledgeable AI assistant. "
        "Answer clearly and concisely. Maintain context from the conversation."
    )

    def __init__(self):
        self._http = None
        self._states = {}
        self._sessions = SessionStore()
        self._health_checker = None

    async def start(self):
        self._http = aiohttp.ClientSession()
        for p in sorted(PROVIDERS, key=lambda x: x.speed_rank):
            self._states[p.name] = ProviderState(config=p)
        self._health_checker = HealthChecker(self._states, self._http)
        await self._health_checker.start()
        await asyncio.sleep(2)
        log.info("AI Gateway ready.")

    async def stop(self):
        if self._health_checker:
            self._health_checker._running = False
        if self._http:
            await self._http.close()

    def _get_sorted_alive_providers(self):
        return sorted(
            [s for s in self._states.values() if s.available],
            key=lambda s: (
                s.requests_today / max(s.config.daily_limit, 1),
                s.config.speed_rank,
            ),
        )

    async def chat(
        self,
        session_id: str,
        user_message: str,
        # ── FIX #1: Accept full conversation history from oura_server ──
        messages: Optional[List[Dict[str, Any]]] = None,
        # ── FIX #2: Accept system_prompt override from oura_server ──
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Main chat entry point.

        Priority for building the messages array sent to the AI:

        STATELESS sessions (session_id starts with "stateless-"):
            1. If caller passed a non-empty 'messages' array → use it directly
               (oura_server has already built the full context with history,
               memory recalls, and system prompt — trust it completely).
            2. Otherwise fall back to a minimal [system, user] pair using
               system_prompt (if given) or self.SYSTEM_PROMPT.

        NON-STATELESS sessions:
            1. If caller passed a non-empty 'messages' array → use it directly
               (caller owns the context; gateway does NOT inject its own history).
            2. Otherwise use gateway's own accumulated session history
               (legacy mode — caller did not pass messages).

        In all cases:
            - Append user_message to gateway session only for non-stateless sessions
              so gateway history stays consistent for legacy callers.
            - The messages array is validated and sanitized before sending to AI.
            - Exactly one system message is guaranteed at position 0.
        """

        is_stateless = session_id.startswith("stateless-")

        # ── Record user turn in gateway session (non-stateless only) ──
        # For stateless, oura_server owns all state; gateway stores nothing.
        self._sessions.append_message(session_id, "user", user_message)
        self._sessions.prune_history(session_id)

        # ── Build the messages array to send to the AI provider ──
        final_messages = self._build_messages(
            session_id=session_id,
            user_message=user_message,
            caller_messages=messages,
            system_prompt=system_prompt,
            is_stateless=is_stateless,
        )

        log.info(
            f"[{session_id[:12]}] {'stateless' if is_stateless else 'stateful'} | "
            f"messages={len(final_messages)} | "
            f"source={'caller' if messages else 'gateway-history'}"
        )

        # ── Select providers ──
        providers = self._get_sorted_alive_providers()
        if not providers:
            self._health_checker.force_refresh()
            await asyncio.sleep(2)
            providers = self._get_sorted_alive_providers()
            if not providers:
                return {
                    "text": (
                        "⚠️ All AI providers temporarily unavailable. "
                        "Please retry in a moment."
                    ),
                    "provider": None,
                    "session_id": session_id,
                    "error": True,
                }

        # ── Try providers in priority order ──
        last_error = None
        for state in providers:
            name = state.config.name
            try:
                result = await self._call_provider(state, final_messages)
                if result:
                    state.requests_today += 1
                    state.last_used = time.time()
                    state.error_streak = 0
                    # Record assistant reply in gateway session (non-stateless only)
                    self._sessions.append_message(session_id, "assistant", result)
                    log.info(
                        f"[{session_id[:12]}] ✅ {name} "
                        f"({len(result)} chars)"
                    )
                    return {
                        "text": result,
                        "provider": name,
                        "session_id": session_id,
                        "error": False,
                    }
            except RateLimitError:
                state.cooldown_until = time.time() + 60
                state.error_streak += 1
                log.warning(f"[{name}] Rate limited, cooling down 60s.")
            except ExpiredKeyError:
                state.expired = True
                log.error(f"[{name}] Key expired.")
            except ProviderError as e:
                last_error = str(e)
                state.error_streak += 1
                if state.error_streak >= 3:
                    state.alive = False
                    state.cooldown_until = time.time() + 120
                    log.error(
                        f"[{name}] 3 consecutive errors, cooling down 120s."
                    )
            except Exception as e:
                last_error = str(e)
                log.error(f"[{name}] Unexpected error:\n{traceback.format_exc()}")

        self._health_checker.force_refresh()
        return {
            "text": (
                f"⚠️ All providers unavailable. "
                f"Last error: {last_error}. Please retry."
            ),
            "provider": None,
            "session_id": session_id,
            "error": True,
        }

    def _build_messages(
        self,
        session_id: str,
        user_message: str,
        caller_messages: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
        is_stateless: bool,
    ) -> List[Dict[str, Any]]:
        """
        Build the final messages array to send to the AI provider.

        Decision tree:
        ┌─ caller passed non-empty messages array?
        │   YES → validate it, ensure exactly one system message, use it.
        │          (Trust oura_server completely — it built the full context.)
        │
        │   NO  → is_stateless?
        │           YES → build minimal [system + user] pair.
        │                 Use system_prompt override if given, else default.
        │
        │           NO  → use gateway's own accumulated session history.
        │                 Prepend system message if missing.
        └─────────────────────────────────────────────────────────────────
        """
        effective_system = system_prompt or self.SYSTEM_PROMPT

        # ── Path 1: Caller supplied a full messages array ──
        if caller_messages and len(caller_messages) > 0:
            log.debug(
                f"[{session_id[:12]}] Using caller-supplied messages array "
                f"({len(caller_messages)} entries)."
            )
            validated = _validate_messages(caller_messages)
            if not validated:
                log.warning(
                    f"[{session_id[:12]}] Caller messages array was empty after "
                    f"validation; falling back to minimal pair."
                )
                return [
                    {"role": "system", "content": effective_system},
                    {"role": "user",   "content": user_message},
                ]
            return _ensure_system_message(validated, effective_system)

        # ── Path 2: No messages array — stateless minimal pair ──
        if is_stateless:
            log.debug(
                f"[{session_id[:12]}] Stateless session, no messages array; "
                f"building minimal [system, user] pair."
            )
            return [
                {"role": "system", "content": effective_system},
                {"role": "user",   "content": user_message},
            ]

        # ── Path 3: No messages array — stateful gateway history ──
        log.debug(
            f"[{session_id[:12]}] Stateful session, no messages array; "
            f"using gateway history."
        )
        history = self._sessions.get_history(session_id)
        if not history:
            return [
                {"role": "system", "content": effective_system},
                {"role": "user",   "content": user_message},
            ]
        return _ensure_system_message(history, effective_system)

    async def _call_provider(self, state, messages):
        if state.config.name == "cloudflare":
            return await self._call_cloudflare(state, messages)
        return await self._call_openai_compat(state, messages)

    async def _call_openai_compat(self, state, messages):
        name = state.config.name
        key = KEYS[name]
        if not key:
            raise ExpiredKeyError(f"{name} key missing")
        headers = {
            state.config.auth_header: f"{state.config.auth_prefix} {key}",
            "Content-Type": "application/json",
        }
        if name == "openrouter":
            headers["HTTP-Referer"] = "https://ai-gateway.local"
            headers["X-Title"] = "AI Gateway"
        payload = {
            "model": state.config.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        try:
            async with self._http.post(
                state.config.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.json()
                if resp.status == 401:
                    raise ExpiredKeyError(f"{name} expired")
                if resp.status == 429:
                    raise RateLimitError(f"{name} rate limited")
                if resp.status >= 500:
                    raise ProviderError(f"{name} server error {resp.status}")
                if resp.status >= 400:
                    raise ProviderError(
                        f"{name} error {resp.status}: "
                        f"{body.get('error', {}).get('message', str(body))}"
                    )
                choices = body.get("choices", [])
                if not choices:
                    raise ProviderError(f"{name} no choices in response")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ProviderError(f"{name} empty content in response")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError):
            raise
        except asyncio.TimeoutError:
            raise ProviderError(f"{name} timeout after 30s")
        except aiohttp.ClientError as e:
            raise ProviderError(f"{name} network error: {e}")

    async def _call_cloudflare(self, state, messages):
        key = KEYS["cloudflare"]
        if not key:
            raise ExpiredKeyError("cloudflare key missing")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {"messages": messages, "max_tokens": 2048}
        try:
            async with self._http.post(
                state.config.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.json()
                if resp.status == 401:
                    raise ExpiredKeyError("cloudflare expired")
                if resp.status == 429:
                    raise RateLimitError("cloudflare rate limited")
                if resp.status >= 400:
                    raise ProviderError(
                        f"cloudflare {resp.status}: {body}"
                    )
                content = body.get("result", {}).get("response", "")
                if not content:
                    raise ProviderError("cloudflare empty response")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError):
            raise
        except asyncio.TimeoutError:
            raise ProviderError("cloudflare timeout after 30s")
        except aiohttp.ClientError as e:
            raise ProviderError(f"cloudflare network error: {e}")

    def get_status(self) -> dict:
        providers = {}
        for name, state in self._states.items():
            providers[name] = {
                "alive": state.alive,
                "available": state.available,
                "expired": state.expired,
                "requests_today": state.requests_today,
                "daily_limit": state.config.daily_limit,
                "utilization_pct": (
                    round(
                        state.requests_today / state.config.daily_limit * 100, 1
                    )
                    if state.config.daily_limit > 0
                    else 0
                ),
                "cooldown_until": (
                    datetime.fromtimestamp(state.cooldown_until).isoformat()
                    if state.cooldown_until > time.time()
                    else None
                ),
                "error_streak": state.error_streak,
                "speed_rank": state.config.speed_rank,
                "model": state.config.model,
            }
        return {
            "providers": providers,
            "active_sessions": len(self._sessions.all_sessions()),
            "alive_providers": sum(
                1 for s in self._states.values() if s.available
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions.get_or_create(sid)
        return sid

    def clear_session(self, session_id: str):
        if session_id in self._sessions._sessions:
            del self._sessions._sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:
    app = FastAPI(title="AI Gateway", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    gateway = AIGateway()

    @app.on_event("startup")
    async def startup():
        await gateway.start()

    @app.on_event("shutdown")
    async def shutdown():
        await gateway.stop()

    class ChatRequest(BaseModel):
        # ── Required: the raw user message ──
        message: str

        # ── Optional: session identifier ──
        session_id: Optional[str] = None

        # ── FIX: Accept full conversation history from oura_server ──
        # When oura_server passes this, gateway uses it INSTEAD of its
        # own session history. This is the primary fix for the echo loop.
        messages: Optional[List[Dict[str, Any]]] = None

        # ── FIX: Accept system prompt override from oura_server ──
        # Allows oura_server to inject Oura-specific system prompt
        # without gateway overwriting it with its own default.
        system_prompt: Optional[str] = None

        @validator("message")
        def message_not_empty(cls, v):
            if not v or not v.strip():
                raise ValueError("message cannot be empty")
            return v.strip()

        @validator("messages", pre=True, always=True)
        def validate_messages_field(cls, v):
            # Accept None or a list; reject anything else
            if v is None:
                return None
            if not isinstance(v, list):
                raise ValueError("messages must be a list or null")
            return v

    @app.post("/chat")
    async def chat_endpoint(req: ChatRequest):
        sid = req.session_id or gateway.new_session()

        result = await gateway.chat(
            session_id=sid,
            user_message=req.message,
            # ── Pass through the messages array from oura_server ──
            messages=req.messages,
            # ── Pass through the system_prompt override ──
            system_prompt=req.system_prompt,
        )
        return result

    @app.post("/session/new")
    async def new_session():
        return {"session_id": gateway.new_session()}

    @app.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        gateway.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/status")
    async def status():
        return gateway.get_status()

    @app.get("/health")
    async def health():
        st = gateway.get_status()
        if st["alive_providers"] == 0:
            raise HTTPException(
                status_code=503,
                detail="No providers available",
            )
        return {
            "status": "ok",
            "alive_providers": st["alive_providers"],
            "timestamp": datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
async def cli_demo():
    print("\n🤖 AI Gateway v2.0 — CLI Demo")
    print("Type 'quit' to exit.\n")
    gw = AIGateway()
    await gw.start()
    sid = gw.new_session()
    print(f"Session: {sid}\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.lower() == "quit":
            break
        result = await gw.chat(sid, msg)
        provider = result.get("provider", "unknown")
        print(f"\nAI [{provider}]: {result['text']}\n")
    await gw.stop()


if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        if not HAS_FASTAPI:
            print(
                "FastAPI not installed. "
                "Run: pip install fastapi uvicorn python-dotenv"
            )
            sys.exit(1)
        import uvicorn
        uvicorn.run(
            "gateway:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )
    else:
        asyncio.run(cli_demo())
