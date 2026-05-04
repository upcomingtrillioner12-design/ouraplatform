"""
AI Gateway - Multi-Provider Fallback System
============================================
Supports: Groq, Cerebras, Mistral, Google Gemini, NVIDIA, GitHub (Azure),
          OpenRouter, Kimi (Moonshot), Cloudflare Workers AI

Features:
- Per-user persistent sessions (in-memory + optional Redis)
- Parallel health checks every 30 seconds
- Auto failover < 100ms
- 401 expiry detection + provider auto-removal
- Full conversation history preserved across failovers
- Logs all provider switches
"""

import os
import time
import uuid
import json
import asyncio
import logging
import aiohttp
import traceback
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gateway.log"),
    ],
)
log = logging.getLogger("ai_gateway")

# ─── API Keys from Environment Variables (NOT hardcoded) ────────────────────────
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

# ─── Provider Configs ─────────────────────────────────────────────────────────
@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    auth_header: str
    auth_prefix: str
    daily_limit: int
    speed_rank: int           # Lower = faster (priority)
    expires_days: Optional[int] = None   # None = never expires

PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1/chat/completions",
        model="llama-3.1-70b-versatile",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=14400,
        speed_rank=1,
        expires_days=None,
    ),
    ProviderConfig(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1/chat/completions",
        model="llama3.1-70b",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=14400,
        speed_rank=2,
        expires_days=30,
    ),
    ProviderConfig(
        name="mistral",
        base_url="https://api.mistral.ai/v1/chat/completions",
        model="mistral-large-latest",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=99999,
        speed_rank=3,
        expires_days=None,
    ),
    ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        model="gemini-2.0-flash-exp",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=1500,
        speed_rank=4,
        expires_days=90,
    ),
    ProviderConfig(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model="nvidia/llama-3.1-70b-instruct",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=9999,
        speed_rank=5,
        expires_days=180,
    ),
    ProviderConfig(
        name="github",
        base_url="https://models.inference.ai.azure.com/chat/completions",
        model="gpt-4o",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=150,
        speed_rank=6,
        expires_days=None,
    ),
    ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        model="meta-llama/llama-3.3-70b-instruct:free",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=50,
        speed_rank=7,
        expires_days=None,
    ),
    ProviderConfig(
        name="kimi",
        base_url="https://api.moonshot.cn/v1/chat/completions",
        model="moonshot-v1-8k",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=9999,
        speed_rank=8,
        expires_days=None,
    ),
    ProviderConfig(
        name="cloudflare",
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        auth_header="Authorization",
        auth_prefix="Bearer",
        daily_limit=10000,
        speed_rank=9,
        expires_days=None,
    ),
]

# ─── Provider State ────────────────────────────────────────────────────────────
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
    day_reset: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))

    def reset_daily_if_needed(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.day_reset != today:
            self.requests_today = 0
            self.day_reset = today
            log.info(f"[{self.config.name}] Daily counter reset.")

    @property
    def available(self) -> bool:
        self.reset_daily_if_needed()
        if self.expired:
            return False
        if not self.alive:
            return False
        if time.time() < self.cooldown_until:
            return False
        if self.requests_today >= self.config.daily_limit:
            return False
        return True

# ─── Session Store ─────────────────────────────────────────────────────────────
class SessionStore:
    """In-memory persistent session store (swap for Redis in production)."""
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
            log.info(f"New session created: {session_id}")
        return self._sessions[session_id]

    def append_message(self, session_id: str, role: str, content: str):
        session = self.get_or_create(session_id)
        session["history"].append({"role": role, "content": content})
        session["last_active"] = time.time()
        if role == "user":
            session["request_count"] += 1

    def get_history(self, session_id: str) -> list:
        return self.get_or_create(session_id).get("history", [])

    def prune_history(self, session_id: str, max_messages: int = 40):
        """Keep last N messages to avoid token overflow."""
        session = self.get_or_create(session_id)
        if len(session["history"]) > max_messages:
            history = session["history"]
            system = [m for m in history if m["role"] == "system"]
            rest = [m for m in history if m["role"] != "system"]
            session["history"] = system + rest[-max_messages:]

    def all_sessions(self) -> list:
        return list(self._sessions.values())

# ─── Health Checker ────────────────────────────────────────────────────────────
class HealthChecker:
    HEALTH_INTERVAL = 30
    FAST_RETRY_INTERVAL = 2

    def __init__(self, states: dict[str, ProviderState], session: aiohttp.ClientSession):
        self._states = states
        self._http = session
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._health_loop())
        log.info("Health checker started.")

    async def _health_loop(self):
        while self._running:
            await self._check_all()
            if not any(s.available for s in self._states.values()):
                log.warning("All providers down! Fast retrying in 2s...")
                await asyncio.sleep(self.FAST_RETRY_INTERVAL)
            else:
                await asyncio.sleep(self.HEALTH_INTERVAL)

    async def _check_all(self):
        tasks = [self._check_one(state) for state in self._states.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_one(self, state: ProviderState):
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
                if not prev:
                    log.info(f"[{name}] ✅ Provider is back ALIVE.")
            else:
                if prev:
                    log.warning(f"[{name}] ❌ Provider went DOWN.")
        except Exception as e:
            state.alive = False
            log.error(f"[{name}] Health check exception: {e}")

    async def _ping(self, state: ProviderState) -> bool:
        name = state.config.name
        key = KEYS.get(name, "")
        if not key:
            log.warning(f"[{name}] No API key found. Marking as expired.")
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
            return await self._ping_cloudflare(state, headers)

        try:
            async with self._http.post(
                state.config.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 401:
                    state.expired = True
                    log.error(f"[{name}] 🔑 API KEY EXPIRED (401). Removing from loop!")
                    return False
                if resp.status == 429:
                    state.cooldown_until = time.time() + 60
                    log.warning(f"[{name}] Rate limited. Cooling down 60s.")
                    return False
                return resp.status < 500
        except asyncio.TimeoutError:
            log.warning(f"[{name}] Ping timeout.")
            return False
        except Exception as e:
            log.error(f"[{name}] Ping error: {e}")
            return False

    async def _ping_cloudflare(self, state: ProviderState, headers: dict) -> bool:
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        try:
            async with self._http.post(
                state.config.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 401:
                    state.expired = True
                    log.error("[cloudflare] 🔑 KEY EXPIRED")
                    return False
                return resp.status < 500
        except Exception:
            return False

    def force_refresh(self):
        asyncio.create_task(self._check_all())

# ─── Core Gateway ──────────────────────────────────────────────────────────────
class AIGateway:
    SYSTEM_PROMPT = (
        "You are a helpful, honest, and knowledgeable AI assistant. "
        "Answer clearly and concisely. Maintain context from the conversation."
    )
    MAX_RETRIES = 9

    def __init__(self):
        self._http: Optional[aiohttp.ClientSession] = None
        self._states: dict[str, ProviderState] = {}
        self._sessions = SessionStore()
        self._health_checker: Optional[HealthChecker] = None

    async def start(self):
        self._http = aiohttp.ClientSession()
        for p in sorted(PROVIDERS, key=lambda x: x.speed_rank):
            self._states[p.name] = ProviderState(config=p)
        self._health_checker = HealthChecker(self._states, self._http)
        await self._health_checker.start()
        await asyncio.sleep(2)
        log.info("AI Gateway ready.")
        self._log_provider_status()

    async def stop(self):
        if self._health_checker:
            self._health_checker._running = False
        if self._http:
            await self._http.close()

    def _log_provider_status(self):
        log.info("=== Provider Status ===")
        for name, state in self._states.items():
            status = "✅ ALIVE" if state.available else ("💀 EXPIRED" if state.expired else "❌ DOWN")
            log.info(f"  [{name:12}] {status} | Daily: {state.requests_today}/{state.config.daily_limit}")

    def _get_sorted_alive_providers(self) -> list[ProviderState]:
        return sorted(
            [s for s in self._states.values() if s.available],
            key=lambda s: (s.requests_today / max(s.config.daily_limit, 1), s.config.speed_rank),
        )

    async def chat(self, session_id: str, user_message: str) -> dict:
        self._sessions.append_message(session_id, "user", user_message)
        self._sessions.prune_history(session_id)
        history = self._sessions.get_history(session_id)

        providers = self._get_sorted_alive_providers()
        if not providers:
            log.warning("All providers unavailable! Forcing health refresh...")
            self._health_checker.force_refresh()
            await asyncio.sleep(2)
            providers = self._get_sorted_alive_providers()
            if not providers:
                return {
                    "text": "⚠️ All AI providers are temporarily unavailable. Please try again in a few seconds.",
                    "provider": None,
                    "session_id": session_id,
                    "error": True,
                }

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + history

        last_error = None
        for state in providers:
            name = state.config.name
            try:
                log.info(f"[{session_id[:8]}] Trying provider: {name}")
                result = await self._call_provider(state, messages)
                if result:
                    state.requests_today += 1
                    state.last_used = time.time()
                    state.error_streak = 0
                    self._sessions.append_message(session_id, "assistant", result)
                    log.info(f"[{session_id[:8]}] ✅ Response from {name} ({len(result)} chars)")
                    return {
                        "text": result,
                        "provider": name,
                        "session_id": session_id,
                        "error": False,
                        "requests_today": state.requests_today,
                    }
            except RateLimitError:
                log.warning(f"[{name}] Rate limit hit. Cooldown 60s. Trying next...")
                state.cooldown_until = time.time() + 60
                state.error_streak += 1
            except ExpiredKeyError:
                log.error(f"[{name}] 🔑 Key expired during request! Removing.")
                state.expired = True
            except ProviderError as e:
                last_error = str(e)
                state.error_streak += 1
                if state.error_streak >= 3:
                    state.alive = False
                    state.cooldown_until = time.time() + 120
                    log.error(f"[{name}] 3 consecutive errors. Cooling down 120s.")
                log.warning(f"[{name}] Error: {e}. Trying next provider...")
            except Exception as e:
                last_error = str(e)
                log.error(f"[{name}] Unexpected error: {traceback.format_exc()}")

        self._health_checker.force_refresh()
        return {
            "text": f"⚠️ All AI providers are currently rate-limited or unavailable. Last error: {last_error}. Please retry in a moment.",
            "provider": None,
            "session_id": session_id,
            "error": True,
        }

    async def _call_provider(self, state: ProviderState, messages: list) -> Optional[str]:
        name = state.config.name
        if name == "cloudflare":
            return await self._call_cloudflare(state, messages)
        return await self._call_openai_compat(state, messages)

    async def _call_openai_compat(self, state: ProviderState, messages: list) -> Optional[str]:
        name = state.config.name
        key = KEYS[name]
        if not key:
            raise ExpiredKeyError(f"{name} API key missing")
            
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
                    raise ExpiredKeyError(f"{name} key invalid/expired")
                if resp.status == 429:
                    raise RateLimitError(f"{name} rate limited")
                if resp.status >= 500:
                    raise ProviderError(f"{name} server error {resp.status}")
                if resp.status >= 400:
                    err_msg = body.get("error", {}).get("message", str(body))
                    raise ProviderError(f"{name} client error {resp.status}: {err_msg}")

                choices = body.get("choices", [])
                if not choices:
                    raise ProviderError(f"{name} returned no choices: {body}")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ProviderError(f"{name} returned empty content")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError):
            raise
        except asyncio.TimeoutError:
            raise ProviderError(f"{name} request timed out")
        except aiohttp.ClientError as e:
            raise ProviderError(f"{name} network error: {e}")

    async def _call_cloudflare(self, state: ProviderState, messages: list) -> Optional[str]:
        key = KEYS["cloudflare"]
        if not key:
            raise ExpiredKeyError("cloudflare API key missing")
            
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
                    raise ExpiredKeyError("cloudflare key expired")
                if resp.status == 429:
                    raise RateLimitError("cloudflare rate limited")
                if resp.status >= 400:
                    raise ProviderError(f"cloudflare error {resp.status}: {body}")
                result = body.get("result", {})
                content = result.get("response", "")
                if not content:
                    raise ProviderError(f"cloudflare empty response: {body}")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError):
            raise
        except asyncio.TimeoutError:
            raise ProviderError("cloudflare timeout")
        except aiohttp.ClientError as e:
            raise ProviderError(f"cloudflare network: {e}")

    def get_status(self) -> dict:
        providers = {}
        for name, state in self._states.items():
            providers[name] = {
                "alive": state.alive,
                "available": state.available,
                "expired": state.expired,
                "requests_today": state.requests_today,
                "daily_limit": state.config.daily_limit,
                "utilization_pct": round(state.requests_today / state.config.daily_limit * 100, 1) if state.config.daily_limit > 0 else 0,
                "cooldown_until": datetime.fromtimestamp(state.cooldown_until).isoformat() if state.cooldown_until > time.time() else None,
                "error_streak": state.error_streak,
                "speed_rank": state.config.speed_rank,
                "model": state.config.model,
                "expires_days": state.config.expires_days,
            }
        return {
            "providers": providers,
            "active_sessions": len(self._sessions.all_sessions()),
            "alive_providers": sum(1 for s in self._states.values() if s.available),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions.get_or_create(sid)
        return sid

    def clear_session(self, session_id: str):
        if session_id in self._sessions._sessions:
            del self._sessions._sessions[session_id]
            log.info(f"Session {session_id} cleared.")


# ─── Custom Exceptions ─────────────────────────────────────────────────────────
class RateLimitError(Exception): pass
class ExpiredKeyError(Exception): pass
class ProviderError(Exception): pass


# ─── FastAPI Server ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:
    app = FastAPI(title="AI Gateway", version="1.0.0")
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
        message: str
        session_id: Optional[str] = None

    @app.post("/chat")
    async def chat_endpoint(req: ChatRequest):
        sid = req.session_id or gateway.new_session()
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        result = await gateway.chat(sid, req.message.strip())
        return result

    @app.post("/session/new")
    async def new_session():
        sid = gateway.new_session()
        return {"session_id": sid}

    @app.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        gateway.clear_session(session_id)
        return {"status": "cleared"}

    @app.get("/status")
    async def status():
        return gateway.get_status()

    @app.get("/health")
    async def health():
        st = gateway.get_status()
        if st["alive_providers"] == 0:
            raise HTTPException(status_code=503, detail="No providers available")
        return {"status": "ok", "alive_providers": st["alive_providers"]}


# ─── CLI Demo Mode ─────────────────────────────────────────────────────────────
async def cli_demo():
    print("\n🤖 AI Gateway - CLI Demo Mode")
    print("=" * 50)
    gw = AIGateway()
    await gw.start()

    sid = gw.new_session()
    print(f"Session ID: {sid}")
    print("Type 'quit' to exit, 'status' to see provider status.\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.lower() == "quit":
            break
        if msg.lower() == "status":
            st = gw.get_status()
            print(f"\n{'Provider':<14} {'Available':<12} {'Used/Limit':<16} {'Utilization'}")
            print("-" * 60)
            for name, info in st["providers"].items():
                avail = "✅" if info["available"] else ("💀" if info["expired"] else "❌")
                print(f"{name:<14} {avail:<12} {info['requests_today']:>5}/{info['daily_limit']:<10} {info['utilization_pct']}%")
            print(f"\nAlive providers: {st['alive_providers']} | Active sessions: {st['active_sessions']}\n")
            continue

        result = await gw.chat(sid, msg)
        provider_tag = f"[{result['provider']}]" if result['provider'] else "[FAILED]"
        print(f"\nAI {provider_tag}: {result['text']}\n")

    await gw.stop()
    print("Goodbye!")


if __name__ == "__main__":
    import sys
    if "--server" in sys.argv:
        if not HAS_FASTAPI:
            print("FastAPI not installed. Run: pip install fastapi uvicorn python-dotenv")
            sys.exit(1)
        import uvicorn
        uvicorn.run("gateway:app", host="0.0.0.0", port=8000, reload=False)
    else:
        asyncio.run(cli_demo())