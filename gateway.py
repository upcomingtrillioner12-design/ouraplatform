"""
AI Gateway - Multi-Provider Fallback System (FIXED)
====================================================
FIX: When session_id starts with "stateless-", gateway does NOT
accumulate history. oura_server owns all state.
Everything else unchanged.
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
    ProviderConfig("groq","https://api.groq.com/openai/v1/chat/completions","llama-3.1-70b-versatile","Authorization","Bearer",14400,1),
    ProviderConfig("cerebras","https://api.cerebras.ai/v1/chat/completions","llama3.1-70b","Authorization","Bearer",14400,2,30),
    ProviderConfig("mistral","https://api.mistral.ai/v1/chat/completions","mistral-large-latest","Authorization","Bearer",99999,3),
    ProviderConfig("gemini","https://generativelanguage.googleapis.com/v1beta/openai/chat/completions","gemini-2.0-flash-exp","Authorization","Bearer",1500,4,90),
    ProviderConfig("nvidia","https://integrate.api.nvidia.com/v1/chat/completions","nvidia/llama-3.1-70b-instruct","Authorization","Bearer",9999,5,180),
    ProviderConfig("github","https://models.inference.ai.azure.com/chat/completions","gpt-4o","Authorization","Bearer",150,6),
    ProviderConfig("openrouter","https://openrouter.ai/api/v1/chat/completions","meta-llama/llama-3.3-70b-instruct:free","Authorization","Bearer",50,7),
    ProviderConfig("kimi","https://api.moonshot.cn/v1/chat/completions","moonshot-v1-8k","Authorization","Bearer",9999,8),
    ProviderConfig("cloudflare",f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast","@cf/meta/llama-3.3-70b-instruct-fp8-fast","Authorization","Bearer",10000,9),
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
    day_reset: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))

    def reset_daily_if_needed(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.day_reset != today:
            self.requests_today = 0
            self.day_reset = today

    @property
    def available(self) -> bool:
        self.reset_daily_if_needed()
        return (not self.expired and self.alive
                and time.time() >= self.cooldown_until
                and self.requests_today < self.config.daily_limit)

class SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "id": session_id, "history": [],
                "created_at": time.time(), "last_active": time.time(),
                "request_count": 0, "user_id": session_id,
            }
        return self._sessions[session_id]

    def append_message(self, session_id: str, role: str, content: str):
        # ── FIX: never accumulate history for stateless sessions ──
        if session_id.startswith("stateless-"):
            return
        session = self.get_or_create(session_id)
        session["history"].append({"role": role, "content": content})
        session["last_active"] = time.time()
        if role == "user":
            session["request_count"] += 1

    def get_history(self, session_id: str) -> list:
        # ── FIX: stateless sessions return empty history ──
        if session_id.startswith("stateless-"):
            return []
        return self.get_or_create(session_id).get("history", [])

    def prune_history(self, session_id: str, max_messages: int = 40):
        if session_id.startswith("stateless-"):
            return
        session = self.get_or_create(session_id)
        if len(session["history"]) > max_messages:
            history = session["history"]
            system = [m for m in history if m["role"] == "system"]
            rest   = [m for m in history if m["role"] != "system"]
            session["history"] = system + rest[-max_messages:]

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
        await asyncio.gather(*[self._check_one(s) for s in self._states.values()], return_exceptions=True)

    async def _check_one(self, state):
        name = state.config.name
        if state.expired: return
        try:
            alive = await self._ping(state)
            prev = state.alive
            state.alive = alive
            state.last_health_check = time.time()
            if alive: state.error_streak = 0
            if alive and not prev: log.info(f"[{name}] ✅ Back alive.")
            elif not alive and prev: log.warning(f"[{name}] ❌ Went down.")
        except Exception as e:
            state.alive = False

    async def _ping(self, state) -> bool:
        name = state.config.name
        key = KEYS.get(name, "")
        if not key:
            state.expired = True
            return False
        headers = {state.config.auth_header: f"{state.config.auth_prefix} {key}", "Content-Type": "application/json"}
        payload = {"model": state.config.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        if name == "cloudflare":
            payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        try:
            async with self._http.post(state.config.base_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 401: state.expired = True; return False
                if resp.status == 429: state.cooldown_until = time.time() + 60; return False
                return resp.status < 500
        except: return False

    def force_refresh(self):
        asyncio.create_task(self._check_all())

class RateLimitError(Exception): pass
class ExpiredKeyError(Exception): pass
class ProviderError(Exception): pass

class AIGateway:
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
        if self._health_checker: self._health_checker._running = False
        if self._http: await self._http.close()

    def _get_sorted_alive_providers(self):
        return sorted(
            [s for s in self._states.values() if s.available],
            key=lambda s: (s.requests_today / max(s.config.daily_limit, 1), s.config.speed_rank),
        )

    async def chat(self, session_id: str, user_message: str) -> dict:
        # Only append to gateway history for real (non-stateless) sessions
        self._sessions.append_message(session_id, "user", user_message)
        self._sessions.prune_history(session_id)
        history = self._sessions.get_history(session_id)

        providers = self._get_sorted_alive_providers()
        if not providers:
            self._health_checker.force_refresh()
            await asyncio.sleep(2)
            providers = self._get_sorted_alive_providers()
            if not providers:
                return {"text": "⚠️ All AI providers temporarily unavailable. Please retry.", "provider": None, "session_id": session_id, "error": True}

        # ── FIX: for stateless sessions, user_message IS the full context blob
        # so just use it as a single user message without injecting old history
        if session_id.startswith("stateless-"):
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}]
        else:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + history

        last_error = None
        for state in providers:
            name = state.config.name
            try:
                result = await self._call_provider(state, messages)
                if result:
                    state.requests_today += 1
                    state.last_used = time.time()
                    state.error_streak = 0
                    self._sessions.append_message(session_id, "assistant", result)
                    log.info(f"[{session_id[:8]}] ✅ {name} ({len(result)} chars)")
                    return {"text": result, "provider": name, "session_id": session_id, "error": False}
            except RateLimitError:
                state.cooldown_until = time.time() + 60
                state.error_streak += 1
            except ExpiredKeyError:
                state.expired = True
            except ProviderError as e:
                last_error = str(e)
                state.error_streak += 1
                if state.error_streak >= 3:
                    state.alive = False
                    state.cooldown_until = time.time() + 120
            except Exception as e:
                last_error = str(e)
                log.error(f"[{name}] {traceback.format_exc()}")

        self._health_checker.force_refresh()
        return {"text": f"⚠️ All providers unavailable. Last error: {last_error}. Please retry.", "provider": None, "session_id": session_id, "error": True}

    async def _call_provider(self, state, messages):
        if state.config.name == "cloudflare":
            return await self._call_cloudflare(state, messages)
        return await self._call_openai_compat(state, messages)

    async def _call_openai_compat(self, state, messages):
        name = state.config.name
        key = KEYS[name]
        if not key: raise ExpiredKeyError(f"{name} key missing")
        headers = {state.config.auth_header: f"{state.config.auth_prefix} {key}", "Content-Type": "application/json"}
        if name == "openrouter":
            headers["HTTP-Referer"] = "https://ai-gateway.local"
            headers["X-Title"] = "AI Gateway"
        payload = {"model": state.config.model, "messages": messages, "max_tokens": 2048, "temperature": 0.7}
        try:
            async with self._http.post(state.config.base_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.json()
                if resp.status == 401: raise ExpiredKeyError(f"{name} expired")
                if resp.status == 429: raise RateLimitError(f"{name} rate limited")
                if resp.status >= 500: raise ProviderError(f"{name} server error {resp.status}")
                if resp.status >= 400: raise ProviderError(f"{name} error {resp.status}: {body.get('error',{}).get('message',str(body))}")
                choices = body.get("choices", [])
                if not choices: raise ProviderError(f"{name} no choices")
                content = choices[0].get("message", {}).get("content", "")
                if not content: raise ProviderError(f"{name} empty content")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError): raise
        except asyncio.TimeoutError: raise ProviderError(f"{name} timeout")
        except aiohttp.ClientError as e: raise ProviderError(f"{name} network: {e}")

    async def _call_cloudflare(self, state, messages):
        key = KEYS["cloudflare"]
        if not key: raise ExpiredKeyError("cloudflare key missing")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"messages": messages, "max_tokens": 2048}
        try:
            async with self._http.post(state.config.base_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.json()
                if resp.status == 401: raise ExpiredKeyError("cloudflare expired")
                if resp.status == 429: raise RateLimitError("cloudflare rate limited")
                if resp.status >= 400: raise ProviderError(f"cloudflare {resp.status}")
                content = body.get("result", {}).get("response", "")
                if not content: raise ProviderError("cloudflare empty")
                return content.strip()
        except (ExpiredKeyError, RateLimitError, ProviderError): raise
        except asyncio.TimeoutError: raise ProviderError("cloudflare timeout")
        except aiohttp.ClientError as e: raise ProviderError(f"cloudflare network: {e}")

    def get_status(self) -> dict:
        providers = {}
        for name, state in self._states.items():
            providers[name] = {
                "alive": state.alive, "available": state.available,
                "expired": state.expired,
                "requests_today": state.requests_today,
                "daily_limit": state.config.daily_limit,
                "utilization_pct": round(state.requests_today / state.config.daily_limit * 100, 1) if state.config.daily_limit > 0 else 0,
                "cooldown_until": datetime.fromtimestamp(state.cooldown_until).isoformat() if state.cooldown_until > time.time() else None,
                "error_streak": state.error_streak,
                "speed_rank": state.config.speed_rank,
                "model": state.config.model,
            }
        return {"providers": providers, "active_sessions": len(self._sessions.all_sessions()),
                "alive_providers": sum(1 for s in self._states.values() if s.available),
                "timestamp": datetime.utcnow().isoformat()}

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions.get_or_create(sid)
        return sid

    def clear_session(self, session_id: str):
        if session_id in self._sessions._sessions:
            del self._sessions._sessions[session_id]

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:
    app = FastAPI(title="AI Gateway", version="1.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    gateway = AIGateway()

    @app.on_event("startup")
    async def startup(): await gateway.start()

    @app.on_event("shutdown")
    async def shutdown(): await gateway.stop()

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
    async def new_session(): return {"session_id": gateway.new_session()}

    @app.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        gateway.clear_session(session_id)
        return {"status": "cleared"}

    @app.get("/status")
    async def status(): return gateway.get_status()

    @app.get("/health")
    async def health():
        st = gateway.get_status()
        if st["alive_providers"] == 0:
            raise HTTPException(status_code=503, detail="No providers available")
        return {"status": "ok", "alive_providers": st["alive_providers"]}

async def cli_demo():
    print("\n🤖 AI Gateway - CLI Demo")
    gw = AIGateway()
    await gw.start()
    sid = gw.new_session()
    print(f"Session: {sid}\n")
    while True:
        try: msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not msg: continue
        if msg.lower() == "quit": break
        result = await gw.chat(sid, msg)
        print(f"\nAI [{result['provider']}]: {result['text']}\n")
    await gw.stop()

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
