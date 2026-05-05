"""
OURA SOFTWIRE — AGI SUSTAINABLE MEMORY SERVER
==============================================
Version: 3.0.0 — HARDENED COMPLETE REWRITE
Author: Oura AGI Team

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Browser (session_id in localStorage)                    │
  └────────────────────┬────────────────────────────────────┘
                       │ POST /api/chat
  ┌────────────────────▼────────────────────────────────────┐
  │  oura_server.py (port 5000)                              │
  │  ├── SessionStore  (thread-safe, JSON-persistent)        │
  │  ├── OuraMemorySystem (Softwire Hopfield, auto-save)     │
  │  ├── ContextBuilder (clean, no speaker tags ever)        │
  │  ├── GatewayClient (circuit breaker + retry)             │
  │  ├── EchoGuard     (multi-pass decontaminator)           │
  │  └── FallbackEngine (rule-based, never echoes)           │
  └─────────────────────────────────────────────────────────┘

Fixes vs previous version:
  [FIX-01] Gateway called with OpenAI-compatible messages array
           via /v1/chat/completions — no more silent field ignore
  [FIX-02] stateless- prefix removed — gateway uses ephemeral
           session that expires immediately, not broken stateless mode
  [FIX-03] store_conversation_turn() bypassed — text stored raw,
           no [speaker] tag ever enters the pattern matrix
  [FIX-04] EchoGuard is multi-pass recursive until stable
  [FIX-05] Dead engine2-10 exec imports removed
  [FIX-06] Persistent memory: auto-save every N stores,
           auto-load on startup from .npz file
  [FIX-07] Gateway health check on startup with fallback plan
  [FIX-08] threading.Lock() on all session mutations
  [FIX-09] Similarity threshold raised to 0.55 (was 0.35)
  [FIX-10] Circuit breaker: 3 failures → skip gateway for 60s
  [FIX-11] Eternal session persistence via sessions.json
  [FIX-12] Tag contamination audit on every store()
"""

# ==============================================================
# STDLIB
# ==============================================================

import gc
import io
import json
import logging
import os
import re
import sys
import time
import threading
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================
# THIRD-PARTY
# ==============================================================

import numpy as np
import requests
from flask import Flask, request, jsonify, g as flask_g
from flask_cors import CORS

# ==============================================================
# LOGGING — structured, level-aware
# ==============================================================

LOG_LEVEL = os.environ.get("OURA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oura")

# ==============================================================
# PATHS & CONSTANTS
# ==============================================================

BASE_DIR        = Path(os.environ.get("OURA_BASE_DIR", r"C:\Users\linka\OneDrive"))
MEMORY_FILE     = BASE_DIR / "oura_memory"          # .npz appended by numpy
SESSIONS_FILE   = BASE_DIR / "oura_sessions.json"
GATEWAY_URL     = os.environ.get("GATEWAY_URL", "http://localhost:8000")
API_KEY         = os.environ.get("OURA_API_KEY", "oura-super-secret-key-change-this")
PORT            = int(os.environ.get("OURA_PORT", 5000))

# Memory tuning
PATTERN_LENGTH      = 512
SOFTWIRE_G          = 11.0
CHUNK_WORDS         = 60
OVERLAP_WORDS       = 20
RECALL_THRESHOLD    = 0.55    # was 0.35 — too low caused false recalls
AUTO_SAVE_EVERY     = 10      # save memory every N stores
HISTORY_MAX_TURNS   = 40      # max turns kept per session
CONTEXT_TURNS       = 6       # how many recent turns sent to AI

# Circuit breaker
CB_FAILURE_LIMIT    = 3       # open after N consecutive failures
CB_RECOVERY_SECS    = 60      # seconds before retry allowed

# ==============================================================
# IMPORTS — SOFTWIRE ENGINES
# ==============================================================

sys.path.insert(0, str(BASE_DIR))

try:
    from text_encoder import OuraMemorySystem, TextEncoder
    log.info("✓ text_encoder imported")
except Exception as exc:
    log.critical("✗ text_encoder import failed: %s", exc)
    sys.exit(1)

# softwireengine1 imported for direct recall if needed
_SoftwireCoreV2 = None
try:
    from softwireengine1 import SoftwireCoreV2 as _SoftwireCoreV2
    log.info("✓ softwireengine1 imported")
except Exception as exc:
    log.warning("softwireengine1 not available: %s", exc)

# NOTE: engine2-10 are NOT imported — they are incompatible dead code
# If you need a specific engine, instantiate it explicitly here.

# ==============================================================
# SECTION 1: THREAD-SAFE SESSION STORE
# ==============================================================

class SessionStore:
    """
    Thread-safe session store with JSON persistence.

    Session schema:
      {
        "history":   [{"role": "user"|"assistant", "content": str}],
        "user_name": str | None,
        "facts":     [str],
        "created":   float (unix timestamp),
        "updated":   float,
      }
    """

    _EMPTY = lambda self: {   # noqa: E731
        "history":   [],
        "user_name": None,
        "facts":     [],
        "created":   time.time(),
        "updated":   time.time(),
    }

    def __init__(self, path: Path):
        self._path   = path
        self._lock   = threading.RLock()
        self._data: Dict[str, dict] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                # Validate each session
                for sid, sdata in raw.items():
                    if isinstance(sdata, dict) and "history" in sdata:
                        self._data[sid] = sdata
                log.info("✓ Sessions loaded: %d sessions", len(self._data))
            except Exception as exc:
                log.warning("Session load failed (starting fresh): %s", exc)
        else:
            log.info("No sessions file found — starting fresh")

    def save(self):
        """Write sessions to disk. Called periodically and on shutdown."""
        try:
            with self._lock:
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, ensure_ascii=False)
                tmp.replace(self._path)
            log.debug("Sessions saved (%d sessions)", len(self._data))
        except Exception as exc:
            log.error("Session save failed: %s", exc)

    # ── access ────────────────────────────────────────────────

    def get(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = self._EMPTY()
            return self._data[session_id]

    def update(self, session_id: str, **kwargs):
        with self._lock:
            s = self.get(session_id)
            s.update(kwargs)
            s["updated"] = time.time()

    def append_turn(self, session_id: str, role: str, content: str):
        with self._lock:
            s = self.get(session_id)
            s["history"].append({"role": role, "content": content})
            # Trim to limit
            if len(s["history"]) > HISTORY_MAX_TURNS:
                s["history"] = s["history"][-HISTORY_MAX_TURNS:]
            s["updated"] = time.time()

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def count(self) -> int:
        with self._lock:
            return len(self._data)


# ==============================================================
# SECTION 2: PERSISTENT SOFTWIRE MEMORY WRAPPER
# ==============================================================

class PersistentMemory:
    """
    Wraps OuraMemorySystem with:
      - Tag-free storage (no [speaker] prefix ever stored)
      - Auto-save every N stores
      - Load on init from disk
      - Similarity-threshold-gated recall
      - Tag contamination audit
    """

    # Regex: strip any [role] prefix that might have crept in historically
    _TAG_RE = re.compile(
        r'^\s*\[(user|assistant|system|User|Assistant|System)\]\s*',
        re.IGNORECASE
    )

    def __init__(self):
        self._lock         = threading.Lock()
        self._store_count  = 0
        self._oms: OuraMemorySystem = None
        self._init_oms()
        self._load()

    def _init_oms(self):
        self._oms = OuraMemorySystem(
            pattern_length=PATTERN_LENGTH,
            g=SOFTWIRE_G,
            chunk_words=CHUNK_WORDS,
            overlap=OVERLAP_WORDS,
        )
        log.info("✓ OuraMemorySystem initialized (N=%d, g=%.1f)", PATTERN_LENGTH, SOFTWIRE_G)

    # ── persistence ───────────────────────────────────────────

    def _load(self):
        npz_path = Path(str(MEMORY_FILE) + ".npz")
        if npz_path.exists():
            try:
                self._oms.load(str(MEMORY_FILE))
                n = self._pattern_count()
                log.info("✓ Memory loaded: %d patterns from %s", n, npz_path)
            except Exception as exc:
                log.warning("Memory load failed (starting fresh): %s", exc)
        else:
            log.info("No memory file found — starting fresh")

    def save(self):
        try:
            with self._lock:
                self._oms.save(str(MEMORY_FILE))
            log.debug("Memory saved (%d patterns)", self._pattern_count())
        except Exception as exc:
            log.error("Memory save failed: %s", exc)

    # ── internal helpers ──────────────────────────────────────

    def _pattern_count(self) -> int:
        return getattr(getattr(self._oms, 'network', None), 'n_patterns', 0)

    def _clean_text(self, text: str) -> str:
        """Remove any speaker tag prefix. Called before store AND after recall."""
        return self._TAG_RE.sub('', text).strip()

    def _audit_text(self, text: str) -> str:
        """
        Audit: warn if text still contains a tag after cleaning.
        Returns cleaned text.
        """
        cleaned = self._clean_text(text)
        if cleaned != text:
            log.warning("[AUDIT] Tag contamination stripped: %r → %r", text[:60], cleaned[:60])
        return cleaned

    # ── public API ────────────────────────────────────────────

    def store(self, speaker: str, text: str) -> Optional[List[int]]:
        """
        Store text without ANY speaker tag.
        [FIX-03] We do NOT call store_conversation_turn() which embeds [speaker].
        We call the underlying chunked store directly with clean text.
        """
        text = self._audit_text(text)
        if not text:
            return None
        try:
            with self._lock:
                # Call store_text directly — bypasses [speaker] embedding
                indices = self._oms.store_text(text)
                self._store_count += 1
                if self._store_count % AUTO_SAVE_EVERY == 0:
                    self._oms.save(str(MEMORY_FILE))
                    log.debug("Auto-saved memory at store #%d", self._store_count)
            log.debug("[MEMORY] Stored (%s): %s…", speaker, text[:60])
            return indices
        except Exception as exc:
            log.error("[MEMORY] store() failed: %s", exc)
            return None

    def recall(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Recall most similar memory to query.
        Returns {"text": str, "similarity": float} or None.
        Threshold: RECALL_THRESHOLD (0.55).
        """
        query = self._audit_text(query)
        if not query:
            return None
        try:
            with self._lock:
                result = self._oms.recall_from_text(query, noise_fraction=0.05)
            if result and result.best_match_text and result.similarity >= RECALL_THRESHOLD:
                clean = self._audit_text(result.best_match_text)
                return {"text": clean, "similarity": float(result.similarity)}
        except Exception as exc:
            log.error("[MEMORY] recall() failed: %s", exc)
        return None

    def search(self, query: str, top_k: int = 2) -> List[Tuple[float, str]]:
        """
        Return top_k similar memories above threshold.
        Each element: (similarity, clean_text).
        """
        query = self._audit_text(query)
        if not query:
            return []
        try:
            with self._lock:
                results = self._oms.search_similar(query, top_k=top_k, threshold=RECALL_THRESHOLD)
            cleaned = []
            for item in results:
                # item is (score, RecallResult) or (score, text) — handle both
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    score = float(item[0])
                    payload = item[1]
                    if hasattr(payload, 'text'):
                        text = self._audit_text(payload.text)
                    else:
                        text = self._audit_text(str(payload))
                    cleaned.append((score, text))
            return cleaned
        except Exception as exc:
            log.error("[MEMORY] search() failed: %s", exc)
            return []

    def stats(self) -> Dict[str, Any]:
        net = getattr(self._oms, 'network', None)
        return {
            "n_patterns":      getattr(net, 'n_patterns', 0),
            "pattern_length":  getattr(net, 'N', PATTERN_LENGTH),
            "g":               getattr(net, 'g', SOFTWIRE_G),
            "temperature":     getattr(net, 'T', round(1.0 / SOFTWIRE_G, 4)),
            "total_stores":    self._store_count,
        }


# ==============================================================
# SECTION 3: ECHO GUARD — MULTI-PASS DECONTAMINATOR
# ==============================================================

class EchoGuard:
    """
    [FIX-04] Multi-pass recursive echo decontaminator.

    Removes:
      1. [role] tags anywhere in text
      2. "I remember something related: ..." chains (recursive)
      3. Repeated sentences (deduplication)
      4. Repeated phrases within a sentence
      5. Wrapping quotes
      6. System prompt fragments leaked into response
    """

    _ROLE_TAGS = re.compile(
        r'\[(user|assistant|system|ASSISTANT|USER|SYSTEM)\]\s*',
        re.IGNORECASE
    )
    _ECHO_PHRASE = re.compile(
        r"I remember something related\s*[:–—-]?\s*[\"']?",
        re.IGNORECASE
    )
    _MEMORY_LEAK = re.compile(
        r"(Relevant past memory|Related context|Known facts)\s*[:]\s*[^\n]*",
        re.IGNORECASE
    )
    _WRAPPING_QUOTES = re.compile(r'^["\'\s]+|["\'\s]+$')
    _SENTENCE_SPLIT  = re.compile(r'(?<=[.!?])\s+')

    # Phrases that indicate the AI is echoing the system prompt
    _SYSTEM_ECHOES = [
        "you are oura",
        "be conversational and helpful",
        "use the context below naturally",
        "never say 'i remember something related'",
        "the user's name is",       # too generic — only flag if it starts the response
    ]

    @classmethod
    def clean(cls, text: str, max_passes: int = 5) -> str:
        if not text:
            return "I'm here. What would you like to talk about?"

        prev = None
        passes = 0
        while prev != text and passes < max_passes:
            prev = text
            text = cls._pass(text)
            passes += 1

        if passes > 1:
            log.debug("[ECHOGUARD] Needed %d passes to stabilize", passes)

        text = text.strip()
        return text if text else "I'm here. What would you like to talk about?"

    @classmethod
    def _pass(cls, text: str) -> str:
        # 1. Strip [role] tags
        text = cls._ROLE_TAGS.sub('', text)

        # 2. Remove "I remember something related: ..." chains
        text = cls._ECHO_PHRASE.sub('', text)

        # 3. Remove system prompt memory fragments that leaked
        text = cls._MEMORY_LEAK.sub('', text)

        # 4. Strip wrapping quotes/whitespace
        text = cls._WRAPPING_QUOTES.sub('', text)

        # 5. Deduplicate sentences
        sentences = cls._SENTENCE_SPLIT.split(text)
        seen: Dict[str, int] = {}
        deduped: List[str] = []
        for s in sentences:
            key = re.sub(r'\s+', ' ', s.strip().lower())[:100]
            if not key:
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 1:
                deduped.append(s.strip())
        text = ' '.join(deduped)

        # 6. Check for system prompt echo at START of response
        first_80 = text[:80].lower()
        for phrase in cls._SYSTEM_ECHOES[:4]:   # skip 'the user's name is'
            if phrase in first_80:
                # Nuclear: drop everything up to first period/newline after the phrase
                idx = text.lower().find(phrase)
                rest = text[idx:]
                end  = re.search(r'[.\n]', rest)
                if end:
                    text = text[idx + end.start() + 1:].strip()
                else:
                    text = ''
                break

        return text


# ==============================================================
# SECTION 4: GATEWAY CLIENT — CIRCUIT BREAKER + RETRY
# ==============================================================

class GatewayClient:
    """
    [FIX-01] Sends full OpenAI-compatible messages array.
    [FIX-02] Uses ephemeral session_id (no stateless- prefix).
    [FIX-10] Circuit breaker: opens after CB_FAILURE_LIMIT consecutive failures.

    Tries endpoints in order:
      1. POST /v1/chat/completions  (OpenAI-compatible)
      2. POST /chat                 (gateway native, with messages)
    """

    def __init__(self, base_url: str):
        self._base = base_url.rstrip('/')
        self._lock            = threading.Lock()
        self._failures        = 0
        self._open_until      = 0.0
        self._healthy: Optional[bool] = None

    # ── circuit breaker ───────────────────────────────────────

    def _is_open(self) -> bool:
        with self._lock:
            if self._open_until > 0 and time.time() < self._open_until:
                return True
            return False

    def _record_success(self):
        with self._lock:
            self._failures   = 0
            self._open_until = 0.0

    def _record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= CB_FAILURE_LIMIT:
                self._open_until = time.time() + CB_RECOVERY_SECS
                log.warning(
                    "[GATEWAY] Circuit breaker OPEN — skipping gateway for %ds",
                    CB_RECOVERY_SECS
                )

    # ── health check ─────────────────────────────────────────

    def health_check(self) -> bool:
        """Called once on startup. Sets _healthy."""
        try:
            r = requests.get(f"{self._base}/health", timeout=3)
            self._healthy = r.status_code == 200
        except Exception:
            self._healthy = False
        log.info(
            "[GATEWAY] Health check: %s at %s",
            "OK" if self._healthy else "UNREACHABLE",
            self._base,
        )
        return self._healthy

    # ── main send ────────────────────────────────────────────

    def send(
        self,
        messages: List[Dict[str, str]],
        timeout: int = 45,
    ) -> Tuple[Optional[str], str]:
        """
        Send messages array to gateway.
        Returns (response_text, provider) or (None, "").
        """
        if self._is_open():
            log.debug("[GATEWAY] Circuit breaker open — skipping")
            return None, ""

        # Ephemeral session: never the same twice → gateway accumulates nothing
        ephemeral_sid = "eph-" + str(uuid.uuid4())

        # Extract user_message (last user turn)
        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = m["content"]
                break

        endpoints = [
            # [FIX-01] Prefer OpenAI-compatible endpoint
            {
                "url":     f"{self._base}/v1/chat/completions",
                "payload": {
                    "model":       "gpt-4o-mini",
                    "messages":    messages,
                    "temperature": 0.7,
                    "max_tokens":  800,
                },
                "parser": self._parse_openai,
            },
            # Fallback: gateway native with messages field added
            {
                "url":     f"{self._base}/chat",
                "payload": {
                    "message":    user_message,
                    "session_id": ephemeral_sid,
                    "messages":   messages,     # gateway may or may not use this
                },
                "parser": self._parse_native,
            },
        ]

        for ep in endpoints:
            try:
                log.debug("[GATEWAY] Trying %s", ep["url"])
                resp = requests.post(
                    ep["url"],
                    json=ep["payload"],
                    headers={
                        "Content-Type":  "application/json",
                        "Authorization": f"Bearer {API_KEY}",
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    text, provider = ep["parser"](resp.json())
                    if text:
                        self._record_success()
                        log.debug("[GATEWAY] OK from %s via %s", provider, ep["url"])
                        return text, provider
                else:
                    log.warning(
                        "[GATEWAY] HTTP %d from %s", resp.status_code, ep["url"]
                    )
            except requests.exceptions.Timeout:
                log.warning("[GATEWAY] Timeout on %s", ep["url"])
                self._record_failure()
            except requests.exceptions.ConnectionError:
                log.warning("[GATEWAY] Connection refused on %s", ep["url"])
                self._record_failure()
                break   # Both endpoints on same host — no point retrying
            except Exception as exc:
                log.error("[GATEWAY] Unexpected error on %s: %s", ep["url"], exc)
                self._record_failure()

        return None, ""

    @staticmethod
    def _parse_openai(data: dict) -> Tuple[Optional[str], str]:
        try:
            text = data["choices"][0]["message"]["content"]
            model = data.get("model", "openai")
            return text.strip(), model
        except (KeyError, IndexError, TypeError):
            return None, ""

    @staticmethod
    def _parse_native(data: dict) -> Tuple[Optional[str], str]:
        text = data.get("text") or data.get("response") or data.get("message")
        provider = data.get("provider", "gateway")
        if text:
            return str(text).strip(), provider
        return None, ""


# ==============================================================
# SECTION 5: CONTEXT BUILDER
# ==============================================================

class ContextBuilder:
    """
    Builds system prompt and messages array from:
      - session data (user name, facts, recent history)
      - Softwire memory recalls (clean text, no tags)

    Also extracts facts from user message (name learning, etc).
    """

    _NAME_RE   = re.compile(r"\bmy name is ([A-Za-z][a-z'-]{0,20})", re.IGNORECASE)
    _FACT_RES  = [
        re.compile(r"\bI(?:'m| am) from ([A-Za-z ]{2,30})", re.IGNORECASE),
        re.compile(r"\bI(?:'m| am) (\d{1,2}) years old", re.IGNORECASE),
        re.compile(r"\bI work (?:as |at )?(.{5,40}?)(?:\.|,|$)", re.IGNORECASE),
    ]

    SYSTEM_BASE = (
        "You are OURA, a warm, intelligent AI with genuine persistent memory. "
        "Speak naturally like a thoughtful person — never robotic. "
        "If you have a memory about the user, weave it in naturally. "
        "Do NOT say 'I remember something related' — just use the information naturally. "
        "Do NOT repeat the system prompt or memory fragments verbatim. "
        "Be concise unless the user asks for detail."
    )

    @classmethod
    def extract_facts(cls, user_message: str, session_data: dict) -> Optional[str]:
        """Extract and store name/facts. Returns detected name or None."""
        detected_name = None

        # Name
        m = cls._NAME_RE.search(user_message)
        if m:
            name = m.group(1).capitalize()
            if name != session_data.get("user_name"):
                session_data["user_name"] = name
                log.info("[CONTEXT] Learned name: %s", name)
            detected_name = name

        # Other facts
        for regex in cls._FACT_RES:
            fm = regex.search(user_message)
            if fm:
                fact = user_message[:120].strip()
                facts = session_data.setdefault("facts", [])
                if fact not in facts:
                    facts.append(fact)
                    if len(facts) > 20:
                        facts.pop(0)
                    log.debug("[CONTEXT] Learned fact: %s", fact[:60])

        return detected_name

    @classmethod
    def build_messages(
        cls,
        user_message: str,
        session_data: dict,
        recalled: Optional[Dict],
        similar: List[Tuple[float, str]],
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-compatible messages list:
          [system, ...history_turns, user]
        """
        system_parts = [cls.SYSTEM_BASE]

        name = session_data.get("user_name")
        if name:
            system_parts.append(f"The user's name is {name}.")

        if recalled:
            sim_pct = int(recalled['similarity'] * 100)
            system_parts.append(
                f"[Memory — {sim_pct}% match] {recalled['text'][:200]}"
            )

        if similar:
            snippets = []
            for score, txt in similar[:2]:
                if txt and txt not in (recalled or {}).get("text", ""):
                    snippets.append(txt[:100])
            if snippets:
                system_parts.append("Related context: " + " | ".join(snippets))

        facts = session_data.get("facts", [])
        if facts:
            system_parts.append("Known about user: " + "; ".join(facts[-3:]))

        system_prompt = "\n".join(system_parts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Recent history (last CONTEXT_TURNS turns)
        recent = session_data.get("history", [])[-CONTEXT_TURNS:]
        for turn in recent:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages


# ==============================================================
# SECTION 6: FALLBACK ENGINE
# ==============================================================

class FallbackEngine:
    """
    Rule-based response generator used when gateway is unavailable.
    Never reads raw memory — uses only pre-processed context.
    Never echoes speaker tags.
    """

    @staticmethod
    def respond(user_message: str, session_data: dict, context: dict) -> str:
        name     = session_data.get("user_name", "")
        greeting = f", {name}" if name else ""
        lower    = user_message.lower()

        recalled = context.get("recalled")

        # Use recalled memory naturally if high confidence
        if recalled and recalled["similarity"] > 0.70:
            snippet = recalled["text"][:150]
            return (
                f"That reminds me of something we've discussed before{greeting}. "
                f"{snippet}. Does that relate to what you're asking?"
            )

        # Name introduction
        if "my name is" in lower:
            return (
                f"Great to meet you{greeting}! "
                "I'll remember that for our future conversations. What's on your mind?"
            )

        # Pickup lines
        if "pickup line" in lower or "chat up" in lower:
            lines = [
                "Are you a Wi-Fi signal? Because I feel a strong connection.",
                "No cap, you just broke my algorithm.",
                "You must be trending, because you're all I see.",
                "Are you my charger? Because I die without you.",
                "Lowkey obsessed with you, not gonna lie.",
                "Did it hurt when you fell from the For You page?",
            ]
            import random
            chosen = random.sample(lines, min(4, len(lines)))
            numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(chosen))
            return f"Here are some pickup lines{greeting}:\n\n{numbered}\n\nWant more? 😄"

        # Memory questions
        if any(w in lower for w in ["remember", "recall", "forgot", "memory"]):
            n = context.get("patterns_stored", 0)
            return (
                f"I have {n} memories stored{greeting}. "
                "Ask me anything specific and I'll try to recall it!"
            )

        # Default
        return (
            f"I'm here and listening{greeting}. "
            "What would you like to talk about?"
        )


# ==============================================================
# SECTION 7: PERIODIC SAVE THREAD
# ==============================================================

class PeriodicSaver(threading.Thread):
    """Background thread that saves sessions + memory every 5 minutes."""

    def __init__(self, session_store: 'SessionStore', memory: 'PersistentMemory'):
        super().__init__(daemon=True, name="PeriodicSaver")
        self._sessions = session_store
        self._memory   = memory
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.wait(timeout=300):   # every 5 minutes
            self._sessions.save()
            self._memory.save()
            log.debug("[SAVER] Periodic save complete")

    def stop(self):
        self._stop_evt.set()


# ==============================================================
# SECTION 8: INITIALIZE ALL SINGLETONS
# ==============================================================

# Memory
_memory = PersistentMemory()

# Session store
_sessions = SessionStore(SESSIONS_FILE)

# Gateway client
_gateway = GatewayClient(GATEWAY_URL)

# Context builder (stateless class methods)
_context_builder = ContextBuilder()

# Fallback engine (stateless)
_fallback = FallbackEngine()

# Echo guard (stateless class methods)
_echo_guard = EchoGuard()

# ==============================================================
# SECTION 9: FLASK APPLICATION
# ==============================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.before_request
def _start_timer():
    flask_g.t_start = time.perf_counter()


@app.after_request
def _add_headers(response):
    elapsed_ms = int((time.perf_counter() - flask_g.t_start) * 1000)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    response.headers["X-Engine"] = "softwire-agi-v3"
    return response


# ── /api/chat ─────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    raw = request.get_json(silent=True) or {}

    user_message: str = str(raw.get("message", "")).strip()
    session_id:   str = str(raw.get("session_id", "")).strip()

    if not user_message:
        return jsonify({"error": "empty message"}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    log.info("[CHAT] sid=%s…  msg=%r", session_id[:8], user_message[:80])

    # 1. Get session (thread-safe)
    session_data = _sessions.get(session_id)

    # 2. Extract facts BEFORE building context
    ContextBuilder.extract_facts(user_message, session_data)

    # 3. [FIX-03] Store user turn RAW — no [speaker] prefix
    _memory.store("user", user_message)

    # 4. Recall from Softwire (clean, tag-free, threshold 0.55)
    recalled = _memory.recall(user_message)
    similar  = _memory.search(user_message, top_k=2)

    # Log recall quality
    if recalled:
        log.debug("[RECALL] %.2f: %s…", recalled["similarity"], recalled["text"][:50])
    else:
        log.debug("[RECALL] No match above threshold")

    # 5. Build OpenAI-compatible messages array
    messages = ContextBuilder.build_messages(
        user_message  = user_message,
        session_data  = session_data,
        recalled      = recalled,
        similar       = similar,
    )

    # 6. Append user turn to history ONCE (before gateway call)
    _sessions.append_turn(session_id, "user", user_message)

    # 7. Call gateway
    response_text, provider = _gateway.send(messages, timeout=45)

    # 8. Fallback if gateway failed / circuit open
    used_fallback = False
    if not response_text:
        context_for_fallback = {
            "recalled":        recalled,
            "patterns_stored": _memory.stats()["n_patterns"],
        }
        response_text = _fallback.respond(user_message, session_data, context_for_fallback)
        provider      = "fallback"
        used_fallback = True

    # 9. [FIX-04] Multi-pass echo decontamination
    response_text = EchoGuard.clean(response_text)

    # 10. Store assistant reply + append to history
    _memory.store("assistant", response_text)
    _sessions.append_turn(session_id, "assistant", response_text)

    # 11. Persist sessions after every turn
    _sessions.save()

    stats = _memory.stats()
    log.info(
        "[CHAT] Done — provider=%s fallback=%s patterns=%d",
        provider, used_fallback, stats["n_patterns"]
    )

    return jsonify({
        "text":             response_text,
        "session_id":       session_id,
        "provider":         provider,
        "patterns_stored":  stats["n_patterns"],
        "user_name":        session_data.get("user_name"),
        "recall_similarity": recalled["similarity"] if recalled else 0.0,
        "used_fallback":    used_fallback,
    })


# ── /api/memory/stats ─────────────────────────────────────────

@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    return jsonify(_memory.stats())


# ── /api/memory/save ──────────────────────────────────────────

@app.route("/api/memory/save", methods=["POST"])
def memory_save():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    _memory.save()
    _sessions.save()
    return jsonify({"status": "saved", "stats": _memory.stats()})


# ── /api/session/<id> ─────────────────────────────────────────

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    s = _sessions.get(session_id)
    return jsonify({
        "session_id":  session_id,
        "user_name":   s.get("user_name"),
        "facts":       s.get("facts", []),
        "turn_count":  len(s.get("history", [])),
        "created":     s.get("created"),
        "updated":     s.get("updated"),
    })


# ── /api/session/<id>/history ─────────────────────────────────

@app.route("/api/session/<session_id>/history", methods=["GET"])
def get_history(session_id: str):
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    s = _sessions.get(session_id)
    return jsonify({
        "session_id": session_id,
        "history":    s.get("history", []),
    })


# ── /api/memory/recall ────────────────────────────────────────

@app.route("/api/memory/recall", methods=["POST"])
def manual_recall():
    """Test endpoint: recall a memory by query string."""
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    data  = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    recalled = _memory.recall(query)
    similar  = _memory.search(query, top_k=3)
    return jsonify({
        "recalled": recalled,
        "similar":  [{"similarity": s, "text": t} for s, t in similar],
    })


# ── /health ───────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    stats = _memory.stats()
    return jsonify({
        "status":           "ok",
        "engine":           "softwire-agi-v3",
        "patterns_stored":  stats["n_patterns"],
        "sessions_active":  _sessions.count(),
        "gateway_healthy":  _gateway._healthy,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    })


# ── /api/status ───────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    stats = _memory.stats()
    return jsonify({
        "status":          "operational",
        "version":         "3.0.0",
        "patterns_stored": stats["n_patterns"],
        "sessions":        _sessions.count(),
        "memory_type":     "OuraMemorySystem-PersistentWrapper",
        "gateway_url":     GATEWAY_URL,
        "gateway_healthy": _gateway._healthy,
        "circuit_breaker": {
            "failures":   _gateway._failures,
            "open_until": _gateway._open_until,
            "is_open":    _gateway._is_open(),
        },
        "memory_stats":    stats,
    })


# ==============================================================
# SECTION 10: STARTUP
# ==============================================================

def startup():
    """
    Called once before Flask starts serving.
    Runs health check, starts periodic saver.
    """
    log.info("=" * 60)
    log.info("  OURA SOFTWIRE — AGI MEMORY SERVER v3.0.0")
    log.info("=" * 60)
    log.info("  Memory file:   %s.npz", MEMORY_FILE)
    log.info("  Sessions file: %s", SESSIONS_FILE)
    log.info("  Gateway URL:   %s", GATEWAY_URL)
    log.info("  Port:          %d", PORT)
    log.info("  Pattern length: %d | g=%.1f | threshold=%.2f",
             PATTERN_LENGTH, SOFTWIRE_G, RECALL_THRESHOLD)

    stats = _memory.stats()
    log.info("  Patterns loaded: %d", stats["n_patterns"])
    log.info("  Sessions loaded: %d", _sessions.count())

    # Gateway health check
    _gateway.health_check()

    # Start periodic saver thread
    saver = PeriodicSaver(_sessions, _memory)
    saver.start()
    log.info("  Periodic saver started (interval: 5min)")

    log.info("=" * 60)
    log.info("  Listening at http://0.0.0.0:%d", PORT)
    log.info("=" * 60)


# ==============================================================
# SECTION 11: ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    startup()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,    # reloader double-starts threads → disable
    )
