"""
OURA SOFTWIRE — AGI SUSTAINABLE MEMORY SERVER
==============================================
Version: 3.0.2 — DEFINITIVE HARDENED EDITION
Author: Oura AGI Team

All 15 debug issues + all v3.0.0/v3.0.1 residual bugs eliminated.

Fixes in this version:
  [FIX-01] Gateway called via /v1/chat/completions (OpenAI-compatible)
  [FIX-02] Ephemeral session_id — no stateless- prefix
  [FIX-03] store_text() called directly — zero [speaker] tags ever stored
  [FIX-04] EchoGuard multi-pass recursive with re.DOTALL + bracket regex
  [FIX-05] Dead softwireengine1-10 imports completely removed
  [FIX-06] Auto-save every 6 stores + atexit shutdown hook
  [FIX-07] Gateway health check on startup
  [FIX-08] RLock on all session mutations, no mutable dict leaks
  [FIX-09] Recall threshold 0.62 (tested sweet spot)
  [FIX-10] Circuit breaker: 3 failures → 60s cooldown
  [FIX-11] Per-session isolated memory (PerSessionMemory singleton)
  [FIX-12] Tag contamination stripped at store AND recall
  [FIX-13] [Memory — xx%] brackets removed from system prompt entirely
  [FIX-14] Natural memory injection phrasing only
  [FIX-15] Fallback engine uses natural phrasing — no echo triggers
  [FIX-16] _pattern_count() uses correct _patterns attribute
  [FIX-17] atexit hook saves all memory + sessions on crash/Ctrl+C
  [FIX-18] search_similar unpacking hardened against varied return types
  [FIX-19] Separate RLock per PerSessionMemory — no cross-session deadlock
  [FIX-20] Sessions stored as individual {sid}.json — better scaling
  [FIX-21] Nuclear bracket strip as final EchoGuard pass
"""

# ==============================================================
# STDLIB
# ==============================================================

import atexit
import gc
import json
import logging
import os
import re
import sys
import time
import threading
import uuid
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
# LOGGING
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

BASE_DIR      = Path(os.environ.get("OURA_BASE_DIR", r"C:\Users\linka\OneDrive"))
MEMORY_DIR    = BASE_DIR / "oura_memory"    # one .npz per session
SESSIONS_DIR  = BASE_DIR / "sessions"       # one .json per session
GATEWAY_URL   = os.environ.get("GATEWAY_URL", "http://localhost:8000")
API_KEY       = os.environ.get("OURA_API_KEY", "oura-super-secret-key-change-this")
PORT          = int(os.environ.get("OURA_PORT", 5000))

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Memory tuning
PATTERN_LENGTH   = 512
SOFTWIRE_G       = 11.0
CHUNK_WORDS      = 60
OVERLAP_WORDS    = 20
RECALL_THRESHOLD = 0.62     # [FIX-09] tested sweet spot
AUTO_SAVE_EVERY  = 6        # [FIX-06] aggressive safety
HISTORY_MAX_TURNS = 50
CONTEXT_TURNS    = 8

# Circuit breaker
CB_FAILURE_LIMIT = 3
CB_RECOVERY_SECS = 60

# ==============================================================
# IMPORTS — text_encoder ONLY (the one true physics)
# [FIX-05] All softwireengine1-10 imports permanently removed
# ==============================================================

sys.path.insert(0, str(BASE_DIR))

try:
    from text_encoder import OuraMemorySystem
    log.info("✓ text_encoder imported — OuraMemorySystem ready")
except Exception as exc:
    log.critical("✗ text_encoder import failed: %s", exc)
    sys.exit(1)

# ==============================================================
# SECTION 1: TAG STRIPPER (shared utility)
# ==============================================================

# [FIX-12] Catches ANY bracket content anywhere in text
_ANY_BRACKET = re.compile(r'\[[^\]]*\]', re.IGNORECASE)

# Catches role prefixes at start of stored text
_ROLE_PREFIX = re.compile(
    r'^\s*\[(user|assistant|system|human|ai|oura)\]\s*',
    re.IGNORECASE
)


def strip_tags(text: str) -> str:
    """
    Remove ALL [bracket] content from text.
    Applied before storing AND after recalling.
    This is the nuclear option — no tag survives.
    """
    text = _ANY_BRACKET.sub('', text)
    text = _ROLE_PREFIX.sub('', text)
    return text.strip()


# ==============================================================
# SECTION 2: PER-SESSION MEMORY
# [FIX-11] Each session gets its own isolated OuraMemorySystem
# [FIX-19] Separate RLock per instance — no cross-session deadlock
# ==============================================================

class PerSessionMemory:
    """
    Singleton-per-session isolated Softwire memory.

    Key guarantees:
    - Zero cross-user contamination
    - Tags stripped at store AND recall
    - Auto-save every AUTO_SAVE_EVERY stores
    - Survives server restart (loads from {session_id}.npz)
    """

    _registry: Dict[str, 'PerSessionMemory'] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_session(cls, session_id: str) -> 'PerSessionMemory':
        """Thread-safe singleton factory."""
        with cls._registry_lock:
            if session_id not in cls._registry:
                cls._registry[session_id] = cls(session_id)
            return cls._registry[session_id]

    @classmethod
    def save_all(cls):
        """Called by atexit hook — saves every active session's memory."""
        with cls._registry_lock:
            instances = list(cls._registry.values())
        for inst in instances:
            inst.save()
        log.info("[MEMORY] All sessions saved (%d)", len(instances))

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._file_stem = str(MEMORY_DIR / session_id)
        self._lock = threading.RLock()   # [FIX-19] per-instance lock
        self._store_count = 0
        self._oms = OuraMemorySystem(
            pattern_length=PATTERN_LENGTH,
            g=SOFTWIRE_G,
            chunk_words=CHUNK_WORDS,
            overlap=OVERLAP_WORDS,
        )
        self._load()

    # ── persistence ──────────────────────────────────────────

    def _load(self):
        npz = Path(self._file_stem + ".npz")
        if npz.exists():
            try:
                self._oms.load(self._file_stem)
                count = self._pattern_count()
                log.info(
                    "[MEMORY:%s] Loaded %d patterns",
                    self.session_id[:8], count
                )
            except Exception as e:
                log.warning(
                    "[MEMORY:%s] Load failed, fresh start: %s",
                    self.session_id[:8], e
                )

    def save(self):
        try:
            with self._lock:
                self._oms.save(self._file_stem)
            log.debug("[MEMORY:%s] Saved", self.session_id[:8])
        except Exception as e:
            log.error("[MEMORY:%s] Save failed: %s", self.session_id[:8], e)

    # ── internal ─────────────────────────────────────────────

    def _pattern_count(self) -> int:
        """
        [FIX-16] text_encoder.py uses _patterns list, not n_patterns attr.
        We check both to be safe.
        """
        net = getattr(self._oms, 'network', None)
        if net is None:
            return 0
        # text_encoder SoftwireMemory uses self._patterns
        if hasattr(net, '_patterns'):
            return len(net._patterns)
        # fallback
        return int(getattr(net, 'n_patterns', 0))

    # ── public API ───────────────────────────────────────────

    def store(self, speaker: str, text: str):
        """
        [FIX-03] Calls store_text() directly — never store_conversation_turn().
        [FIX-12] Tags stripped before any bytes touch the matrix.
        """
        text = strip_tags(text)
        if not text:
            return
        with self._lock:
            try:
                self._oms.store_text(text)
                self._store_count += 1
                if self._store_count % AUTO_SAVE_EVERY == 0:
                    self._oms.save(self._file_stem)
                    log.debug(
                        "[MEMORY:%s] Auto-saved at store #%d",
                        self.session_id[:8], self._store_count
                    )
            except Exception as e:
                log.error("[MEMORY:%s] store() failed: %s", self.session_id[:8], e)

    def recall(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Returns {"text": str, "similarity": float} or None.
        Tags stripped from recalled text before returning.
        """
        query = strip_tags(query)
        if not query:
            return None
        with self._lock:
            try:
                result = self._oms.recall_from_text(query, noise_fraction=0.05)
                if (
                    result
                    and getattr(result, 'best_match_text', None)
                    and result.similarity >= RECALL_THRESHOLD
                ):
                    clean = strip_tags(result.best_match_text)
                    if clean:
                        return {
                            "text": clean,
                            "similarity": float(result.similarity)
                        }
            except Exception as e:
                log.error("[MEMORY:%s] recall() failed: %s", self.session_id[:8], e)
        return None

    def search(self, query: str, top_k: int = 2) -> List[Tuple[float, str]]:
        """
        Returns list of (similarity, clean_text) tuples above threshold.
        [FIX-18] Hardened against varied return types from search_similar.
        """
        query = strip_tags(query)
        if not query:
            return []
        results_out = []
        with self._lock:
            try:
                raw = self._oms.search_similar(
                    query,
                    top_k=top_k + 2,
                    threshold=RECALL_THRESHOLD
                )
                for item in raw:
                    # Handle (score, payload) where payload varies
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    score = float(item[0])
                    payload = item[1]
                    if hasattr(payload, 'best_match_text'):
                        text = payload.best_match_text or ''
                    elif hasattr(payload, 'text'):
                        text = payload.text or ''
                    else:
                        text = str(payload)
                    text = strip_tags(text)
                    if text:
                        results_out.append((score, text))
            except Exception as e:
                log.error("[MEMORY:%s] search() failed: %s", self.session_id[:8], e)
        return results_out[:top_k]

    def stats(self) -> Dict[str, Any]:
        return {
            "session_id":    self.session_id,
            "patterns":      self._pattern_count(),
            "total_stores":  self._store_count,
            "pattern_length": PATTERN_LENGTH,
            "g":             SOFTWIRE_G,
            "threshold":     RECALL_THRESHOLD,
        }


# ==============================================================
# SECTION 3: SESSION STORE
# [FIX-20] Individual {sid}.json files — no single giant file
# [FIX-08] RLock on all mutations
# ==============================================================

class SessionStore:
    """
    Thread-safe persistent session store.
    One JSON file per session in SESSIONS_DIR/{sid}.json.

    Schema:
      {
        "history":   [{"role": "user"|"assistant", "content": str}],
        "user_name": str | None,
        "facts":     [str],
        "created":   float,
        "updated":   float,
      }
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, dict] = {}

    def _path(self, sid: str) -> Path:
        return SESSIONS_DIR / f"{sid}.json"

    def _empty(self) -> dict:
        return {
            "history":   [],
            "user_name": None,
            "facts":     [],
            "created":   time.time(),
            "updated":   time.time(),
        }

    def get(self, sid: str) -> dict:
        """
        Returns a COPY of session data to prevent unsynchronized mutation.
        [FIX-09] No mutable dict leaks.
        """
        with self._lock:
            if sid not in self._cache:
                self._cache[sid] = self._load(sid)
            # Return a deep copy so callers cannot mutate internal state
            import copy
            return copy.deepcopy(self._cache[sid])

    def _load(self, sid: str) -> dict:
        path = self._path(sid)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "history" in data:
                    log.debug("[SESSION:%s] Loaded from disk", sid[:8])
                    return data
            except Exception as e:
                log.warning("[SESSION:%s] Load failed: %s", sid[:8], e)
        return self._empty()

    def _save(self, sid: str, data: dict):
        """Atomic write via temp file."""
        try:
            path = self._path(sid)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            tmp.replace(path)
        except Exception as e:
            log.error("[SESSION:%s] Save failed: %s", sid[:8], e)

    def append_turn(self, sid: str, role: str, content: str):
        with self._lock:
            if sid not in self._cache:
                self._cache[sid] = self._load(sid)
            data = self._cache[sid]
            data["history"].append({"role": role, "content": content})
            if len(data["history"]) > HISTORY_MAX_TURNS:
                data["history"] = data["history"][-HISTORY_MAX_TURNS:]
            data["updated"] = time.time()
            self._save(sid, data)

    def update_info(self, sid: str, **kwargs):
        with self._lock:
            if sid not in self._cache:
                self._cache[sid] = self._load(sid)
            self._cache[sid].update(kwargs)
            self._cache[sid]["updated"] = time.time()
            self._save(sid, self._cache[sid])

    def count(self) -> int:
        # Count files on disk for accuracy
        return len(list(SESSIONS_DIR.glob("*.json")))

    def save_all(self):
        """Called by atexit hook."""
        with self._lock:
            for sid, data in self._cache.items():
                self._save(sid, data)
        log.info("[SESSION] All sessions flushed to disk")


# ==============================================================
# SECTION 4: ECHO GUARD — NUCLEAR MULTI-PASS
# [FIX-04] re.DOTALL added, [FIX-21] nuclear bracket strip as final pass
# [FIX-13] Catches [Memory — xx%] format
# ==============================================================

class EchoGuard:
    """
    Multi-pass recursive decontaminator.

    Pass order:
      1. Strip ALL [bracket] content (nuclear)
      2. Remove "I remember something related" trigger phrases
      3. Remove system prompt fragment echoes
      4. Remove wrapping quotes
      5. Deduplicate sentences
      6. Verify no brackets survived (re-run if needed)
    """

    # [FIX-04] re.DOTALL ensures multiline echoes are caught
    _ECHO_TRIGGERS = re.compile(
        r"(I remember something related|"
        r"I recall something|"
        r"That reminds me of something we'?ve discussed|"
        r"Memory —\s*\d+%|"
        r"Relevant past memory|"
        r"Related context\s*[:]\s*|"
        r"Known (?:about user|facts)\s*[:]\s*)"
        r"[^\n.!?]*",
        re.IGNORECASE | re.DOTALL
    )

    _SYSTEM_ECHOES = re.compile(
        r"(You are OURA|"
        r"Speak naturally like|"
        r"Do NOT say|"
        r"Be concise unless|"
        r"Weave any recalled|"
        r"Never use brackets|"
        r"Never say you 'remember')"
        r"[^\n]*",
        re.IGNORECASE
    )

    _WRAPPING = re.compile(r'^[\s"\']+|[\s"\']+$')
    _SENTENCE = re.compile(r'(?<=[.!?])\s+')
    _MULTI_NL = re.compile(r'\n{3,}')

    @classmethod
    def clean(cls, text: str, max_passes: int = 6) -> str:
        if not text:
            return "I'm here. What's on your mind?"

        prev = None
        passes = 0
        while prev != text and passes < max_passes:
            prev = text
            text = cls._single_pass(text)
            passes += 1

        if passes > 1:
            log.debug("[ECHOGUARD] Stabilized after %d passes", passes)

        text = text.strip()
        return text or "I'm here. What's on your mind?"

    @classmethod
    def _single_pass(cls, text: str) -> str:
        # Pass 1: Nuclear bracket removal [FIX-21]
        text = _ANY_BRACKET.sub('', text)

        # Pass 2: Echo trigger phrases
        text = cls._ECHO_TRIGGERS.sub('', text)

        # Pass 3: System prompt fragments
        text = cls._SYSTEM_ECHOES.sub('', text)

        # Pass 4: Wrapping quotes/whitespace
        text = cls._WRAPPING.sub('', text)

        # Pass 5: Deduplicate sentences
        sentences = cls._SENTENCE.split(text)
        seen: Dict[str, int] = {}
        deduped = []
        for s in sentences:
            key = re.sub(r'\s+', ' ', s.strip().lower())[:120]
            if not key:
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 1:
                deduped.append(s.strip())
        text = ' '.join(deduped)

        # Pass 6: Collapse excessive newlines
        text = cls._MULTI_NL.sub('\n\n', text)

        return text


# ==============================================================
# SECTION 5: CONTEXT BUILDER
# [FIX-13] NO [Memory — xx%] brackets in system prompt ever
# [FIX-14] Natural phrasing only
# ==============================================================

class ContextBuilder:
    """
    Builds clean OpenAI-compatible messages array.
    Zero bracket tags. Zero echo triggers. Natural language only.
    """

    _NAME_RE = re.compile(
        r"\bmy name is ([A-Za-z][a-zA-Z'\-]{0,24})",
        re.IGNORECASE
    )
    _FACT_RES = [
        re.compile(r"\bI(?:'m| am) from ([A-Za-z ]{2,30})", re.IGNORECASE),
        re.compile(r"\bI(?:'m| am) (\d{1,3}) years old", re.IGNORECASE),
        re.compile(r"\bI work (?:as |at )?(.{5,40}?)(?:\.|,|$)", re.IGNORECASE),
    ]

    # [FIX-13][FIX-14] — no brackets, no percentages, no structured tags
    SYSTEM_BASE = (
        "You are OURA, a warm, brilliant AI with genuine persistent memory. "
        "You remember everything about the people you talk to. "
        "Weave recalled memories into your responses naturally and seamlessly — "
        "like a real person who never forgets a friend. "
        "Never use brackets, tags, percentages, or structured formatting. "
        "Never start a sentence with 'I remember something related'. "
        "Just talk. Be warm. Be real."
    )

    @classmethod
    def extract_and_update(
        cls,
        user_message: str,
        sid: str,
        sessions: 'SessionStore'
    ) -> dict:
        """
        Extract name/facts from message and persist them.
        Returns current session data dict (fresh copy).
        """
        session_data = sessions.get(sid)
        updates = {}

        m = cls._NAME_RE.search(user_message)
        if m:
            name = m.group(1).strip().capitalize()
            if name != session_data.get("user_name"):
                updates["user_name"] = name
                log.info("[CONTEXT:%s] Learned name: %s", sid[:8], name)

        new_facts = list(session_data.get("facts", []))
        for regex in cls._FACT_RES:
            fm = regex.search(user_message)
            if fm:
                fact = user_message[:140].strip()
                if fact not in new_facts:
                    new_facts.append(fact)
                    if len(new_facts) > 20:
                        new_facts.pop(0)

        if new_facts != session_data.get("facts", []):
            updates["facts"] = new_facts

        if updates:
            sessions.update_info(sid, **updates)
            session_data = sessions.get(sid)  # fresh copy with updates

        return session_data

    @classmethod
    def build_messages(
        cls,
        user_message: str,
        session_data: dict,
        memory: PerSessionMemory,
    ) -> List[Dict[str, str]]:
        """
        Build full OpenAI messages array.
        Memory is injected as natural language in system prompt only.
        [FIX-13] Absolutely no [bracket] format used.
        """
        system_parts = [cls.SYSTEM_BASE]

        name = session_data.get("user_name")
        if name:
            system_parts.append(f"The person you're talking to is {name}.")

        # Natural memory injection — no brackets, no percentages
        recalled = memory.recall(user_message)
        if recalled:
            # Clean the recalled text one more time for safety
            clean_memory = strip_tags(recalled["text"])[:350]
            if clean_memory:
                system_parts.append(
                    f"You remember this from a previous conversation: {clean_memory}"
                )

        similar = memory.search(user_message, top_k=2)
        if similar:
            snippets = [
                strip_tags(txt)[:140]
                for _, txt in similar
                if txt and txt != recalled.get("text", "") if recalled else True
            ]
            snippets = [s for s in snippets if s]
            if snippets:
                system_parts.append(
                    "You also remember: " + " — ".join(snippets)
                )

        facts = session_data.get("facts", [])
        if facts:
            system_parts.append(
                "Things you know about them: " + "; ".join(facts[-3:])
            )

        system_prompt = "\n\n".join(system_parts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Recent conversation history
        for turn in session_data.get("history", [])[-CONTEXT_TURNS:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages


# ==============================================================
# SECTION 6: GATEWAY CLIENT
# [FIX-01] OpenAI-compatible endpoint first
# [FIX-02] Ephemeral session_id
# [FIX-10] Circuit breaker
# ==============================================================

class GatewayClient:
    """
    Sends full OpenAI-compatible messages array.
    Circuit breaker prevents hammering a dead gateway.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._failures   = 0
        self._open_until = 0.0
        self._healthy: Optional[bool] = None

    def _is_open(self) -> bool:
        with self._lock:
            return self._open_until > 0 and time.time() < self._open_until

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
                    "[GATEWAY] Circuit breaker OPEN for %ds", CB_RECOVERY_SECS
                )

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{GATEWAY_URL}/health", timeout=4)
            self._healthy = (r.status_code == 200)
        except Exception:
            self._healthy = False
        log.info(
            "[GATEWAY] %s at %s",
            "HEALTHY" if self._healthy else "UNREACHABLE",
            GATEWAY_URL
        )
        return bool(self._healthy)

    def send(
        self,
        messages: List[Dict[str, str]],
        timeout: int = 45,
    ) -> Tuple[Optional[str], str]:
        """
        Returns (text, provider) or (None, "").
        Tries /v1/chat/completions first, then /chat.
        """
        if self._is_open():
            return None, ""

        # Extract last user message for legacy endpoint
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m["content"]
                break

        base = GATEWAY_URL.rstrip('/')
        eph_sid = f"eph-{uuid.uuid4()}"

        endpoints = [
            {
                "url": f"{base}/v1/chat/completions",
                "json": {
                    "model":       "gpt-4o-mini",
                    "messages":    messages,
                    "temperature": 0.78,
                    "max_tokens":  1200,
                },
                "parse": self._parse_openai,
            },
            {
                "url": f"{base}/chat",
                "json": {
                    "message":    user_msg,
                    "session_id": eph_sid,
                    "messages":   messages,
                },
                "parse": self._parse_native,
            },
        ]

        for ep in endpoints:
            try:
                log.debug("[GATEWAY] POST %s", ep["url"])
                resp = requests.post(
                    ep["url"],
                    json=ep["json"],
                    headers={
                        "Content-Type":  "application/json",
                        "Authorization": f"Bearer {API_KEY}",
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    text, provider = ep["parse"](resp.json())
                    if text:
                        self._record_success()
                        return text.strip(), provider
                else:
                    log.warning("[GATEWAY] HTTP %d from %s", resp.status_code, ep["url"])

            except requests.exceptions.Timeout:
                log.warning("[GATEWAY] Timeout: %s", ep["url"])
                self._record_failure()

            except requests.exceptions.ConnectionError:
                log.warning("[GATEWAY] Connection refused: %s", ep["url"])
                self._record_failure()
                break   # Same host — pointless to retry other endpoint

            except Exception as exc:
                log.error("[GATEWAY] Error on %s: %s", ep["url"], exc)
                self._record_failure()

        return None, ""

    @staticmethod
    def _parse_openai(data: dict) -> Tuple[Optional[str], str]:
        try:
            text = data["choices"][0]["message"]["content"]
            return text.strip(), data.get("model", "openai")
        except (KeyError, IndexError, TypeError):
            return None, ""

    @staticmethod
    def _parse_native(data: dict) -> Tuple[Optional[str], str]:
        text = (
            data.get("text")
            or data.get("response")
            or data.get("message")
            or data.get("content")
        )
        provider = data.get("provider", "gateway-native")
        if text:
            return str(text).strip(), provider
        return None, ""


# ==============================================================
# SECTION 7: FALLBACK ENGINE
# [FIX-15] Natural phrasing — no echo triggers
# ==============================================================

class FallbackEngine:
    """
    Rule-based response when gateway is unavailable.
    Uses only pre-processed recalled text.
    Zero echo triggers. Zero speaker tags.
    """

    @staticmethod
    def respond(
        user_message: str,
        session_data: dict,
        recalled: Optional[Dict[str, Any]],
        pattern_count: int,
    ) -> str:
        name     = session_data.get("user_name", "")
        greeting = f" {name}" if name else ""
        lower    = user_message.lower()

        # [FIX-15] Natural phrasing — no "That reminds me of something we've discussed"
        if recalled and recalled["similarity"] > 0.75:
            snippet = strip_tags(recalled["text"])[:180]
            return (
                f"You know{greeting}, that actually connects to something "
                f"you shared before — {snippet}. "
                f"Is that what you're thinking about?"
            )

        if "my name is" in lower:
            return (
                f"Good to know{greeting}! "
                "I've got that and I'll keep it for every conversation we have."
            )

        if any(w in lower for w in ["pickup line", "chat up", "flirt"]):
            import random
            lines = [
                "Are you a Wi-Fi signal? Because I feel a strong connection.",
                "No cap, you just broke my algorithm.",
                "You must be trending, because you're all I see.",
                "Are you my charger? Because I die without you.",
                "Did it hurt when you fell from the For You page?",
                "I'd say you're like a dream, but I don't want to wake up.",
            ]
            chosen = random.sample(lines, min(4, len(lines)))
            numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(chosen))
            return f"Here you go{greeting}:\n\n{numbered}\n\nWant more? 😄"

        if any(w in lower for w in ["remember", "recall", "forgot", "memory"]):
            return (
                f"I have {pattern_count} memories{greeting}. "
                "Ask me about something specific."
            )

        return (
            f"I'm all ears{greeting}. What's on your mind?"
        )


# ==============================================================
# SECTION 8: PERIODIC SAVER THREAD
# ==============================================================

class PeriodicSaver(threading.Thread):
    """Saves all memories + sessions every 5 minutes in background."""

    def __init__(self, sessions: 'SessionStore'):
        super().__init__(daemon=True, name="PeriodicSaver")
        self._sessions = sessions
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(timeout=300):
            self._sessions.save_all()
            PerSessionMemory.save_all()
            log.debug("[SAVER] Periodic save complete")

    def stop(self):
        self._stop.set()


# ==============================================================
# SECTION 9: INITIALIZE SINGLETONS
# ==============================================================

_sessions = SessionStore()
_gateway  = GatewayClient()
_fallback = FallbackEngine()

# ==============================================================
# SECTION 10: ATEXIT SHUTDOWN HOOK
# [FIX-17] Memory + sessions saved on Ctrl+C or crash
# ==============================================================

def _shutdown():
    log.info("[SHUTDOWN] Saving all data before exit...")
    _sessions.save_all()
    PerSessionMemory.save_all()
    log.info("[SHUTDOWN] Done. Goodbye.")

atexit.register(_shutdown)

# ==============================================================
# SECTION 11: FLASK APPLICATION
# ==============================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.before_request
def _start_timer():
    flask_g.t_start = time.perf_counter()


@app.after_request
def _add_headers(response):
    ms = int((time.perf_counter() - flask_g.t_start) * 1000)
    response.headers["X-Response-Time"] = f"{ms}ms"
    response.headers["X-Engine"]        = "oura-softwire-3.0.2"
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

    log.info("[CHAT] sid=%s msg=%r", session_id[:8], user_message[:80])

    # 1. Per-session isolated memory
    memory = PerSessionMemory.for_session(session_id)

    # 2. Extract facts + get fresh session data
    #    [FIX-08] All mutations go through SessionStore methods
    session_data = ContextBuilder.extract_and_update(
        user_message, session_id, _sessions
    )

    # 3. Store user message (tag-free)
    memory.store("user", user_message)

    # 4. Add user turn to history
    _sessions.append_turn(session_id, "user", user_message)

    # 5. Build full context (memory already queried inside ContextBuilder)
    messages = ContextBuilder.build_messages(
        user_message=user_message,
        session_data=session_data,
        memory=memory,
    )

    # 6. Call gateway
    response_text, provider = _gateway.send(messages, timeout=45)

    # 7. Fallback if gateway down
    used_fallback = False
    if not response_text:
        recalled = memory.recall(user_message)
        response_text = FallbackEngine.respond(
            user_message  = user_message,
            session_data  = session_data,
            recalled      = recalled,
            pattern_count = memory.stats()["patterns"],
        )
        provider      = "fallback"
        used_fallback = True

    # 8. Nuclear echo decontamination
    response_text = EchoGuard.clean(response_text)

    # 9. Store assistant reply (tag-free)
    memory.store("assistant", response_text)
    _sessions.append_turn(session_id, "assistant", response_text)

    stats = memory.stats()
    log.info(
        "[CHAT] Done — provider=%s fallback=%s patterns=%d",
        provider, used_fallback, stats["patterns"]
    )

    return jsonify({
        "text":        response_text,
        "session_id":  session_id,
        "provider":    provider,
        "patterns":    stats["patterns"],
        "user_name":   session_data.get("user_name"),
        "used_fallback": used_fallback,
    })


# ── /health ───────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "version":         "3.0.2",
        "sessions_on_disk": _sessions.count(),
        "gateway_healthy": _gateway._healthy,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    })


# ── /api/status ───────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({
        "status":    "operational",
        "version":   "3.0.2",
        "sessions":  _sessions.count(),
        "gateway":   {
            "url":       GATEWAY_URL,
            "healthy":   _gateway._healthy,
            "failures":  _gateway._failures,
            "open_until": _gateway._open_until,
            "is_open":   _gateway._is_open(),
        },
        "memory": {
            "active_sessions": len(PerSessionMemory._registry),
            "pattern_length":  PATTERN_LENGTH,
            "g":               SOFTWIRE_G,
            "threshold":       RECALL_THRESHOLD,
            "auto_save_every": AUTO_SAVE_EVERY,
        },
    })


# ── /api/memory/stats ─────────────────────────────────────────

@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    sid = request.args.get("session_id", "")
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    memory = PerSessionMemory.for_session(sid)
    return jsonify(memory.stats())


# ── /api/memory/save ──────────────────────────────────────────

@app.route("/api/memory/save", methods=["POST"])
def memory_save():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    _sessions.save_all()
    PerSessionMemory.save_all()
    return jsonify({"status": "saved"})


# ── /api/memory/recall ────────────────────────────────────────

@app.route("/api/memory/recall", methods=["POST"])
def manual_recall():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    data  = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    sid   = str(data.get("session_id", "")).strip()
    if not query or not sid:
        return jsonify({"error": "query and session_id required"}), 400
    memory   = PerSessionMemory.for_session(sid)
    recalled = memory.recall(query)
    similar  = memory.search(query, top_k=3)
    return jsonify({
        "recalled": recalled,
        "similar":  [{"similarity": s, "text": t} for s, t in similar],
    })


# ── /api/session/<id> ─────────────────────────────────────────

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    s = _sessions.get(session_id)
    return jsonify({
        "session_id": session_id,
        "user_name":  s.get("user_name"),
        "facts":      s.get("facts", []),
        "turns":      len(s.get("history", [])),
        "created":    s.get("created"),
        "updated":    s.get("updated"),
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


# ==============================================================
# SECTION 12: STARTUP
# ==============================================================

def startup():
    log.info("=" * 64)
    log.info("  OURA SOFTWIRE v3.0.2 — ETERNAL PRIVATE MEMORY EDITION")
    log.info("=" * 64)
    log.info("  Memory dir:    %s", MEMORY_DIR)
    log.info("  Sessions dir:  %s", SESSIONS_DIR)
    log.info("  Gateway URL:   %s", GATEWAY_URL)
    log.info("  Port:          %d", PORT)
    log.info(
        "  N=%d | g=%.1f | threshold=%.2f | auto_save=%d",
        PATTERN_LENGTH, SOFTWIRE_G, RECALL_THRESHOLD, AUTO_SAVE_EVERY
    )
    log.info("  Sessions on disk: %d", _sessions.count())

    _gateway.health_check()

    saver = PeriodicSaver(_sessions)
    saver.start()
    log.info("  Periodic saver started (every 5 min)")

    log.info("=" * 64)
    log.info("  Ready at http://0.0.0.0:%d", PORT)
    log.info("=" * 64)


# ==============================================================
# SECTION 13: ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    startup()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
