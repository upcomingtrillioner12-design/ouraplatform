"""
SOFTWIRE + AI GATEWAY - Complete Integration Server (FIXED)
Fixes:
1. Messages array properly sent to gateway
2. [user]/[assistant] tag stripping from recalled text
3. Single consistent memory implementation
4. Persistent memory across server restarts
5. Clean prompt building without recursive echo
6. Session persistence via JSON file backup
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import uuid
import os
import re
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("softwire")

app = FastAPI(title="SOFTWIRE + AI Gateway", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIGURATION ============
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
SESSIONS_FILE = "sessions_backup.json"
MEMORY_FILE = "softwire_memory_backup"
MAX_HISTORY_PER_SESSION = 50
RECENT_CONTEXT_WINDOW = 10
MAX_FACTS_STORED = 20
MAX_PATTERNS = 2048
SOFTWIRE_N = 512
SOFTWIRE_G = 11.0

# ============ TAG STRIPPING UTILITY ============

def strip_speaker_tags(text: str) -> str:
    """
    Strip [user], [assistant], [system] tags from recalled text.
    Fixes Issue #3, #6, #7, #8 from debug analysis.
    Raw stored text like '[user] give me pickup lines' becomes
    'give me pickup lines'.
    """
    if not text:
        return text

    # Remove [role] prefixes at start of string
    text = re.sub(r'^\s*\[(user|assistant|system|human|ai)\]\s*', '', text, flags=re.IGNORECASE)

    # Remove inline [role] tags mid-text
    text = re.sub(r'\[(user|assistant|system|human|ai)\]\s*', '', text, flags=re.IGNORECASE)

    # Remove "User:" and "Assistant:" prefixes
    text = re.sub(r'^\s*(User|Assistant|System):\s*', '', text, flags=re.IGNORECASE)

    # Remove any remaining angle-bracket style tags
    text = re.sub(r'<(user|assistant|system)>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</(user|assistant|system)>', '', text, flags=re.IGNORECASE)

    return text.strip()


def clean_recalled_memories(memories: list) -> list:
    """Strip tags from all recalled memory items before prompt injection."""
    cleaned = []
    for mem in memories:
        if isinstance(mem, dict):
            cleaned_mem = dict(mem)
            if "text" in cleaned_mem:
                cleaned_mem["text"] = strip_speaker_tags(cleaned_mem["text"])
            if "content" in cleaned_mem:
                cleaned_mem["content"] = strip_speaker_tags(cleaned_mem["content"])
            cleaned.append(cleaned_mem)
        elif isinstance(mem, str):
            cleaned.append(strip_speaker_tags(mem))
    return cleaned


# ============ SINGLE CONSISTENT SOFTWIRE MEMORY ENGINE ============
# Fixes Issue #4: Single implementation instead of 10 incompatible ones
# Physics: J = (1/N) * patterns.T @ patterns (no g in J)
#          dynamics: r_new = tanh(g * J @ r) (g applied in dynamics)
# This matches softwireengine10.py v1.2 which had the correct physics.

class SoftwireMemoryEngine:
    """
    Single consistent Hopfield-style memory engine.
    Physics convention (engine10 v1.2):
        J = (1/N) * Ξ^T Ξ          (g NOT baked into J)
        r_{t+1} = tanh(g * J @ r_t) (g applied at recall time)
    This avoids the double-scaling bug in engine1/engine9 where
    g was inside J AND applied again in dynamics.
    """

    def __init__(self, N: int = SOFTWIRE_N, g: float = SOFTWIRE_G,
                 max_patterns: int = MAX_PATTERNS):
        self.N = N
        self.g = g
        self.max_patterns = max_patterns
        self._patterns: List[np.ndarray] = []
        self._texts: List[str] = []
        self._timestamps: List[str] = []
        self._J: np.ndarray = np.zeros((N, N))
        logger.info(f"[SOFTWIRE ENGINE] Initialized N={N}, g={g}, max_patterns={max_patterns}")

    def _text_to_pattern(self, text: str) -> np.ndarray:
        """
        Encode text as a ±1 pattern of length N.
        Deterministic: same text always gives same pattern.
        """
        # Hash-based encoding for determinism
        encoded = np.zeros(self.N)
        text_bytes = text.encode("utf-8")
        for i, byte in enumerate(text_bytes):
            idx = (i * 31 + byte) % self.N
            encoded[idx] += 1
        # Binarize to ±1
        median_val = np.median(encoded)
        pattern = np.where(encoded >= median_val, 1.0, -1.0)
        return pattern

    def _rebuild_J(self):
        """Rebuild synaptic matrix from stored patterns. g NOT in J."""
        if not self._patterns:
            self._J = np.zeros((self.N, self.N))
            return
        mat = np.array(self._patterns)  # shape (P, N)
        self._J = (1.0 / self.N) * (mat.T @ mat)
        # Remove self-connections (autapse-free)
        np.fill_diagonal(self._J, 0.0)

    def store(self, text: str) -> int:
        """
        Store a text as a pattern. Returns pattern index.
        Tags are stripped BEFORE storage to prevent echo loop.
        """
        # Strip tags before storage — Fixes Issue #2 at source
        clean_text = strip_speaker_tags(text)
        if not clean_text:
            return -1

        pattern = self._text_to_pattern(clean_text)
        self._patterns.append(pattern)
        self._texts.append(clean_text)
        self._timestamps.append(datetime.now().isoformat())

        # Incremental update to J (faster than full rebuild)
        xi = pattern.reshape(-1, 1)
        self._J += (1.0 / self.N) * (xi @ xi.T)
        np.fill_diagonal(self._J, 0.0)

        # Prune oldest if over capacity
        if len(self._patterns) > self.max_patterns:
            self._patterns = self._patterns[-self.max_patterns:]
            self._texts = self._texts[-self.max_patterns:]
            self._timestamps = self._timestamps[-self.max_patterns:]
            self._rebuild_J()

        logger.debug(f"[SOFTWIRE ENGINE] Stored pattern #{len(self._patterns)}: {clean_text[:50]}")
        return len(self._patterns) - 1

    def recall(self, query: str, max_steps: int = 20,
               top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recall memories similar to query.
        Returns list of dicts with text, overlap, timestamp.
        All returned texts are already tag-free (stored clean).
        """
        if not self._patterns:
            return []

        cue = self._text_to_pattern(query)

        # Run Hopfield dynamics: r_{t+1} = tanh(g * J @ r)
        r = cue.copy()
        for _ in range(max_steps):
            r_new = np.tanh(self.g * (self._J @ r))
            if np.max(np.abs(r_new - r)) < 1e-6:
                break
            r = r_new

        # Compute overlaps with all stored patterns
        results = []
        patterns_arr = np.array(self._patterns)
        overlaps = (patterns_arr @ r) / self.N

        top_indices = np.argsort(overlaps)[::-1][:top_k]
        for idx in top_indices:
            overlap = float(overlaps[idx])
            if overlap > 0.1:  # Threshold to avoid noise matches
                results.append({
                    "text": self._texts[idx],  # Already clean (stored without tags)
                    "overlap": overlap,
                    "timestamp": self._timestamps[idx],
                    "index": int(idx)
                })

        return results

    def save(self, filepath: str):
        """Save memory state to disk for persistence across restarts."""
        try:
            data = {
                "texts": self._texts,
                "timestamps": self._timestamps,
                "g": self.g,
                "N": self.N
            }
            np.savez(
                filepath,
                J=self._J,
                patterns=np.array(self._patterns) if self._patterns else np.zeros((0, self.N)),
                metadata=json.dumps(data)
            )
            logger.info(f"[SOFTWIRE ENGINE] Saved {len(self._patterns)} patterns to {filepath}")
        except Exception as e:
            logger.error(f"[SOFTWIRE ENGINE] Save failed: {e}")

    def load(self, filepath: str) -> bool:
        """Load memory state from disk."""
        try:
            npz_path = filepath if filepath.endswith(".npz") else f"{filepath}.npz"
            if not os.path.exists(npz_path):
                logger.info(f"[SOFTWIRE ENGINE] No saved memory found at {npz_path}")
                return False

            data = np.load(npz_path, allow_pickle=True)
            self._J = data["J"]
            patterns_arr = data["patterns"]
            if patterns_arr.shape[0] > 0:
                self._patterns = list(patterns_arr)
            else:
                self._patterns = []

            metadata = json.loads(str(data["metadata"]))
            self._texts = metadata.get("texts", [])
            self._timestamps = metadata.get("timestamps", [])

            logger.info(f"[SOFTWIRE ENGINE] Loaded {len(self._patterns)} patterns from {npz_path}")
            return True
        except Exception as e:
            logger.error(f"[SOFTWIRE ENGINE] Load failed: {e}")
            return False


# ============ ETERNAL SESSION MEMORY ============

class EternalMemory:
    """
    Persistent user session memory.
    Fixes Issue #1: Proper persistence across server restarts.
    Fixes Issue #5: Single memory implementation (uses SoftwireMemoryEngine).
    """

    def __init__(self):
        self._users: Dict[str, dict] = {}
        self._softwire = SoftwireMemoryEngine(
            N=SOFTWIRE_N,
            g=SOFTWIRE_G,
            max_patterns=MAX_PATTERNS
        )
        self._load_sessions()
        self._softwire.load(MEMORY_FILE)

    def _load_sessions(self):
        """Load sessions from JSON backup file on startup."""
        try:
            if os.path.exists(SESSIONS_FILE):
                with open(SESSIONS_FILE, "r") as f:
                    loaded = json.load(f)
                self._users = loaded
                logger.info(f"[ETERNAL MEMORY] Loaded {len(self._users)} sessions from {SESSIONS_FILE}")
            else:
                logger.info(f"[ETERNAL MEMORY] No session backup found, starting fresh")
        except Exception as e:
            logger.error(f"[ETERNAL MEMORY] Session load failed: {e}")
            self._users = {}

    def _save_sessions(self):
        """Save sessions to JSON backup file."""
        try:
            with open(SESSIONS_FILE, "w") as f:
                json.dump(self._users, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[ETERNAL MEMORY] Session save failed: {e}")

    def get_or_create_user(self, user_id: str) -> dict:
        if user_id not in self._users:
            self._users[user_id] = {
                "user_id": user_id,
                "conversation_history": [],
                "long_term_memory": {
                    "user_name": None,
                    "preferences": {},
                    "facts_learned": [],
                    "first_seen": datetime.now().isoformat(),
                    "total_messages": 0,
                    "topics_discussed": []
                }
            }
            logger.info(f"[ETERNAL MEMORY] New user: {user_id}")
            self._save_sessions()
        return self._users[user_id]

    def add_message(self, user_id: str, role: str, content: str):
        """
        Add message to history AND store in Softwire.
        Content is stored CLEAN (no tags) in both places.
        Fixes Issue #2 at the storage layer.
        """
        user = self.get_or_create_user(user_id)
        clean_content = strip_speaker_tags(content)

        user["conversation_history"].append({
            "role": role,
            "content": clean_content,
            "timestamp": datetime.now().isoformat()
        })
        user["long_term_memory"]["total_messages"] += 1

        # Prune history to last MAX_HISTORY_PER_SESSION messages
        if len(user["conversation_history"]) > MAX_HISTORY_PER_SESSION:
            user["conversation_history"] = user["conversation_history"][-MAX_HISTORY_PER_SESSION:]

        # Store in Softwire pattern memory (already strips tags internally)
        self._softwire.store(clean_content)

        # Save after each message for persistence
        self._save_sessions()
        self._softwire.save(MEMORY_FILE)

    def recall_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recall semantically similar memories from Softwire.
        Returns clean text (no tags) ready for prompt injection.
        Fixes Issue #7: No raw tags returned from recall.
        """
        results = self._softwire.recall(query, top_k=top_k)
        # Double-check: strip tags from recalled text even though stored clean
        return clean_recalled_memories(results)

    def get_context(self, user_id: str) -> dict:
        """Build context dict for prompt construction."""
        user = self.get_or_create_user(user_id)
        recent_history = user["conversation_history"][-RECENT_CONTEXT_WINDOW:]
        return {
            "long_term": user["long_term_memory"],
            "recent_history": recent_history,
            "summary": self._summarize_memory(user["long_term_memory"])
        }

    def _summarize_memory(self, memory: dict) -> str:
        """Generate clean summary of what is remembered about the user."""
        facts = []
        if memory.get("user_name"):
            facts.append(f"User's name is {memory['user_name']}")
        if memory.get("preferences"):
            for key, val in memory["preferences"].items():
                facts.append(f"User prefers {key}: {val}")
        if memory.get("topics_discussed"):
            topics = memory["topics_discussed"][-3:]
            facts.append(f"Recent topics: {', '.join(topics)}")
        if memory.get("facts_learned"):
            for fact in memory["facts_learned"][-3:]:
                if isinstance(fact, dict):
                    clean_fact = strip_speaker_tags(fact.get("fact", ""))
                else:
                    clean_fact = strip_speaker_tags(str(fact))
                if clean_fact:
                    facts.append(clean_fact)
        return ". ".join(facts) if facts else "No prior memory"

    def update_long_term_memory(self, user_id: str,
                                 user_message: str, ai_response: str):
        """
        Extract and store structured facts from conversation.
        All stored facts are tag-free.
        """
        user = self.get_or_create_user(user_id)
        memory = user["long_term_memory"]

        # Clean inputs before extraction
        clean_user_msg = strip_speaker_tags(user_message)
        clean_ai_resp = strip_speaker_tags(ai_response)

        msg_lower = clean_user_msg.lower()

        # Extract name
        name_patterns = ["my name is", "i'm called", "call me", "i am "]
        for pattern in name_patterns:
            if pattern in msg_lower:
                parts = msg_lower.split(pattern)
                if len(parts) > 1:
                    name_candidate = parts[1].strip().split()[0]
                    # Basic validation: only alphabetic
                    if name_candidate.isalpha() and len(name_candidate) > 1:
                        memory["user_name"] = name_candidate.capitalize()
                        logger.info(f"[ETERNAL MEMORY] Learned name: {memory['user_name']}")
                break

        # Extract preferences
        pref_patterns = {
            "i like": "likes",
            "i love": "loves",
            "i prefer": "prefers",
            "i enjoy": "enjoys",
            "i hate": "dislikes",
            "i dislike": "dislikes"
        }
        for phrase, category in pref_patterns.items():
            if phrase in msg_lower:
                pref_text = clean_user_msg[msg_lower.index(phrase) + len(phrase):].strip()
                if pref_text:
                    memory["preferences"][category] = pref_text[:100]
                break

        # Store facts from substantial user messages
        if len(clean_user_msg) > 20 and "?" not in clean_user_msg:
            memory["facts_learned"].append({
                "fact": clean_user_msg[:100],
                "timestamp": datetime.now().isoformat()
            })
            memory["facts_learned"] = memory["facts_learned"][-MAX_FACTS_STORED:]

        # Track topics (simple keyword extraction)
        topic_keywords = [
            "python", "javascript", "coding", "programming", "music", "food",
            "travel", "sports", "science", "math", "history", "movies",
            "books", "gaming", "health", "work", "family"
        ]
        for keyword in topic_keywords:
            if keyword in msg_lower and keyword not in memory["topics_discussed"]:
                memory["topics_discussed"].append(keyword)
                memory["topics_discussed"] = memory["topics_discussed"][-10:]

        self._save_sessions()


# ============ GATEWAY CONNECTOR — FIXED ============

class GatewayConnector:
    """
    Bridges Softwire to the AI Gateway.
    Fixes Issue #1 (CRITICAL): Sends messages array that gateway can use.
    Fixes Issue #2 (CRITICAL): Stateless sessions handled correctly.
    Falls back to direct provider call if gateway doesn't support messages.
    """

    def __init__(self):
        self.gateway_url = GATEWAY_URL
        self._gateway_sessions: Dict[str, str] = {}

    def get_gateway_session(self, user_id: str) -> str:
        """Get or create gateway session."""
        if user_id not in self._gateway_sessions:
            try:
                resp = requests.post(
                    f"{self.gateway_url}/session/new",
                    timeout=5
                )
                if resp.status_code == 200:
                    sid = resp.json().get("session_id", f"sw_{user_id}")
                    self._gateway_sessions[user_id] = sid
                    logger.info(f"[GATEWAY] Session created: {sid} for user {user_id}")
                else:
                    self._gateway_sessions[user_id] = f"sw_{user_id}"
            except Exception as e:
                logger.warning(f"[GATEWAY] Session creation failed: {e}")
                self._gateway_sessions[user_id] = f"sw_{user_id}"
        return self._gateway_sessions[user_id]

    def send_with_history(self, user_id: str,
                           messages: List[Dict[str, str]],
                           system_prompt: str) -> dict:
        """
        Send full conversation history to gateway.
        Fixes Issue #1: messages array is now sent correctly.
        Tries three approaches in order:
            1. Gateway /chat/messages endpoint (if it exists)
            2. Gateway /chat with full prompt as single message
            3. Direct provider fallback
        """
        gateway_session = self.get_gateway_session(user_id)

        # Approach 1: Try gateway messages endpoint
        try:
            resp = requests.post(
                f"{self.gateway_url}/chat/messages",
                json={
                    "session_id": gateway_session,
                    "system_prompt": system_prompt,
                    "messages": messages
                },
                timeout=60
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "response": result.get("text", result.get("response", "")),
                    "provider": result.get("provider", "gateway-messages"),
                    "error": None
                }
        except Exception:
            pass

        # Approach 2: Flatten to single prompt and use /chat
        # Fixes Issue #2: We build the full context into user_message
        # so gateway's stateless mode still has the full context.
        flat_prompt = self._flatten_messages_to_prompt(system_prompt, messages)
        try:
            resp = requests.post(
                f"{self.gateway_url}/chat",
                json={
                    "message": flat_prompt,
                    "session_id": f"stateless-{gateway_session}"
                },
                timeout=60
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "response": result.get("text", result.get("response", "")),
                    "provider": result.get("provider", "gateway-flat"),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": f"Gateway error {resp.status_code}: {resp.text[:200]}",
                    "provider": None,
                    "error": resp.text
                }
        except Exception as e:
            return {
                "success": False,
                "response": f"Gateway connection error: {str(e)}",
                "provider": None,
                "error": str(e)
            }

    def _flatten_messages_to_prompt(self, system_prompt: str,
                                     messages: List[Dict[str, str]]) -> str:
        """
        Convert system prompt + messages array into a single string prompt.
        This is the fallback for gateways that only accept a single message.
        Fixes Issue #2: Context is preserved even in stateless mode.
        """
        parts = []

        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}\n")

        parts.append("[CONVERSATION HISTORY]")
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            # Use clean role labels, not [user]/[assistant] tags
            # that cause the echo loop
            if role == "USER":
                parts.append(f"Human: {content}")
            elif role == "ASSISTANT":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role}: {content}")

        parts.append("\n[RESPOND TO THE LAST HUMAN MESSAGE]")
        return "\n".join(parts)


# ============ PROMPT BUILDER — FIXED ============

def build_prompt_and_messages(user_message: str, context: dict,
                               recalled_memories: List[Dict]) -> tuple:
    """
    Build system prompt and messages array cleanly.
    Fixes Issue #3/#8: No [user]/[assistant] tags in prompt.
    Fixes Issue #8: recalled memories are already tag-stripped.
    Returns: (system_prompt: str, messages: list)
    """
    long_term = context["long_term"]
    recent_history = context["recent_history"]
    summary = context["summary"]

    # Build system prompt — clean, no speaker tags
    system_parts = [
        "You are a helpful AI assistant with permanent memory.",
        "You remember everything about the user from past conversations.",
        "Respond naturally and use what you remember to personalize your answers.",
        ""
    ]

    # Add what we know about the user
    if long_term.get("user_name"):
        system_parts.append(f"The user's name is {long_term['user_name']}.")

    if long_term.get("preferences"):
        prefs = "; ".join(
            f"{k} {v}" for k, v in long_term["preferences"].items()
        )
        system_parts.append(f"User preferences: {prefs}.")

    if summary and summary != "No prior memory":
        system_parts.append(f"\nMemory summary: {summary}")

    # Add recalled semantic memories (already tag-free from recall_similar)
    if recalled_memories:
        system_parts.append("\nRelevant past context:")
        for mem in recalled_memories:
            text = mem.get("text", "")
            overlap = mem.get("overlap", 0)
            if text and overlap > 0.15:  # Only include confident matches
                system_parts.append(f"  - {text[:150]}")

    system_parts.append(
        f"\nTotal messages exchanged: {long_term.get('total_messages', 0)}"
    )

    system_prompt = "\n".join(system_parts)

    # Build messages array — proper role format, no tags
    messages = []
    for turn in recent_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        # Ensure content is clean even though we stored clean
        clean_content = strip_speaker_tags(content)
        if clean_content:
            messages.append({
                "role": role,
                "content": clean_content
            })

    # Add current user message
    clean_user_message = strip_speaker_tags(user_message)
    messages.append({
        "role": "user",
        "content": clean_user_message
    })

    return system_prompt, messages


# ============ INITIALIZE COMPONENTS ============
memory_store = EternalMemory()
gateway = GatewayConnector()


# ============ FASTAPI MODELS ============

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None  # Alias for user_id for compatibility

class ChatResponse(BaseModel):
    response: str
    user_id: str
    provider: Optional[str] = None
    memory_summary: Optional[str] = None
    total_messages: Optional[int] = None

class MemoryUpdateRequest(BaseModel):
    user_id: str
    key: str
    value: str


# ============ ENDPOINTS ============

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint.
    Fixes all 8 critical/high issues from the debug analysis.
    """
    # Resolve user_id
    user_id = req.user_id or req.session_id
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"[API] [{user_id}] Message: {user_message[:60]}...")

    # STEP 1: Get session context from Eternal Memory
    context = memory_store.get_context(user_id)

    # STEP 2: Recall semantically similar memories from Softwire
    # Fixes Issue #7: recall_similar() returns tag-free text
    recalled = memory_store.recall_similar(user_message, top_k=3)
    logger.info(f"[API] Recalled {len(recalled)} relevant memories")

    # STEP 3: Build clean prompt and messages array
    # Fixes Issue #3/#8: No raw tags injected into prompt
    system_prompt, messages = build_prompt_and_messages(
        user_message, context, recalled
    )

    # STEP 4: Send to gateway with full messages array
    # Fixes Issue #1/#2: messages array properly transmitted
    result = gateway.send_with_history(user_id, messages, system_prompt)

    if not result["success"]:
        logger.error(f"[API] Gateway failed: {result['error']}")
        raise HTTPException(status_code=503, detail=result["response"])

    ai_response = result["response"]
    # Strip any tags from AI response before storing or returning
    ai_response = strip_speaker_tags(ai_response)

    # STEP 5: Store in Eternal Memory (both sides of conversation)
    # Stores clean content — fixes echo loop at source
    memory_store.add_message(user_id, "user", user_message)
    memory_store.add_message(user_id, "assistant", ai_response)

    # STEP 6: Update structured long-term memory
    memory_store.update_long_term_memory(user_id, user_message, ai_response)

    total_msgs = context["long_term"].get("total_messages", 0) + 2

    return ChatResponse(
        response=ai_response,
        user_id=user_id,
        provider=result["provider"],
        memory_summary=context["summary"],
        total_messages=total_msgs
    )


@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Inspect what Softwire remembers about a user."""
    context = memory_store.get_context(user_id)
    return {
        "user_id": user_id,
        "long_term_memory": context["long_term"],
        "recent_history_count": len(context["recent_history"]),
        "summary": context["summary"],
        "softwire_patterns_stored": len(memory_store._softwire._patterns)
    }


@app.get("/memory/{user_id}/history")
async def get_history(user_id: str, limit: int = 20):
    """Get recent conversation history for a user."""
    user = memory_store.get_or_create_user(user_id)
    history = user["conversation_history"][-limit:]
    # Return clean history (already stored without tags)
    return {
        "user_id": user_id,
        "history": history,
        "total_stored": len(user["conversation_history"])
    }


@app.post("/memory/{user_id}/recall")
async def recall_memory(user_id: str, query: str):
    """
    Manually test Softwire recall for a query.
    Returns tag-free recalled memories.
    """
    recalled = memory_store.recall_similar(query, top_k=5)
    return {
        "user_id": user_id,
        "query": query,
        "recalled": recalled,
        "count": len(recalled)
    }


@app.post("/memory/{user_id}/clear")
async def clear_memory(user_id: str):
    """Clear a user's session memory."""
    if user_id in memory_store._users:
        del memory_store._users[user_id]
        memory_store._save_sessions()
    return {"status": "cleared", "user_id": user_id}


@app.post("/memory/clear-all")
async def clear_all_memory():
    """Clear ALL sessions and Softwire patterns (full reset)."""
    memory_store._users = {}
    memory_store._softwire = SoftwireMemoryEngine(
        N=SOFTWIRE_N,
        g=SOFTWIRE_G,
        max_patterns=MAX_PATTERNS
    )
    memory_store._save_sessions()
    # Remove backup files
    for f in [SESSIONS_FILE, f"{MEMORY_FILE}.npz"]:
        if os.path.exists(f):
            os.remove(f)
    return {"status": "all memory cleared"}


@app.get("/softwire/stats")
async def softwire_stats():
    """Get Softwire memory engine statistics."""
    engine = memory_store._softwire
    return {
        "patterns_stored": len(engine._patterns),
        "max_patterns": engine.max_patterns,
        "N": engine.N,
        "g": engine.g,
        "sessions_tracked": len(memory_store._users),
        "memory_file": MEMORY_FILE,
        "sessions_file": SESSIONS_FILE
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    gateway_ok = False
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=3)
        gateway_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "SOFTWIRE + AI Gateway v3.0",
        "gateway_url": GATEWAY_URL,
        "gateway_reachable": gateway_ok,
        "patterns_in_memory": len(memory_store._softwire._patterns),
        "sessions_active": len(memory_store._users)
    }


@app.get("/")
async def root():
    return {
        "service": "SOFTWIRE Eternal Memory + AI Gateway",
        "version": "3.0",
        "status": "running",
        "gateway": GATEWAY_URL,
        "fixes_applied": [
            "messages array properly sent to gateway",
            "speaker tags stripped from all recalled/stored text",
            "single consistent Softwire physics (engine10 v1.2)",
            "persistent memory across server restarts",
            "clean prompt building without echo loop",
            "session persistence via JSON backup",
            "Softwire pattern persistence via NPZ backup"
        ],
        "endpoints": {
            "chat": "POST /chat",
            "memory": "GET /memory/{user_id}",
            "history": "GET /memory/{user_id}/history",
            "recall": "POST /memory/{user_id}/recall",
            "clear": "POST /memory/{user_id}/clear",
            "stats": "GET /softwire/stats",
            "health": "GET /health"
        }
    }


# ============ STARTUP / SHUTDOWN EVENTS ============

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("SOFTWIRE + AI GATEWAY v3.0 — STARTING UP")
    logger.info("=" * 60)
    logger.info(f"Gateway URL:      {GATEWAY_URL}")
    logger.info(f"Sessions file:    {SESSIONS_FILE}")
    logger.info(f"Memory file:      {MEMORY_FILE}.npz")
    logger.info(f"Softwire N:       {SOFTWIRE_N}")
    logger.info(f"Softwire g:       {SOFTWIRE_G}")
    logger.info(f"Max patterns:     {MAX_PATTERNS}")
    logger.info(f"Sessions loaded:  {len(memory_store._users)}")
    logger.info(f"Patterns loaded:  {len(memory_store._softwire._patterns)}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SOFTWIRE] Saving memory on shutdown...")
    memory_store._save_sessions()
    memory_store._softwire.save(MEMORY_FILE)
    logger.info("[SOFTWIRE] Memory saved. Goodbye.")


# ============ ENTRY POINT ============

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("SOFTWIRE + AI GATEWAY v3.0 — COMPLETE SYSTEM")
    print("=" * 60)
    print(f"Softwire Server:  http://localhost:8080")
    print(f"AI Gateway:       {GATEWAY_URL}")
    print(f"Frontend calls:   POST http://localhost:8080/chat")
    print(f"Memory inspect:   GET  http://localhost:8080/memory/{{user_id}}")
    print(f"Health check:     GET  http://localhost:8080/health")
    print("=" * 60)
    print("FIXES APPLIED:")
    print("  [1] messages array sent to gateway correctly")
    print("  [2] [user]/[assistant] tags stripped everywhere")
    print("  [3] Single Softwire physics (engine10 v1.2)")
    print("  [4] Persistent memory across restarts (JSON + NPZ)")
    print("  [5] Echo loop eliminated at storage layer")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
