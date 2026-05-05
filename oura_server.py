"""
OURA SOFTWIRE — AGI SUSTAINABLE MEMORY SERVER (FIXED)
=====================================================
Fixes applied:
1. Single history store — oura_server owns it, gateway is stateless per-call
2. system_prompt sent as system role, NOT as user message
3. Softwire memory tags stripped before injecting into prompts
4. No double-append of history
"""

import sys
import time
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

# ============================================
# IMPORT YOUR EXISTING SOFTWIRE FILES
# ============================================

sys.path.insert(0, r'C:\Users\linka\OneDrive')

try:
    from text_encoder import OuraMemorySystem, TextEncoder
    print("✓ Imported text_encoder.py")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

try:
    from softwireengine1 import SoftwireCoreV2
    print("✓ Imported softwireengine1.py")
except Exception as e:
    print(f"✗ softwireengine1.py: {e}")

for i in range(2, 11):
    try:
        exec(f"import softwireengine{i}")
        print(f"✓ Imported softwireengine{i}.py")
    except:
        pass

# ============================================
# INITIALIZE SOFTWIRE MEMORY
# ============================================

memory = OuraMemorySystem(pattern_length=512, g=11.0, chunk_words=60, overlap=20)
print("✓ OuraMemorySystem initialized")

# oura_server owns ALL session state — gateway is called statelessly
sessions = defaultdict(lambda: {
    "history": [],        # full conversation, kept here only
    "user_name": None,
    "facts": [],
})

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GATEWAY_URL = "http://localhost:8000"
API_KEY = "oura-super-secret-key-change-this"

# ============================================
# MEMORY HELPERS
# ============================================

def store_in_softwire(speaker: str, text: str):
    """Store a turn — no speaker tag in the text itself"""
    try:
        indices = memory.store_conversation_turn(speaker, text)
        print(f"[MEMORY] Stored ({speaker}): {text[:50]}...")
        return indices
    except Exception as e:
        print(f"[MEMORY ERROR] {e}")
        return None


def recall_from_softwire(query: str):
    """Recall — returns clean text, no [speaker] prefix"""
    try:
        result = memory.recall_from_text(query, noise_fraction=0.05)
        if result and result.best_match_text and result.similarity > 0.35:
            # Strip any [speaker] tag that got stored previously
            text = result.best_match_text
            import re
            text = re.sub(r'^\[(user|assistant|User|Assistant)\]\s*', '', text)
            return {"text": text, "similarity": result.similarity}
    except Exception as e:
        print(f"[RECALL ERROR] {e}")
    return None


def search_similar_memories(query: str, top_k: int = 2):
    try:
        results = memory.search_similar(query, top_k=top_k, threshold=0.35)
        cleaned = []
        for r in results:
            import re
            text = re.sub(r'^\[(user|assistant|User|Assistant)\]\s*', '', r[1].text)
            cleaned.append((r[0], text))
        return cleaned
    except:
        return []


# ============================================
# CONTEXT BUILDER
# ============================================

def build_context(user_message: str, session_data: dict):
    ctx = {
        "user_name": session_data.get("user_name"),
        "facts": session_data.get("facts", []),
        "recent_history": session_data["history"][-6:],   # last 3 pairs
        "recalled": None,
        "similar": [],
    }

    recalled = recall_from_softwire(user_message)
    if recalled:
        ctx["recalled"] = recalled

    similar = search_similar_memories(user_message, top_k=2)
    ctx["similar"] = similar

    # Learn user name
    if "my name is" in user_message.lower():
        import re
        m = re.search(r"my name is ([A-Za-z]+)", user_message, re.IGNORECASE)
        if m:
            name = m.group(1).capitalize()
            ctx["user_name"] = name
            session_data["user_name"] = name
            print(f"[MEMORY] Learned name: {name}")

    return ctx


# ============================================
# GATEWAY CALL — sends history properly via messages array
# ============================================

def call_gateway(user_message: str, context: dict, session_id: str):
    """
    Send to gateway using a FRESH session each call.
    oura_server owns history — gateway must NOT accumulate its own.
    We pass history as the messages array, system prompt as system role.
    """

    # Build system prompt — context only, no history, no user message
    system_parts = [
        "You are OURA, an AI with persistent memory. "
        "Be conversational and helpful. "
        "Use the context below naturally — never say 'I remember something related'."
    ]

    if context.get("user_name"):
        system_parts.append(f"The user's name is {context['user_name']}.")

    if context.get("recalled"):
        system_parts.append(
            f"Relevant past memory: {context['recalled']['text'][:200]}"
        )

    if context.get("similar"):
        snippets = "; ".join(t[:100] for _, t in context["similar"][:2])
        system_parts.append(f"Related context: {snippets}")

    if context.get("facts"):
        system_parts.append("Known facts: " + "; ".join(context["facts"][-3:]))

    system_prompt = "\n".join(system_parts)

    # Build messages: system + recent history (already alternating user/assistant) + new user msg
    messages = [{"role": "system", "content": system_prompt}]
    for turn in context["recent_history"]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # Use a THROWAWAY session id so gateway never accumulates history
    throwaway_sid = "stateless-" + str(uuid.uuid4())

    try:
        # Try /chat/completions style (messages array)
        resp = requests.post(
            f"{GATEWAY_URL}/chat",
            json={
                "message": user_message,      # gateway's ChatRequest field
                "session_id": throwaway_sid,  # fresh every time = no gateway history buildup
                "messages": messages,         # full context (if gateway supports it)
            },
            timeout=45
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            provider = data.get("provider", "softwire")
            if text:
                return text, provider
    except Exception as e:
        print(f"[GATEWAY ERROR] {e}")

    return None, None


# ============================================
# FALLBACK RESPONSE
# ============================================

def fallback_response(user_message: str, context: dict) -> str:
    user_lower = user_message.lower()
    name = context.get("user_name", "")
    greeting = f", {name}" if name else ""

    if context.get("recalled"):
        return f"Based on what we've discussed before{greeting}: {context['recalled']['text'][:150]}. How can I help further?"

    if "pickup line" in user_lower:
        return (
            "Here are some Gen Z pickup lines:\n\n"
            "1. 'Are you a Wi-Fi signal? I'm feeling a strong connection.'\n"
            "2. 'No cap, you just broke my algorithm.'\n"
            "3. 'You must be trending, because you're all I see.'\n"
            "4. 'Are you my phone charger? Because I die without you.'\n"
            "5. 'Lowkey obsessed with you, not gonna lie.'\n\n"
            "Want more? 😄"
        )

    if "my name is" in user_lower:
        return f"Nice to meet you{greeting}! I'll remember that. What's on your mind?"

    return f"Got it{greeting}. I'm here — what would you like to talk about?"


# ============================================
# MAIN CHAT ENDPOINT
# ============================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "")

    if not user_message:
        return jsonify({"error": "empty message"}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    session_data = sessions[session_id]

    print(f"\n[CHAT] Session: {session_id[:8]} | User: {session_data.get('user_name','?')}")
    print(f"[CHAT] Message: {user_message[:100]}")

    # 1. Store user turn in softwire (clean, no tag prefix)
    store_in_softwire("user", user_message)

    # 2. Build context from memory (does NOT modify history yet)
    context = build_context(user_message, session_data)

    # 3. Append user message to history ONCE
    session_data["history"].append({"role": "user", "content": user_message})

    # 4. Call gateway statelessly
    response_text, provider = call_gateway(user_message, context, session_id)

    # 5. Fallback if gateway failed
    if not response_text:
        response_text = fallback_response(user_message, context)
        provider = "fallback"

    # 6. Store assistant reply in softwire + history ONCE
    store_in_softwire("assistant", response_text)
    session_data["history"].append({"role": "assistant", "content": response_text})

    # 7. Trim history (keep last 40 turns = 20 exchanges)
    if len(session_data["history"]) > 40:
        session_data["history"] = session_data["history"][-40:]

    patterns_stored = getattr(getattr(memory, 'network', None), 'n_patterns', 0)
    print(f"[CHAT] Response from {provider} | Patterns: {patterns_stored}")

    return jsonify({
        "text": response_text,
        "session_id": session_id,
        "provider": provider or "softwire-agi",
        "patterns_stored": patterns_stored,
        "user_name": session_data.get("user_name"),
    })


# ============================================
# OTHER ENDPOINTS (unchanged)
# ============================================

@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    n = getattr(getattr(memory, 'network', None), 'n_patterns', 0)
    N = getattr(getattr(memory, 'network', None), 'N', 512)
    g = getattr(getattr(memory, 'network', None), 'g', 11.0)
    T = getattr(getattr(memory, 'network', None), 'T', 0.0909)
    return jsonify({"total_patterns": n, "pattern_length": N, "g": g, "temperature": T})


@app.route("/health", methods=["GET"])
def health():
    n = getattr(getattr(memory, 'network', None), 'n_patterns', 0)
    return jsonify({
        "status": "ok",
        "engine": "softwire-agi",
        "patterns_stored": n,
        "sessions_active": len(sessions),
        "timestamp": time.time(),
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    n = getattr(getattr(memory, 'network', None), 'n_patterns', 0)
    return jsonify({
        "status": "operational",
        "patterns_stored": n,
        "sessions": len(sessions),
        "memory_type": "OuraMemorySystem",
    })


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OURA SOFTWIRE — AGI SUSTAINABLE MEMORY (FIXED)")
    print("=" * 60)
    n = getattr(getattr(memory, 'network', None), 'n_patterns', 0)
    print(f"  Memory patterns: {n}")
    print(f"  Gateway URL: {GATEWAY_URL}")
    print(f"  Endpoint: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
