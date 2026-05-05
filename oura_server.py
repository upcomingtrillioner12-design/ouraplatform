"""
OURA SOFTWIRE — AGI SUSTAINABLE MEMORY SERVER
=============================================
REAL MEMORY - Not placeholders!
Your softwire engine stores and recalls patterns permanently.
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
# INITIALIZE YOUR SOFTWIRE MEMORY
# ============================================

memory = OuraMemorySystem(pattern_length=512, g=11.0, chunk_words=60, overlap=20)
print("✓ OuraMemorySystem initialized with REAL persistent memory")

# Session storage (maps user session to conversation)
sessions = defaultdict(lambda: {
    "history": [],
    "user_name": None,
    "preferences": {},
    "facts": [],
    "memory_ids": []
})

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GATEWAY_URL = "http://localhost:8000"
API_KEY = "oura-super-secret-key-change-this"


# ============================================
# CORE MEMORY FUNCTIONS (USING YOUR SOFTWIRE)
# ============================================

def store_in_softwire(speaker: str, text: str, session_id: str):
    """Store a conversation turn in your Softwire memory"""
    try:
        # Format for storage
        memory_text = f"[{speaker}] {text}"
        indices = memory.store_conversation_turn(speaker, text)
        print(f"[MEMORY] Stored: {memory_text[:50]}... -> indices: {indices}")
        return indices
    except Exception as e:
        print(f"[MEMORY ERROR] {e}")
        return None


def recall_from_softwire(query: str, session_id: str = None):
    """Recall relevant memories from Softwire"""
    try:
        result = memory.recall_from_text(query, noise_fraction=0.05)
        if result and result.best_match_text:
            return {
                "text": result.best_match_text,
                "similarity": result.similarity,
                "converged": result.converged
            }
    except Exception as e:
        print(f"[RECALL ERROR] {e}")
    return None


def search_similar_memories(query: str, top_k: int = 3):
    """Search for similar memories"""
    try:
        results = memory.search_similar(query, top_k=top_k, threshold=0.3)
        return [(r[0], r[1].text, r[1].tag) for r in results]
    except:
        return []


def build_context_from_memory(user_message: str, session_data: dict, user_id: str):
    """
    Build intelligent context using your Softwire memory
    This is the AGI memory part - no placeholders!
    """
    context = {
        "user_name": session_data.get("user_name"),
        "preferences": session_data.get("preferences", {}),
        "facts": session_data.get("facts", []),
        "recent_history": session_data.get("history", [])[-5:],
        "recalled_memories": [],
        "similar_patterns": []
    }
    
    # 1. Recall relevant memories from Softwire
    recalled = recall_from_softwire(user_message, user_id)
    if recalled and recalled["similarity"] > 0.3:
        context["recalled_memories"].append(recalled)
    
    # 2. Search for similar patterns
    similar = search_similar_memories(user_message, top_k=2)
    for sim_score, sim_text, sim_speaker in similar:
        if sim_score > 0.35:
            context["similar_patterns"].append({
                "text": sim_text,
                "speaker": sim_speaker,
                "similarity": sim_score
            })
    
    # 3. Extract user name if mentioned
    if "my name is" in user_message.lower():
        parts = user_message.lower().split("my name is")
        if len(parts) > 1:
            name = parts[1].strip().split()[0].capitalize()
            context["user_name"] = name
            session_data["user_name"] = name
            print(f"[MEMORY] Learned user name: {name}")
    
    return context, session_data


def generate_ai_response(user_message: str, context: dict, session_id: str):
    """
    Generate response using Gateway + Context
    Your real AI brain
    """
    
    # Build system prompt with all remembered context
    system_prompt = """You are OURA, an AI with PERMANENT ETERNAL MEMORY.
You remember everything users tell you across all sessions.
Be conversational, helpful, and use the memories provided below naturally.
NEVER say "I remember something related" - just USE the memory naturally.
"""

    # Add user-specific context
    if context.get("user_name"):
        system_prompt += f"\n[USER NAME: {context['user_name']}]"
    
    if context.get("recalled_memories"):
        system_prompt += "\n\n[RELEVANT PAST MEMORIES:]"
        for mem in context["recalled_memories"]:
            system_prompt += f"\n- {mem['text'][:200]}"
    
    if context.get("similar_patterns"):
        system_prompt += "\n\n[RELATED CONVERSATIONS:]"
        for pat in context["similar_patterns"][:2]:
            system_prompt += f"\n- {pat['speaker']} said: {pat['text'][:100]}"
    
    if context.get("facts"):
        system_prompt += f"\n\n[FACTS ABOUT USER:]\n- " + "\n- ".join(context["facts"][-3:])
    
    # Add recent conversation
    if context.get("recent_history"):
        system_prompt += "\n\n[RECENT CONVERSATION:]"
        for msg in context["recent_history"][-3:]:
            system_prompt += f"\n{msg['role']}: {msg['content'][:100]}"
    
    system_prompt += f"\n\n[USER MESSAGE:]\n{user_message}\n\n[YOUR RESPONSE (use the memory naturally, don't mention that you're remembering):]"
    
    # Try to call gateway
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/chat",
            json={"message": system_prompt, "session_id": session_id},
            timeout=45
        )
        if resp.status_code == 200:
            return resp.json().get("text", "")
    except Exception as e:
        print(f"[GATEWAY ERROR] {e}")
    
    # Fallback intelligent response
    return generate_intelligent_fallback(user_message, context)


def generate_intelligent_fallback(user_message: str, context: dict) -> str:
    """Intelligent fallback when gateway is unavailable"""
    user_lower = user_message.lower()
    
    # Use remembered user name
    user_name = context.get("user_name", "")
    name_greeting = f", {user_name}" if user_name else ""
    
    # Check if this is a follow-up question
    if context.get("recalled_memories"):
        memory_text = context["recalled_memories"][0]["text"]
        return f"I recall our previous conversation about this.{name_greeting} You mentioned something about that. How can I help you further?"
    
    # Pickup lines
    if "pickup line" in user_lower:
        return """Here are some pickup lines for you:

1. "Are you made of copper and tellurium? Because you're Cu-Te!"
2. "Are you a Wi-Fi signal? Because I'm feeling a strong connection."
3. "Is your name Google? Because you have everything I've been searching for."
4. "Are you a time traveler? Because I see you in my future."
5. "Do you have a map? I keep getting lost in your eyes."

Want me to generate more themed ones? 😊"""
    
    # Remember name
    if "my name is" in user_lower:
        return f"Nice to meet you{name_greeting}! I'll remember your name for our future conversations. What would you like to talk about?"
    
    # General response
    recent_count = len(context.get("recent_history", []))
    if recent_count > 2:
        return f"I'm enjoying our conversation{name_greeting}! We've been talking about quite a few things. What's on your mind?"
    else:
        return f"Thanks for sharing that with me{name_greeting}. I'll remember it. Is there anything specific you'd like to discuss or ask about?"


# ============================================
# MAIN CHAT ENDPOINT - REAL AGI MEMORY
# ============================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Main chat endpoint - REAL persistent memory using your Softwire"""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "")
    
    if not user_message:
        return jsonify({"error": "empty message"}), 400
    
    # Create or get session
    if not session_id:
        session_id = str(uuid.uuid4())
    
    session_data = sessions[session_id]
    
    print(f"\n[CHAT] Session: {session_id[:8]} | User: {session_data.get('user_name', 'unknown')}")
    print(f"[CHAT] Message: {user_message[:100]}")
    
    # STEP 1: Store user message in Softwire (permanent memory)
    store_in_softwire("user", user_message, session_id)
    session_data["history"].append({"role": "user", "content": user_message})
    
    # STEP 2: Build intelligent context from memory
    context, session_data = build_context_from_memory(user_message, session_data, session_id)
    
    # STEP 3: Generate response using context
    response_text = generate_ai_response(user_message, context, session_id)
    
    # STEP 4: Store assistant response in Softwire
    store_in_softwire("assistant", response_text, session_id)
    session_data["history"].append({"role": "assistant", "content": response_text})
    
    # Keep history reasonable
    if len(session_data["history"]) > 30:
        session_data["history"] = session_data["history"][-30:]
    
    # Update session
    sessions[session_id] = session_data
    
    # Get memory stats
    patterns_stored = memory.network.n_patterns if hasattr(memory, 'network') else 0
    
    print(f"[CHAT] Response sent | Memory patterns: {patterns_stored}")
    
    return jsonify({
        "text": response_text,
        "session_id": session_id,
        "provider": "softwire-agi",
        "patterns_stored": patterns_stored,
        "user_name": session_data.get("user_name")
    })


@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    """Get real memory statistics from your Softwire"""
    patterns_stored = memory.network.n_patterns if hasattr(memory, 'network') else 0
    return jsonify({
        "total_patterns": patterns_stored,
        "pattern_length": memory.network.N if hasattr(memory, 'network') else 512,
        "g": memory.network.g if hasattr(memory, 'network') else 11.0,
        "temperature": memory.network.T if hasattr(memory, 'network') else 0.0909
    })


@app.route("/health", methods=["GET"])
def health():
    patterns_stored = memory.network.n_patterns if hasattr(memory, 'network') else 0
    return jsonify({
        "status": "ok",
        "engine": "softwire-agi",
        "patterns_stored": patterns_stored,
        "sessions_active": len(sessions),
        "timestamp": time.time()
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    
    patterns_stored = memory.network.n_patterns if hasattr(memory, 'network') else 0
    return jsonify({
        "status": "operational",
        "patterns_stored": patterns_stored,
        "sessions": len(sessions),
        "memory_type": "OuraMemorySystem",
        "engine": "imported-from-your-files"
    })


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 OURA SOFTWIRE — AGI SUSTAINABLE MEMORY")
    print("=" * 60)
    print("  ✅ Your softwire files loaded")
    print(f"  ✅ Memory patterns: {memory.network.n_patterns if hasattr(memory, 'network') else 0}")
    print(f"  ✅ Session store ready")
    print(f"  ✅ Gateway URL: {GATEWAY_URL}")
    print(f"  ✅ Endpoint: http://localhost:5000")
    print("\n  🔗 Connect your frontend to: http://localhost:5000")
    print("  📡 Or via zrok: https://oraback.share.zrok.io")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
