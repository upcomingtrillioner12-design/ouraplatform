"""
SOFTWIRE + AI GATEWAY - Complete Integration Server
Your Softwire Brain + Gateway Router = Eternal Memory AI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import uuid
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("softwire")

app = FastAPI(title="SOFTWARE + AI Gateway", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIGURATION ============
GATEWAY_URL = "http://localhost:8000"  # Your AI Gateway

# ============ YOUR SOFTWIRE - ETERNAL MEMORY STORAGE ============
class EternalMemory:
    """This is YOUR SOFTWIRE BRAIN - Stores everything forever"""
    
    def __init__(self):
        # In production, use Redis/PostgreSQL. For now, in-memory
        self._users: Dict[str, dict] = {}
    
    def get_or_create_user(self, user_id: str) -> dict:
        if user_id not in self._users:
            self._users[user_id] = {
                "user_id": user_id,
                "session_id": None,
                "conversation_history": [],
                "long_term_memory": {
                    "user_name": None,
                    "preferences": {},
                    "facts_learned": [],
                    "first_seen": datetime.now().isoformat(),
                    "total_messages": 0
                },
                "gateway_session": None
            }
            logger.info(f"[SOFTWIRE] New user created: {user_id}")
        return self._users[user_id]
    
    def add_message(self, user_id: str, role: str, content: str):
        user = self.get_or_create_user(user_id)
        user["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        user["long_term_memory"]["total_messages"] += 1
        
        # Keep only last 50 messages in active history (prune old)
        if len(user["conversation_history"]) > 50:
            user["conversation_history"] = user["conversation_history"][-50:]
    
    def get_context(self, user_id: str) -> dict:
        """YOUR SOFTWIRE - Builds context from eternal memory"""
        user = self.get_or_create_user(user_id)
        
        # Get last 10 messages for recent context
        recent_history = user["conversation_history"][-10:]
        
        # Build context window
        context = {
            "long_term": user["long_term_memory"],
            "recent_history": recent_history,
            "summary": self._summarize_memory(user["long_term_memory"])
        }
        return context
    
    def _summarize_memory(self, memory: dict) -> str:
        """Generate a summary of what Softwire remembers about user"""
        facts = []
        if memory.get("user_name"):
            facts.append(f"User's name is {memory['user_name']}")
        if memory.get("preferences"):
            facts.append(f"User prefers: {memory['preferences']}")
        if memory.get("facts_learned"):
            facts.extend(memory["facts_learned"][-3:])  # Last 3 facts
        
        return ". ".join(facts) if facts else "No prior memory"
    
    def update_long_term_memory(self, user_id: str, user_message: str, ai_response: str):
        """YOUR SOFTWIRE - Extracts and stores important information"""
        user = self.get_or_create_user(user_id)
        memory = user["long_term_memory"]
        
        # Extract name
        if "my name is" in user_message.lower():
            parts = user_message.lower().split("my name is")
            if len(parts) > 1:
                name = parts[1].strip().split()[0].capitalize()
                memory["user_name"] = name
                logger.info(f"[SOFTWIRE] Learned user name: {name}")
        
        # Extract preferences (simple example - enhance as needed)
        if "i like" in user_message.lower() or "i prefer" in user_message.lower():
            memory["preferences"]["last_mentioned"] = user_message
        
        # Store interesting facts
        if len(user_message) > 20 and "?" not in user_message:
            memory["facts_learned"].append({
                "fact": user_message[:100],
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 20 facts
            memory["facts_learned"] = memory["facts_learned"][-20:]


# ============ GATEWAY CONNECTOR ============
class GatewayConnector:
    """Bridges YOUR SOFTWIRE to the AI Gateway"""
    
    def __init__(self):
        self.gateway_url = GATEWAY_URL
        self._gateway_sessions: Dict[str, str] = {}
    
    def get_gateway_session(self, user_id: str) -> str:
        """Get or create gateway session for this user"""
        if user_id not in self._gateway_sessions:
            try:
                resp = requests.post(f"{self.gateway_url}/session/new", timeout=5)
                if resp.status_code == 200:
                    self._gateway_sessions[user_id] = resp.json()["session_id"]
                    logger.info(f"[GATEWAY] Session created for user {user_id}")
                else:
                    self._gateway_sessions[user_id] = f"gateway_{user_id}"
            except Exception as e:
                logger.error(f"[GATEWAY] Failed to create session: {e}")
                self._gateway_sessions[user_id] = f"gateway_{user_id}"
        return self._gateway_sessions[user_id]
    
    def send_message(self, user_id: str, full_prompt: str) -> dict:
        """Send message to AI Gateway"""
        gateway_session = self.get_gateway_session(user_id)
        
        try:
            resp = requests.post(
                f"{self.gateway_url}/chat",
                json={
                    "message": full_prompt,
                    "session_id": gateway_session
                },
                timeout=60
            )
            
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "response": result["text"],
                    "provider": result.get("provider", "unknown"),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": f"Gateway error: {resp.status_code}",
                    "provider": None,
                    "error": resp.text
                }
        except Exception as e:
            return {
                "success": False,
                "response": f"Connection error: {str(e)}",
                "provider": None,
                "error": str(e)
            }


# ============ CREATE PROMPT WITH SOFTWIRE MEMORY ============
def build_prompt_with_memory(user_message: str, context: dict) -> str:
    """
    YOUR SOFTWIRE - Builds the prompt including eternal memory
    This is what gives your AI persistent memory!
    """
    long_term = context["long_term"]
    recent = context["recent_history"]
    summary = context["summary"]
    
    # Build system message with memory
    system_parts = [
        "You are an AI assistant with PERMANENT MEMORY.",
        "You remember everything about the user across all sessions.",
        "",
        "=== WHAT YOU REMEMBER ABOUT THIS USER ===",
    ]
    
    if long_term.get("user_name"):
        system_parts.append(f"User's name: {long_term['user_name']}")
    
    if long_term.get("preferences"):
        system_parts.append(f"User preferences: {json.dumps(long_term['preferences'], indent=2)}")
    
    if summary and summary != "No prior memory":
        system_parts.append(f"\nSummary of past conversations:\n{summary}")
    
    if recent:
        system_parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent[-5:]:  # Last 5 messages
            system_parts.append(f"{msg['role']}: {msg['content']}")
    
    system_parts.append(f"\n=== CURRENT USER MESSAGE ===\n{user_message}")
    system_parts.append("\nRespond naturally while remembering everything from above.")
    
    return "\n".join(system_parts)


# ============ FASTAPI ENDPOINTS ============
memory_store = EternalMemory()
gateway = GatewayConnector()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: str
    provider: Optional[str] = None
    memory_summary: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Main endpoint - Frontend calls this"""
    
    # Generate or use existing user_id
    user_id = req.user_id
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"[API] Message from {user_id}: {req.message[:50]}...")
    
    # STEP 1: Get Softwire memory context
    context = memory_store.get_context(user_id)
    
    # STEP 2: Build prompt with eternal memory
    full_prompt = build_prompt_with_memory(req.message, context)
    
    # STEP 3: Send to Gateway (which routes to free providers)
    result = gateway.send_message(user_id, full_prompt)
    
    if result["success"]:
        # STEP 4: Store in Softwire's eternal memory
        memory_store.add_message(user_id, "user", req.message)
        memory_store.add_message(user_id, "assistant", result["response"])
        memory_store.update_long_term_memory(user_id, req.message, result["response"])
        
        return ChatResponse(
            response=result["response"],
            user_id=user_id,
            provider=result["provider"],
            memory_summary=context["summary"]
        )
    else:
        raise HTTPException(status_code=503, detail=result["response"])

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Debug endpoint - See what Softwire remembers"""
    context = memory_store.get_context(user_id)
    return {
        "user_id": user_id,
        "long_term_memory": context["long_term"],
        "conversation_count": len(context["recent_history"]),
        "summary": context["summary"]
    }

@app.post("/memory/{user_id}/clear")
async def clear_memory(user_id: str):
    """Clear user's memory (for testing)"""
    if user_id in memory_store._users:
        del memory_store._users[user_id]
    return {"status": "cleared", "user_id": user_id}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "SOFTWIRE + AI Gateway"}

@app.get("/")
async def root():
    return {
        "service": "SOFTWIRE Eternal Memory + AI Gateway",
        "status": "running",
        "gateway": GATEWAY_URL
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🔥 SOFTWIRE + AI GATEWAY - COMPLETE SYSTEM")
    print("="*50)
    print(f"Softwire Server: http://localhost:8080")
    print(f"AI Gateway:      {GATEWAY_URL}")
    print(f"Frontend should call: http://localhost:8080/chat")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)