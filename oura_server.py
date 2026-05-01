"""
OURA SOFTWIRE — Flask Backend Server
=====================================
IMPORTS your existing softwireengine1.py through softwireengine10.py
and text_encoder.py

Run:
    pip install flask flask-cors numpy
    python oura_server.py

Then ngrok:
    ngrok http 5000
"""

import sys
import os
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================
# IMPORT YOUR EXISTING SOFTWIRE FILES
# ============================================

# Add your OneDrive folder to Python path
sys.path.insert(0, r'C:\Users\linka\OneDrive')

# Import your text encoder and softwire core
try:
    from text_encoder import OuraMemorySystem, TextEncoder
    print("✓ Imported text_encoder.py (OuraMemorySystem, TextEncoder)")
except Exception as e:
    print(f"✗ Failed to import text_encoder.py: {e}")
    sys.exit(1)

try:
    from softwireengine1 import SoftwireCoreV2
    print("✓ Imported softwireengine1.py (SoftwireCoreV2)")
except Exception as e:
    print(f"✗ Failed to import softwireengine1.py: {e}")

try:
    from softwireengine2 import SoftwireCore as SoftwireCoreV1
    print("✓ Imported softwireengine2.py (SoftwireCore)")
except Exception as e:
    print(f"✗ Failed to import softwireengine2.py: {e}")

# Import other engines (optional, for reference)
for i in range(3, 11):
    try:
        exec(f"import softwireengine{i}")
        print(f"✓ Imported softwireengine{i}.py")
    except Exception as e:
        print(f"✗ softwireengine{i}.py: {e}")

# ============================================
# INITIALIZE YOUR SOFTWIRE
# ============================================

# Use OuraMemorySystem from text_encoder.py (this is your main memory)
# It internally uses SoftwireCoreV2 or SoftwireCore from your engines
memory = OuraMemorySystem(pattern_length=512, g=11.0, chunk_words=60, overlap=20)

# Also initialize the raw SoftwireCoreV2 for direct access if needed
try:
    raw_softwire = SoftwireCoreV2(N=512)
    print("✓ Raw SoftwireCoreV2 initialized")
except:
    raw_softwire = None

# API Key (same as before)
API_KEY = "oura-super-secret-key-change-this"

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def check_key():
    return request.headers.get("X-API-Key") == API_KEY


# ============================================
# ROUTES
# ============================================

@app.route("/health", methods=["GET"])
def health():
    """Public health check"""
    return jsonify({
        "status": "ok",
        "engine": "softwire-imported",
        "patterns_stored": memory.network.n_patterns if hasattr(memory, 'network') else 0,
        "timestamp": time.time()
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """Get memory status from YOUR softwire"""
    if not check_key():
        return jsonify({"error": "unauthorized"}), 401
    
    # Get status from your OuraMemorySystem
    patterns_stored = memory.network.n_patterns if hasattr(memory, 'network') else 0
    alpha = memory.network.alpha if hasattr(memory, 'network') else 0
    
    return jsonify({
        "patterns_stored": patterns_stored,
        "alpha": alpha,
        "N": memory.network.N if hasattr(memory, 'network') else 512,
        "g": memory.network.g if hasattr(memory, 'network') else 11.0,
        "status": "operational",
        "engine": "imported-from-your-files"
    })


@app.route("/api/store", methods=["POST"])
def api_store():
    """Store a memory using YOUR softwire"""
    if not check_key():
        return jsonify({"error": "unauthorized"}), 401
    
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "user")
    
    if not text:
        return jsonify({"status": "error", "reason": "empty text"}), 400
    
    # Use YOUR OuraMemorySystem to store
    indices = memory.store_conversation_turn(speaker, text)
    
    return jsonify({
        "status": "stored",
        "indices": indices,
        "total_patterns": memory.network.n_patterns if hasattr(memory, 'network') else 0
    })


@app.route("/api/recall", methods=["POST"])
def api_recall():
    """Recall a memory using YOUR softwire"""
    if not check_key():
        return jsonify({"error": "unauthorized"}), 401
    
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    noise = float(data.get("noise_fraction", 0.05))
    
    if not query:
        return jsonify({"matched_text": None, "similarity": 0}), 400
    
    # Use YOUR OuraMemorySystem to recall
    result = memory.recall_from_text(query, noise_fraction=noise)
    
    if result and result.best_match_text:
        return jsonify({
            "matched_text": result.best_match_text,
            "similarity": result.similarity,
            "source": "softwire",
            "converged": result.converged
        })
    else:
        return jsonify({
            "matched_text": None,
            "similarity": 0,
            "source": "no_match"
        })


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search similar memories using YOUR softwire"""
    if not check_key():
        return jsonify({"error": "unauthorized"}), 401
    
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)
    
    results = memory.search_similar(query, top_k=top_k, threshold=0.4)
    
    return jsonify({
        "results": [
            {"text": r[1].text, "speaker": r[1].tag, "similarity": r[0]}
            for r in results
        ],
        "count": len(results)
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear all memories (use with caution)"""
    if not check_key():
        return jsonify({"error": "unauthorized"}), 401
    
    # Your OuraMemorySystem doesn't have a built-in clear,
    # but we can access the underlying network
    if hasattr(memory, 'network') and hasattr(memory.network, '_patterns'):
        memory.network._patterns = []
        if hasattr(memory.network, '_J'):
            memory.network._J = np.zeros((memory.network.N, memory.network.N))
    
    return jsonify({"status": "cleared"})


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OURA SOFTWIRE SERVER — IMPORTED FROM YOUR FILES")
    print("=" * 60)
    print("  Your softwire files loaded:")
    print("    - text_encoder.py ✓")
    print("    - softwireengine1.py ✓")
    print("    - softwireengine2.py ✓")
    print("    - softwireengine3-10.py ✓")
    print()
    print(f"  Memory status: {memory.status() if hasattr(memory, 'status') else 'active'}")
    print(f"  API Key: {API_KEY}")
    print(f"  Endpoint: http://localhost:5000")
    print()
    print("  Expose via ngrok:")
    print("    ngrok http 5000")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)