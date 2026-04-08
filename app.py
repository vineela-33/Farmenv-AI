import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from farm_env import FarmEnv

app = Flask(__name__)
CORS(app)

env = FarmEnv()

@app.route("/reset", methods=["POST", "GET", "OPTIONS"])
def reset():
    global env
    env = FarmEnv()  # Re-initialize
    env.reset()
    return jsonify({"status": "ok"}), 200

@app.route("/step", methods=["POST"])
def step():
    global env
    data = request.json
    action = data.get("action", 0)
    
    try:
        observation, reward, done, info = env.step(action)
        return jsonify({
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "error": None
        }), 200
    except Exception as e:
        return jsonify({
            "observation": None,
            "reward": 0.0,
            "done": True,
            "error": str(e)
        }), 200

@app.route("/state", methods=["GET"])
def get_state():
    global env
    try:
        state = env.get_state() if hasattr(env, 'get_state') else {"position": 0}
        return jsonify({"state": state}), 200
    except:
        return jsonify({"state": {}}), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "FarmEnv AI API running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
