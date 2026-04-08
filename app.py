import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from farm_env import FarmEnv
from agent import FarmAgent

app = Flask(__name__)
CORS(app)

env = FarmEnv()
agent = FarmAgent()
episode_count = 0
best_reward = -999
reward_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/reset", methods=["POST"])
def reset():
    observation = env.reset()
    return jsonify({
        "observation": observation,
        "success": True
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.json
    action = data.get("action", 3)
    state, reward, done, info = env.step(action)
    return jsonify({
        "observation": state,
        "reward": round(reward, 2),
        "done": done,
        "error": None,
        "info": info,
        "success": True
    })

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify({
        "observation": env.get_state(),
        "success": True
    })

@app.route("/agent/run", methods=["POST"])
def agent_run():
    global episode_count, best_reward, reward_history

    episode_count += 1
    total_reward, steps = agent.run_episode(env)
    reward_history.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward

    return jsonify({
        "success": True,
        "episode": episode_count,
        "total_reward": total_reward,
        "best_reward": best_reward,
        "epsilon": round(agent.epsilon, 3),
        "steps": steps,
        "reward_history": reward_history[-20:]
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "success": True,
        "episode_count": episode_count,
        "best_reward": best_reward,
        "epsilon": round(agent.epsilon, 3),
        "reward_history": reward_history[-20:]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
