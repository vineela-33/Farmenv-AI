"""
Inference Script - FarmEnv-AI
OpenEnv Hackathon Submission
"""

import os
import requests
import textwrap
from typing import List, Optional
from openai import OpenAI

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("FARM_TASK", "irrigate")
BENCHMARK = os.getenv("FARM_BENCHMARK", "farmenv-ai")
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

# HuggingFace Space base URL
SPACE_URL = os.getenv("SPACE_URL", "https://vineela-n-farmenv-ai.hf.space")

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent managing a smart farm environment.
    Each turn you must choose one action:
    0 = do_nothing
    1 = irrigate_low
    2 = irrigate_medium
    3 = irrigate_high
    4 = apply_fertilizer

    Your goal is to maximize crop health and minimize water waste.
    Reply with exactly one integer (0-4). No explanation, just the number.
""").strip()


# ─── Logging helpers ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Environment API calls ────────────────────────────────────────────────────

def env_reset():
    response = requests.post(f"{SPACE_URL}/reset")
    return response.json()


def env_step(action: int):
    response = requests.post(f"{SPACE_URL}/step", json={"action": action})
    return response.json()


def env_state():
    response = requests.get(f"{SPACE_URL}/state")
    return response.json()


# ─── LLM action selection ─────────────────────────────────────────────────────

def get_action(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> int:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Step: {step}
        Current farm state: {observation}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Choose your next action (0-4).
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = int(text[0]) if text and text[0].isdigit() else 1
        return max(0, min(4, action))
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 1  # default: irrigate_low


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs = env_reset()
        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM
            action = get_action(client, step, obs, last_reward, history)

            # Take step in environment
            result = env_step(action)

            obs = result.get("observation", result)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("error", None)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=str(action), reward=reward, done=done, error=error)

            history.append(f"Step {step}: action={action} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate final score
        total_reward = sum(rewards)
        max_possible = MAX_STEPS * 10.0
        score = min(max(total_reward / max_possible, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
