import os
import textwrap
from openai import OpenAI
from farm_env import FarmEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = textwrap.dedent("""
You are controlling a farm. You have these actions:
0: Water crops
1: Apply fertilizer
2: Apply pesticide
3: Do nothing
4: Harvest

Your goal: Maximize harvest yield. Water when dry, fertilize when soil is low, use pesticide when pests are high.
Only harvest when growth_stage >= 3.

Return ONLY the action number (0-4), nothing else.
""").strip()

def predict(observation, config=None):
    user_prompt = f"""
Current farm state:
- Day: {observation.get('day', 0)}
- Water level: {observation.get('water_level', 50)}
- Soil health: {observation.get('soil_health', 50)}
- Pest level: {observation.get('pest_level', 50)}
- Growth stage: {observation.get('growth_stage', 0)}/5
- Weather: {observation.get('weather', 'sunny')}

Choose action (0-4):
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=10
    )
    
    try:
        action = int(response.choices[0].message.content.strip())
        action = max(0, min(4, action))
    except:
        action = 0
    
    return action

# Required for the validator
def reset():
    env = FarmEnv()
    return env.reset()

def step(action):
    # This is just a stub - actual step happens in the HF Space
    pass
