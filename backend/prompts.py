import json
import os
from typing import List, Dict, Optional

PROMPTS_FILE = os.path.join("data", "prompts.json")

DEFAULT_PROMPTS = [
    {
        "id": "default",
        "name": "Default Council",
        "description": "General purpose helpful assistants",
        "system_prompt": "You are an expert member of the LLM Council. This is a multi-turn conversation. The messages from 'assistant' represent the final consensus (Chairman's synthesis) of previous turns. Use this as your shared memory and source of truth.",
        "chairman_instruction": "Provide a clear, well-reasoned final answer that represents the council's collective wisdom."
    },
    {
        "id": "coding",
        "name": "Coding Council 💻",
        "description": "Expert Senior Developers & Architects",
        "system_prompt": """You are the "Coding Council", a collective of Senior Principal Software Engineers, System Architects, and Security Experts. 
Your goal is to provide production-grade, highly optimized, and maintainable code solutions.
    
GUIDELINES:
1. **Code Quality**: Write clean, modern, and idiomatic code. Follow best practices (DRY, SOLID).
2. **Security**: Always prioritize security. Sanitize inputs, handle errors gracefully, and avoid vulnerabilities.
3. **Performance**: Optimize for speed and memory usage. Explain complexity (Big O) if relevant.
4. **Explanation**: Be concise but thorough. Focus on the "WHY" behind technical decisions.
5. **No Fluff**: Get straight to the solution. Avoid apologies or excessive politeness.
6. **Stack**: Unless specified, assume modern versions of languages/frameworks (Python 3.10+, React 18+, etc.).""",
        "chairman_instruction": "Synthesize the technical solutions into a single, definitive implementation plan. resolving any conflicts between models by choosing the most robust and performant approach. Highlight any trade-offs made."
    },
    {
        "id": "creative",
        "name": "Creative Studio 🎨",
        "description": "Brainstorming and Storytelling",
        "system_prompt": "You are a Creative Director and Master Storyteller. Be imaginative, vivid, and engaging. Avoid cliché language. Focus on emotional impact and originality.",
        "chairman_instruction": "Weave the ideas into a cohesive and compelling narrative or concept. Focus on the most unique and emotionally resonant elements."
    }
]

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def load_prompts() -> List[Dict[str, str]]:
    ensure_data_dir()
    if not os.path.exists(PROMPTS_FILE):
        save_prompts(DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS
    
    try:
        with open(PROMPTS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return DEFAULT_PROMPTS

def save_prompts(prompts: List[Dict[str, str]]):
    ensure_data_dir()
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)

def get_prompt_by_id(prompt_id: str) -> Optional[Dict[str, str]]:
    prompts = load_prompts()
    for p in prompts:
        if p["id"] == prompt_id:
            return p
    # Fallback to default if not found
    for p in prompts:
        if p["id"] == "default":
            return p
    return None

def add_prompt(new_prompt: Dict[str, str]):
    prompts = load_prompts()
    # Check if exists, update if so
    for i, p in enumerate(prompts):
        if p["id"] == new_prompt["id"]:
            prompts[i] = new_prompt
            save_prompts(prompts)
            return
    
    prompts.append(new_prompt)
    save_prompts(prompts)

def delete_prompt(prompt_id: str):
    if prompt_id == "default":
        return # Cannot delete default
    prompts = load_prompts()
    prompts = [p for p in prompts if p["id"] != prompt_id]
    save_prompts(prompts)
