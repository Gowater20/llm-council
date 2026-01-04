"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

council_type = os.getenv('COUNCIL_TYPE', 'paid')

if council_type == 'paid':
    # Council members - list of OpenRouter model identifiers (real models)
    COUNCIL_MODELS = [
        "openai/gpt-4o",
        "google/gemini-flash-1.5",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    # Chairman model - synthesizes final response
    CHAIRMAN_MODEL = "anthropic/claude-3.5-sonnet"
elif council_type == 'free':
    COUNCIL_MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-vl-7b-instruct:free"
    ]
    # Use a solid free model as Chairman
    CHAIRMAN_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
