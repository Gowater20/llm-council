"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'usage': data.get('usage', {})
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def fetch_available_models() -> List[Dict[str, Any]]:
    """
    Fetch the list of available models from OpenRouter.

    Returns:
        List of model dictionaries with id, name, and pricing
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()
            
            models = []
            for m in data.get('data', []):
                models.append({
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "pricing": m.get("pricing", {}),
                    "context_length": m.get("context_length"),
                    "description": m.get("description")
                })
            return models
    except Exception as e:
        print(f"Error fetching models from OpenRouter: {e}")
        return []


def calculate_cost(usage: Dict[str, Any], pricing: Dict[str, Any]) -> float:
    """
    Calculate the cost of a request based on usage and pricing.

    Args:
        usage: Usage dictionary from OpenRouter response
        pricing: Pricing dictionary (prompt and completion cost per 1M tokens)

    Returns:
        Calculated cost in USD
    """
    if not usage or not pricing:
        return 0.0

    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)

    # Pricing is per 1M tokens, so divide by 1,000,000
    prompt_price = float(pricing.get('prompt', 0)) / 1_000_000
    completion_price = float(pricing.get('completion', 0)) / 1_000_000

    return (prompt_tokens * prompt_price) + (completion_tokens * completion_price)
