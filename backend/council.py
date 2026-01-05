"""3-stage LLM Council orchestration."""

import asyncio
import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
try:
    from .openrouter import query_model, calculate_cost
    from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
except ImportError:
    from openrouter import query_model, calculate_cost
    from config import COUNCIL_MODELS, CHAIRMAN_MODEL

async def stage1_collect_responses(user_query: str, council_models: List[str] = None, history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    """
    if council_models is None:
        council_models = COUNCIL_MODELS

    messages = []
    if history:
        messages.extend(history)
    
    # Add a memory-reinforcing instruction for models in Stage 1
    query_with_context = user_query
    if history and len(history) > 1: # history[0] is always the system_hint
        # The history already contains previous Chairman syntheses as 'assistant' messages
        query_with_context = (
            f"[Current Council Turn: Follow-up]\n"
            f"Please review the previous consensus in the chat history and answer the new question.\n\n"
            f"New Question: {user_query}"
        )
        
    messages.append({"role": "user", "content": query_with_context})

    print(f"Stage 1: Querying {len(council_models)} models for query: '{user_query}'")

    async def query_and_process(model):
        # Add a small random jitter to avoid hitting rate limits with simultaneous requests
        delay = random.uniform(0.1, 1.2)
        await asyncio.sleep(delay)
        
        res = await query_model(model, messages)
        
        if res and isinstance(res, dict) and "error" in res:
            return {
                "model": model, 
                "status": "error", 
                "error": res["error"], 
                "response": "Failed to get response from model.", 
                "usage": {}
            }
        
        if res and isinstance(res, dict) and res.get('content'):
            return {
                "model": model, 
                "status": "success", 
                "response": res['content'], 
                "usage": res.get('usage', {})
            }
            
        return {
            "model": model, 
            "status": "error", 
            "error": "Model returned empty response or timed out.", 
            "response": "Failed to get response from model.", 
            "usage": {}
        }

    tasks = [query_and_process(model) for model in council_models]
    stage1_results = await asyncio.gather(*tasks)

    print(f"Stage 1 complete: {len([r for r in stage1_results if r['status'] == 'success'])} successful responses.")
    return stage1_results

async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: List[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.
    """
    if council_models is None:
        council_models = COUNCIL_MODELS

    # Create anonymized labels ONLY for successful Stage 1 responses
    successful_stage1 = [r for r in stage1_results if r['status'] == 'success']
    
    if not successful_stage1:
        return [], {}

    labels = [chr(65 + i) for i in range(len(successful_stage1))]
    
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, successful_stage1)
    }

    # Build the ranking prompt using successful responses
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, successful_stage1)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    print(f"Stage 2: Asking {len(council_models)} models to rank {len(successful_stage1)} responses")
    
    async def query_and_process_ranking(model):
        delay = random.uniform(0.1, 1.2)
        await asyncio.sleep(delay)

        res = await query_model(model, messages)
        
        if res and isinstance(res, dict) and "error" in res:
            return {
                "model": model, 
                "status": "error", 
                "error": res["error"], 
                "ranking": "Failed to get ranking from model.", 
                "parsed_ranking": [], 
                "usage": {}
            }
            
        if res and isinstance(res, dict) and res.get('content'):
            ranking_text = res.get('content', '')
            return {
                "model": model, 
                "status": "success", 
                "ranking": ranking_text, 
                "parsed_ranking": parse_ranking_from_text(ranking_text), 
                "usage": res.get('usage', {})
            }
            
        return {
            "model": model, 
            "status": "error", 
            "error": "Model returned empty ranking or timed out.", 
            "ranking": "Failed to get ranking from model.", 
            "parsed_ranking": [], 
            "usage": {}
        }

    tasks = [query_and_process_ranking(model) for model in council_models]
    stage2_results = await asyncio.gather(*tasks)

    print(f"Stage 2 complete: {len([r for r in stage2_results if r['status'] == 'success'])} successful rankings.")
    return stage2_results, label_to_model

async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: str = None,
    history: List[Dict[str, str]] = None,
    chairman_instruction_override: str = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.
    """
    if chairman_model is None:
        chairman_model = CHAIRMAN_MODEL

    chair_instruction = chairman_instruction_override or (
        """Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""
    )

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results if result['status'] == 'success'
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results if result['status'] == 'success'
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

{chair_instruction}"""

    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": chairman_prompt})

    # Query the chairman model with retries
    response = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"Stage 3: Retry attempt {attempt} for Chairman ({chairman_model})...")
            await asyncio.sleep(2 + random.uniform(0.5, 2.0))

        response = await query_model(chairman_model, messages)
        if response and isinstance(response, dict) and response.get('content'):
            break
        elif response and isinstance(response, dict) and "error" in response:
            print(f"Stage 3: Attempt {attempt} failed: {response['error']}")

    if response is None or not isinstance(response, dict) or not response.get('content'):
        error_msg = response.get("error") if (response and isinstance(response, dict) and "error" in response) else "Unable to generate final synthesis after multiple attempts."
        return {
            "model": chairman_model,
            "response": f"Error: {error_msg} Please try again or check your API credits.",
            "usage": {}
        }

    return {
        "model": chairman_model,
        "response": response.get('content', ''),
        "usage": response.get('usage', {})
    }

def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """Parse the FINAL RANKING section from the model's response."""
    if not ranking_text:
        return []
        
    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches

def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Calculate aggregate rankings across all models."""
    from collections import defaultdict

    model_positions = defaultdict(list)

    for ranking in stage2_results:
        if ranking['status'] != 'success':
            continue
            
        ranking_text = ranking['ranking']
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    aggregate.sort(key=lambda x: x['average_rank'])
    return aggregate

async def generate_conversation_title(user_query: str) -> str:
    """Generate a short title for a conversation."""
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
Question: {user_query}
Title:"""

    messages = [{"role": "user", "content": title_prompt}]
    response = await query_model("mistralai/mistral-7b-instruct:free", messages, timeout=30.0)

    if not response or not isinstance(response, dict) or not response.get('content'):
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip().strip('"\'')
    if len(title) > 50:
        title = title[:47] + "..."

    return title

async def run_full_council(
    user_query: str,
    council_models: List[str] = None,
    chairman_model: str = None,
    pricing_map: Optional[Dict[str, Dict[str, Any]]] = None,
    history: List[Dict[str, str]] = None
) -> Tuple[List, List, Dict, Dict]:
    """Run the complete 3-stage council process."""
    stage1_results = await stage1_collect_responses(user_query, council_models, history)

    if not any(r['status'] == 'success' for r in stage1_results):
        return stage1_results, [], {
            "model": "error",
            "response": "All models failed to respond in Stage 1. Check logs for details.",
            "usage": {}
        }, {}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results, council_models)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_model,
        history
    )

    total_cost = 0.0
    if pricing_map:
        for res in stage1_results:
            if res['status'] == 'success' and res['model'] in pricing_map:
                res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[res['model']])
                total_cost += res['cost']

        for res in stage2_results:
            if res['status'] == 'success' and res['model'] in pricing_map:
                res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[res['model']])
                total_cost += res['cost']

        mid = stage3_result['model']
        if mid in pricing_map:
            stage3_result['cost'] = calculate_cost(stage3_result.get('usage', {}), pricing_map[mid])
            total_cost += stage3_result['cost']

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "total_cost": round(total_cost, 6)
    }

    return stage1_results, stage2_results, stage3_result, metadata
