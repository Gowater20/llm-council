"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model, calculate_cost
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL


async def stage1_collect_responses(user_query: str, council_models: List[str] = None) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        council_models: Optional list of specific models to query

    Returns:
        List of dicts with 'model', 'response', and 'usage' keys
    """
    if council_models is None:
        council_models = COUNCIL_MODELS

    messages = [{"role": "user", "content": user_query}]

    print(f"Stage 1: Querying {len(council_models)} models for query: '{user_query}'")
    # Query all models in parallel
    responses = await query_models_parallel(council_models, messages)

    # Format results
    stage1_results = []
    for model in council_models:
        response = responses.get(model)
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get('content', ''),
                "usage": response.get('usage', {}),
                "status": "success"
            })
        else:
            stage1_results.append({
                "model": model,
                "response": "Failed to get response from model.",
                "usage": {},
                "status": "error",
                "error": "Timeout or API error"
            })

    print(f"Stage 1 complete: {len([r for r in stage1_results if r['status'] == 'success'])} successful responses.")
    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: List[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        council_models: Optional list of models to perform the ranking

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    if council_models is None:
        council_models = COUNCIL_MODELS

    # Create anonymized labels ONLY for successful Stage 1 responses
    successful_stage1 = [r for r in stage1_results if r['status'] == 'success']
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

    print(f"Stage 2: Asking {len(council_models)} models to rank {len(stage1_results)} responses")
    # Get rankings from all council models in parallel
    responses = await query_models_parallel(council_models, messages)

    # Format results
    stage2_results = []
    for model in council_models:
        response = responses.get(model)
        if response is not None:
            ranking_text = response.get('content', '')
            stage2_results.append({
                "model": model,
                "ranking": ranking_text,
                "parsed_ranking": parse_ranking_from_text(ranking_text),
                "usage": response.get('usage', {}),
                "status": "success"
            })
        else:
            stage2_results.append({
                "model": model,
                "ranking": "Failed to get ranking from model.",
                "parsed_ranking": [],
                "usage": {},
                "status": "error",
                "error": "Timeout or API error"
            })

    print(f"Stage 2 complete: {len([r for r in stage2_results if r['status'] == 'success'])} successful rankings.")
    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: str = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        chairman_model: Optional specific model to act as Chairman

    Returns:
        Dict with 'model', 'response', and 'usage' keys
    """
    if chairman_model is None:
        chairman_model = CHAIRMAN_MODEL

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    print(f"Stage 3: Asking Chairman ({chairman_model}) to synthesize")
    # Query the chairman model
    response = await query_model(chairman_model, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": chairman_model,
            "response": "Error: Unable to generate final synthesis.",
            "usage": {}
        }

    return {
        "model": chairman_model,
        "response": response.get('content', ''),
        "usage": response.get('usage', {})
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use a free model for title generation
    response = await query_model("mistralai/mistral-7b-instruct:free", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    council_models: List[str] = None,
    chairman_model: str = None,
    pricing_map: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        council_models: Optional list of specific models to query
        chairman_model: Optional specific model to act as Chairman
        pricing_map: Optional mapping of model ID to pricing info

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, council_models)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
            "usage": {}
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results, council_models)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_model
    )

    # Calculate costs if pricing_map is provided
    total_cost = 0.0
    if pricing_map:
        # Cost from Stage 1
        for res in stage1_results:
            model_id = res['model']
            if model_id in pricing_map:
                res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[model_id])
                total_cost += res['cost']

        # Cost from Stage 2
        for res in stage2_results:
            model_id = res['model']
            if model_id in pricing_map:
                res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[model_id])
                total_cost += res['cost']

        # Cost from Stage 3
        model_id = stage3_result['model']
        if model_id in pricing_map:
            stage3_result['cost'] = calculate_cost(stage3_result.get('usage', {}), pricing_map[model_id])
            total_cost += stage3_result['cost']

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "total_cost": round(total_cost, 6)
    }

    return stage1_results, stage2_results, stage3_result, metadata
