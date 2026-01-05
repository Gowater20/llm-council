"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .openrouter import fetch_available_models, calculate_cost
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for models
MODELS_CACHE = {"data": None, "timestamp": 0}
CACHE_TTL = 3600  # 1 hour


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    council_models: List[str] = None
    chairman_model: str = None
    prompt_id: str = None


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/models")
async def list_available_models():
    """List all available models from OpenRouter (with caching)."""
    import time
    now = time.time()
    
    if MODELS_CACHE["data"] and (now - MODELS_CACHE["timestamp"] < CACHE_TTL):
        return MODELS_CACHE["data"]
    
    models = await fetch_available_models()
    if models:
        MODELS_CACHE["data"] = models
        MODELS_CACHE["timestamp"] = now
        return models
    
    return []


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    storage.delete_conversation(conversation_id)
    return {"status": "success", "message": "Conversation deleted"}


async def get_pricing_map():
    """Get a mapping of model ID to pricing info."""
    models = await list_available_models()
    return {m['id']: m['pricing'] for m in models if 'id' in m and 'pricing' in m}


try:
    from .prompts import load_prompts, add_prompt, delete_prompt, get_prompt_by_id
except ImportError:
    from prompts import load_prompts, add_prompt, delete_prompt, get_prompt_by_id

# ... (existing code)

class PromptModel(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    chairman_instruction: str

@app.get("/api/prompts")
async def get_prompts():
    return load_prompts()

@app.post("/api/prompts")
async def create_prompt(prompt: PromptModel):
    add_prompt(prompt.dict())
    return {"status": "success"}

@app.delete("/api/prompts/{prompt_id}")
async def remove_prompt(prompt_id: str):
    delete_prompt(prompt_id)
    return {"status": "success"}

# Modify prepare_history to accept system_content override
def prepare_history(messages: List[Dict[str, Any]], system_content: str = None) -> List[Dict[str, str]]:
    """
    Prepare an optimized history for the council.
    Uses "Consensus History" and a sliding window.
    """
    history = []
    # Only keep the last 10 turns (20 messages max)
    relevant_messages = messages[-20:] if len(messages) > 20 else messages
    
    for msg in relevant_messages:
        if msg.get("role") == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg.get("role") == "assistant" and "stage3" in msg:
            response = msg["stage3"].get("response", "")
            if response and not response.startswith("Error:"):
                history.append({"role": "assistant", "content": response})
    
    # Use provided system content or default
    final_system_content = system_content or (
        "You are an expert member of the LLM Council. This is a multi-turn conversation. "
        "The messages from 'assistant' represent the final consensus (Chairman's synthesis) "
        "of previous turns. Use this as your shared memory and source of truth."
    )

    system_hint = {
        "role": "system", 
        "content": final_system_content
    }
    
    history.insert(0, system_hint)
    return history


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Get pricing map for cost calculation
    pricing_map = await get_pricing_map()

    # Prepare history for context
    history = prepare_history(conversation["messages"])

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content,
        request.council_models,
        request.chairman_model,
        pricing_map,
        history
    )
    
    # Store pricing map in metadata for the frontend
    metadata["pricing_map"] = pricing_map

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Verify conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 1. Add user message
    storage.add_user_message(conversation_id, request.content)

    # Get pricing map
    pricing_map = await get_pricing_map()

    # Get prompt settings
    prompt_settings = get_prompt_by_id(request.prompt_id or "default")
    system_prompt = prompt_settings["system_prompt"] if prompt_settings else None
    chairman_instruction = prompt_settings["chairman_instruction"] if prompt_settings else None

    # Prepare history for context with custom system prompt
    history = prepare_history(conversation["messages"], system_content=system_prompt)

    async def event_generator():
        try:
            # Check if this is the first message (after adding user message)
            is_first_message = len(conversation["messages"]) == 0 # This will be 0 if it was truly the first message before adding the user message.
            # If the conversation was empty before adding the user message, it's the first message.
            # The `conversation` object loaded here is before the `add_message` call, so its `messages` list reflects the state before the current user message.
            # So, if `len(conversation["messages"])` is 0, it means the user message just added is the first one.
            
            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content, request.council_models, history)
            
            if not stage1_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'All models failed to respond in Stage 1. Please try again.'})}\n\n"
                return

            # Calculate cost for stage 1
            if pricing_map:
                for res in stage1_results:
                    if res['model'] in pricing_map:
                        res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[res['model']])

            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results, request.council_models)
            
            # Calculate cost for stage 2
            if pricing_map:
                for res in stage2_results:
                    if res['model'] in pricing_map:
                        res['cost'] = calculate_cost(res.get('usage', {}), pricing_map[res['model']])

            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results, request.chairman_model, history)
            
            # Calculate cost for stage 3
            if pricing_map and stage3_result['model'] in pricing_map:
                stage3_result['cost'] = calculate_cost(stage3_result.get('usage', {}), pricing_map[stage3_result['model']])

            # Calculate total cost
            total_cost = sum(r.get('cost', 0) for r in stage1_results) + \
                         sum(r.get('cost', 0) for r in stage2_results) + \
                         stage3_result.get('cost', 0)

            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'metadata': {'total_cost': round(total_cost, 6)}})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
