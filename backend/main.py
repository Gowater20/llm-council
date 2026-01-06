"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
import uuid
import json
import asyncio
import time

from . import storage
from .openrouter import fetch_available_models, calculate_cost
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Allow * for external tools like Cursor
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
    """Request to send a message in a conversation (Internal App API)."""
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
        elif msg.get("role") == "assistant":
            # Check if this is a Council message structure or a standard OpenAI message
            if "stage3" in msg:
                response = msg["stage3"].get("response", "")
                if response and not response.startswith("Error:"):
                    history.append({"role": "assistant", "content": response})
            elif "content" in msg:
                history.append({"role": "assistant", "content": msg["content"]})

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
    
    # Check if there is already a system message in the input messages (for OpenAI API)
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        history.insert(0, system_hint)
    else:
        # If input messages have system prompt, we might want to respect it,
        # but the Council logic relies on the specific "Council Member" persona.
        # For now, let's prepend our hint anyway, or maybe append it to the existing system prompt.
        # Simpler approach: Prepend our Council context instructions as a system message.
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

    # Get prompt settings
    prompt_settings = get_prompt_by_id(request.prompt_id or "default")
    chairman_instruction = prompt_settings["chairman_instruction"] if prompt_settings else None

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content,
        request.council_models,
        request.chairman_model,
        pricing_map,
        history,
        chairman_instruction_override=chairman_instruction
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
            is_first_message = len(conversation["messages"]) == 0 
            
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
            stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results, request.chairman_model, history, chairman_instruction_override=chairman_instruction)
            
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

class OpenAIMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[List[OpenAIMessage]] = None
    input: Optional[List[OpenAIMessage]] = None # Cursor sometimes sends 'input' instead of 'messages'
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None # Support new field
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # Allow tools/functions even if we don't support them yet, to avoid 422
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream_options: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Optional[str] = "stop"

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class OpenAIChoiceDelta(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChoiceDelta]


# --- Routes ---

# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # print(f"Request: {request.method} {request.url}") # Reduce noise if needed
    try:
        response = await call_next(request)
        # print(f"Response: {response.status_code}")
        return response
    except Exception as e:
        print(f"Request processing failed: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}

# --- OpenAI Compatible Endpoint ---

class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "council"

class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]

@app.get("/v1/models")
async def list_openai_models():
    """
    List models in OpenAI format.
    Required by some clients (like Cursor/VS Code) to verify connectivity.
    """
    return OpenAIModelList(
        data=[
            OpenAIModel(id="council-llm", created=int(time.time()), owned_by="llm-council"),
            OpenAIModel(id="gpt-4o", created=int(time.time()), owned_by="llm-council"), # Alias
            # We can also expose the underlying OpenRouter models if we want
            # but for now let's just expose the main one
        ]
    )

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    """
    OpenAI-compatible endpoint for LLM Council.
    This allows tools like Cursor, VS Code (Continue, Cline), etc. to use the Council.
    """
    try:
        # 1. Extract the latest user query from messages
        messages = request.messages or request.input
        
        if not messages:
            print(f"DEBUG: Body received: {request.dict()}")
            raise HTTPException(status_code=400, detail="No messages or input provided")
        
        # Find the last message from user
        user_query = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_query = msg.content
                # Handle list content (multimodal) - just take the text parts
                if isinstance(user_query, list):
                    text_parts = [p.get('text', '') for p in user_query if p.get('type') == 'text']
                    user_query = " ".join(text_parts)
                elif user_query is None:
                     user_query = "" # Skip null content
                break
        
        if not user_query:
            # Fallback: if no user message found (e.g. only system), try to use the last message
            if messages:
                 last_msg = messages[-1]
                 if last_msg.content:
                      user_query = last_msg.content
                      if isinstance(user_query, list):
                            text_parts = [p.get('text', '') for p in user_query if p.get('type') == 'text']
                            user_query = " ".join(text_parts)
            
            if not user_query:
                 raise HTTPException(status_code=400, detail="No user message found")

        # 2. Prepare history from previous messages
        # We need to convert OpenAIMessage objects to dicts
        raw_messages = [m.dict() for m in messages[:-1]] # Exclude the last one which is the current query (added by council logic typically, but let's be safe)
        if messages[-1].role == "user":
             raw_messages = [m.dict() for m in messages[:-1]]
        else:
             raw_messages = [m.dict() for m in messages]

        history = prepare_history(raw_messages)

        # 3. Handle System Prompts if present in the request
        system_content = None
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break

        # 4. Run the Full Council
        # Note: We are NOT using the 'model' parameter from the request to select the Council.
        # The Council configuration is determined by the server's environment/config.
        # Ideally, we could map 'model' to different council presets if we wanted.
        
        # We create a temporary pricing map
        pricing_map = await get_pricing_map()

        # Execute!
        # Note: This is a blocking operation for the client unless we stream.
        # For the first version, we'll wait for the full response (non-streaming).
        
        if request.stream:
             # Streaming support
             async def openai_stream_generator():
                # We start only with a "role" chunk to acknowledge the request immediately
                request_id = str(uuid.uuid4())
                created_time = int(time.time())
                
                # Yield initial chunk to establish connection
                initial_chunk = OpenAIChatCompletionChunk(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        OpenAIChoiceDelta(
                            index=0,
                            delta={"role": "assistant", "content": ""},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {initial_chunk.json()}\n\n"

                # Run council in background
                task = asyncio.create_task(run_full_council(
                    user_query,
                    council_models=None, # Use default from config
                    chairman_model=None, # Use default from config
                    pricing_map=pricing_map,
                    history=history
                ))

                # Wait for task while sending keep-alive comments
                while not task.done():
                    # SSE comment to keep connection open (ignored by clients)
                    yield ": keep-alive\n\n" 
                    await asyncio.sleep(0.5)
                
                # Get result
                try:
                    stage1, stage2, stage3, meta = await task
                except Exception as e:
                     # If task failed, yield error
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    return

                final_content = stage3.get("response", "")
                
                # Send the content in small chunks to simulate streaming (or just one big chunk)
                chunk_size = 100
                for i in range(0, len(final_content), chunk_size):
                    chunk_text = final_content[i:i+chunk_size]
                    chunk = OpenAIChatCompletionChunk(
                        id=request_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            OpenAIChoiceDelta(
                                index=0,
                                delta={"content": chunk_text},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.json()}\n\n"
                    await asyncio.sleep(0.01) # Tiny sleep to be polite

                # Send finish
                end_chunk = OpenAIChatCompletionChunk(
                        id=request_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            OpenAIChoiceDelta(
                                index=0,
                                delta={},
                                finish_reason="stop"
                            )
                        ]
                    )
                yield f"data: {end_chunk.json()}\n\n"
                yield "data: [DONE]\n\n"

             return StreamingResponse(
                openai_stream_generator(),
                media_type="text/event-stream"
            )

        else:
            # Non-streaming response
            stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                user_query,
                council_models=None, # Use default from config
                chairman_model=None, # Use default from config
                pricing_map=pricing_map,
                history=history
            )

            final_response_text = stage3_result.get("response", "")
            
            # Simple token estimation
            completion_tokens = len(final_response_text) // 4
            prompt_tokens = len(user_query) // 4
            
            return OpenAIChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model, # Echo back the requested model
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(
                            role="assistant",
                            content=final_response_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )

    except Exception as e:
        print(f"Error in OpenAI API adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

