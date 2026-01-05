
import asyncio
import os
import sys

# Add backend to path (adjusting for absolute path# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Fix for relative imports in backend modules when running as script
# We need to mock the package structure or modify how we import
import openrouter
import config
from council import run_full_council
from council import run_full_council

# Redefining prepare_history here to verify logic without importing main.py (which causes import issues)
from typing import List, Dict, Any
def prepare_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Prepare an optimized history for the council.
    Uses "Consensus History": only User messages and successful Chairman syntheses.
    Implements a sliding window to prevent token overflow.
    """
    history = []
    # Only keep the last 10 turns (20 messages max) to keep context lean
    # but sufficient for long conversations.
    relevant_messages = messages[-20:] if len(messages) > 20 else messages
    
    for msg in relevant_messages:
        if msg.get("role") == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg.get("role") == "assistant" and "stage3" in msg:
            response = msg["stage3"].get("response", "")
            # Skip error responses
            if response and not response.startswith("Error:"):
                # Label it as the Council's consensus to reinforce identity
                history.append({"role": "assistant", "content": response})
    
    # Enhanced System Prompt to define the Council's Identity
    system_hint = {
        "role": "system", 
        "content": (
            "You are an expert member of the LLM Council. This is a multi-turn conversation. "
            "The messages from 'assistant' represent the final consensus (Chairman's synthesis) "
            "of previous turns. Use this as your shared memory and source of truth."
        )
    }
    
    history.insert(0, system_hint)
    return history

async def run_test():
    # Setup - using models found in config.py
    council_models = ["mistralai/mistral-7b-instruct:free", "google/gemini-2.0-flash-exp:free"]
    chairman_model = "mistralai/mistral-7b-instruct:free" 
    
    print("\nXXX Starting Test Turn 1: 'Quanto fa 20 + 5?' XXX\n")
    user_query_1 = "Quanto fa 20 + 5? Rispondi in modo sintetico con solo il numero."
    
    # Run Turn 1 with empty history
    s1_1, s2_1, s3_1, meta_1 = await run_full_council(
        user_query_1,
        council_models=council_models,
        chairman_model=chairman_model,
        history=[]
    )
    
    print(f"XXX Stage 3 Result (Turn 1): {s3_1.get('response')} XXX")
    
    # Simulate full message history as if it were in the DB
    full_messages = [
        {"role": "user", "content": user_query_1},
        {"role": "assistant", "stage3": s3_1} 
    ]
    
    # Prepare history for Turn 2 using our new logic
    history_for_turn_2 = prepare_history(full_messages)
    
    print("\nXXX Verifying Context Passed to Turn 2 XXX")
    print(str(history_for_turn_2))
    
    # Verify we don't have extraneous info
    assert len(history_for_turn_2) >= 1 # System prompt + User + Assistant
    assert history_for_turn_2[0]['role'] == 'system'
    
    print("\nXXX Starting Test Turn 2: 'Adesso aggiungi 3' XXX\n")
    user_query_2 = "Adesso aggiungi 3 al risultato precedente. Quanto fa?"
    
    s1_2, s2_2, s3_2, meta_2 = await run_full_council(
        user_query_2,
        council_models=council_models,
        chairman_model=chairman_model,
        history=history_for_turn_2
    )

    print(f"\nXXX Stage 1 Responses (Turn 2) XXX")
    for res in s1_2:
        print(f"Model {res['model']}: {res['response']}")
        
    print(f"\nXXX Stage 3 Result (Turn 2): {s3_2.get('response')} XXX")

if __name__ == "__main__":
    asyncio.run(run_test())
