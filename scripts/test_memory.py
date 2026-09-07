"""Test script for Conversational Short-Term Memory."""

import uuid
import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.rag.pipeline import query
from src.storage.history import get_chat_history
from src.storage.cache import get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_test():
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    print(f"=== Starting Memory Test with Session ID: {session_id} ===")
    
    # 1. Ask the first question
    q1 = "What is the Project Apollo?"
    print(f"\nUser: {q1}")
    r1 = query(user_input=q1, use_agent=False, session_id=session_id)
    print(f"AI: {r1.answer}")
    
    # 2. Ask a follow-up question that requires history
    q2 = "Who is leading it?"
    print(f"\nUser: {q2}")
    r2 = query(user_input=q2, use_agent=False, session_id=session_id)
    print(f"AI: {r2.answer}")
    
    # 3. Print the chat history to verify it was stored correctly
    print("\n=== Chat History in Redis ===")
    history = get_chat_history(session_id)
    for msg in history:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
        
    print("\n=== Test Complete ===")
    
if __name__ == "__main__":
    run_test()
