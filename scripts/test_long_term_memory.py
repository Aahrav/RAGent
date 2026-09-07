"""Test script for Long-Term Memory (User Facts).

This script verifies that the system extracts personal facts from user messages
in the background, saves them to Qdrant, and successfully retrieves them in a
completely separate chat session.
"""

import time
import uuid

from src.services.rag import pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_long_term_memory():
    # We use a single user_id to represent our user across multiple sessions
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    
    print("=== Testing Long-Term Memory Extraction ===")
    print(f"User ID: {user_id}")
    print("\n--- Session 1: Fact Ingestion ---")
    session_id_1 = str(uuid.uuid4())
    
    message_1 = "Hi, I'm just setting up my profile. I work in the Boston office as a backend engineer, and I absolutely love programming in Rust."
    print(f"User: {message_1}")
    
    # 1. Send the message. The pipeline will trigger extraction in the background.
    result_1 = pipeline.query(
        user_input=message_1,
        use_agent=False,
        session_id=session_id_1,
        user_id=user_id,
    )
    print(f"Assistant: {result_1.answer}")
    
    # Wait a few seconds for the background extraction thread to finish
    # invoking the LLM and saving to Qdrant.
    print("\n[Waiting 5 seconds for background extraction to complete...]")
    time.sleep(5)
    
    print("\n--- Session 2: Fact Retrieval ---")
    session_id_2 = str(uuid.uuid4())
    print(f"Starting a completely new session (Session ID: {session_id_2})")
    
    message_2 = "Can you check my profile and tell me which office I'm in and what my favorite language is?"
    print(f"User: {message_2}")
    
    # 2. Ask a question relying entirely on the extracted facts from Session 1
    result_2 = pipeline.query(
        user_input=message_2,
        use_agent=False,
        session_id=session_id_2,
        user_id=user_id,
    )
    print(f"Assistant: {result_2.answer}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_long_term_memory()
