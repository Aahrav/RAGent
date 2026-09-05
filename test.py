import sys
from src.services.rag.pipeline import query
from src.config import get_settings

def test():
    # 1. Ask a question and start a session
    session_id = "test-session-123"
    print("User: My name is Aahrav and my favorite color is blue.")
    
    # We force the use of the Agent so we know it hits the memory saver
    result1 = query(
        user_input="My name is Aahrav and my favorite color is blue.",
        use_agent=True,
        session_id=session_id
    )
    print(f"Agent: {result1.answer}\n")
    
    # 2. Ask a follow-up question
    print("User: What is my name and my favorite color?")
    result2 = query(
        user_input="What is my name and my favorite color?",
        use_agent=True,
        session_id=session_id
    )
    print(f"Agent: {result2.answer}\n")
    
if __name__ == "__main__":
    test()