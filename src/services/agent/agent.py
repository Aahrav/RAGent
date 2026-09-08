"""LangGraph agent nodes and definitions."""

import json
import uuid

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from src.ml.llm import get_llm
from src.services.agent.state import AgentState
from src.services.agent.tools import calculator, get_current_datetime, rag_search, web_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 1. Gather all the tools we created in Issue #16
tools = [rag_search, web_search, calculator, get_current_datetime]

# 2. Get the LLM (from Phase 1) and bind the tools to it
llm = get_llm()
model_with_tools = llm.bind_tools(tools)

# 3. Create the prebuilt ToolNode which automatically executes any tool requested by the LLM
tool_node = ToolNode(tools)

def call_model(state: AgentState) -> dict:
    """The node responsible for invoking the LLM (the 'brain').
    
    It reads the message history, passes it to the LLM, and returns the response.
    If the LLM decides to call a tool, the response will contain an AIMessage
    with `tool_calls` attached.
    """
    logger.debug("Agent reasoning step started", extra={"message_count": len(state["messages"])})
    
    # We pass the full conversation history to the model
    response = model_with_tools.invoke(state["messages"])
    
    # --- HOTFIX FOR LOCAL OLLAMA JSON TOOL CALLING ---
    # Open-source models often return the tool call as raw JSON inside a conversational preamble
    # instead of populating the native LangChain `tool_calls` attribute.
    if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None) and response.content:
        content_str = str(response.content).strip()
        
        # Look for a JSON object containing "name" and "arguments"
        import re
        # Find the last JSON block that looks like a tool call
        match = re.search(r'(\{[\s\S]*?"name"[\s\S]*?"arguments"[\s\S]*?\})', content_str)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if "name" in parsed and "arguments" in parsed:
                    logger.info("Intercepted raw JSON tool call from Ollama", extra={"tool": parsed["name"]})
                    
                    # Convert arguments string to dict if the model double-encoded it
                    args = parsed["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"query": args} # Fallback
                            
                    response.tool_calls = [{
                        "name": parsed["name"],
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }]
                    # Clear content so it doesn't get rendered to user, or keep preamble
                    response.content = ""
            except json.JSONDecodeError:
                pass
    # -------------------------------------------------
    
    # Return the response as a dictionary matching the AgentState schema.
    # The `add_messages` reducer will automatically append it to the list.
    return {"messages": [response]}
