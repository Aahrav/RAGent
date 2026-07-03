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
    # Llama3 often returns the tool call as raw JSON in the content string
    # instead of populating the native LangChain `tool_calls` attribute.
    if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None) and response.content:
        content_str = str(response.content).strip()
        # Sometimes the LLM wraps it in a markdown block
        if content_str.startswith("```json"):
            content_str = content_str[7:].strip()
        if content_str.endswith("```"):
            content_str = content_str[:-3].strip()
            
        if content_str.startswith("{") and content_str.endswith("}"):
            try:
                parsed = json.loads(content_str)
                if "name" in parsed and "arguments" in parsed:
                    logger.info("Intercepted raw JSON tool call from Ollama", extra={"tool": parsed["name"]})
                    response.tool_calls = [{
                        "name": parsed["name"],
                        "args": parsed["arguments"],
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }]
                    response.content = ""
            except json.JSONDecodeError:
                pass
    # -------------------------------------------------
    
    # Return the response as a dictionary matching the AgentState schema.
    # The `add_messages` reducer will automatically append it to the list.
    return {"messages": [response]}
