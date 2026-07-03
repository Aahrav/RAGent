"""LangGraph agent nodes and definitions."""

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
    
    # Return the response as a dictionary matching the AgentState schema.
    # The `add_messages` reducer will automatically append it to the list.
    return {"messages": [response]}
