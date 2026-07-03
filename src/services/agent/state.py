"""Agent state TypedDict for LangGraph."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent during execution.
    
    Attributes:
        messages: The conversation history in the current execution loop.
            The `add_messages` reducer ensures that new messages (from the user,
            the LLM, or tools) are appended to the list rather than overwriting it.
    """
    messages: Annotated[list[BaseMessage], add_messages]
