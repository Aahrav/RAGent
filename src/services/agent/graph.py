"""Agent Graph.

This module defines the StateGraph for the LangGraph agent, which creates
a cyclic execution loop allowing the LLM to reason, call tools, observe the
results, and iterate until it finds the final answer.
"""

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition

from src.services.agent.agent import call_model, tool_node
from src.services.agent.state import AgentState

# ============================================================================
# GRAPH COMPILATION
# ============================================================================

# 1. Initialize the StateGraph with our state schema
workflow = StateGraph(AgentState)

# 2. Add our two nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 3. Define the edges (the flow of execution)
# The loop always starts with the agent reasoning
workflow.add_edge(START, "agent")

# After the agent thinks, we use a conditional edge to decide what happens next.
# `tools_condition` automatically checks if the LLM's response has `tool_calls`.
# - If it does, we route to the "tools" node.
# - If it doesn't, we route to END (the agent gave its final answer).
workflow.add_conditional_edges("agent", tools_condition)

# After the tools execute, they MUST return their results back to the agent
# so the agent can read the results and decide if it needs to do anything else.
workflow.add_edge("tools", "agent")

# 4. Compile the graph into an executable runnable
agent_app = workflow.compile()
