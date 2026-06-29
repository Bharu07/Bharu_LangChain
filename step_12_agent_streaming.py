# ============================================================
# LANGCHAIN — STEP 12: Agent Streaming and Structured Output
# Goal: Stream agent responses in real-time (see thinking + typing),
#       and get structured Pydantic responses from agents.
# Run:  python step_12_agent_streaming.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
def search_knowledge(topic: str) -> str:
    """Search for information about a topic."""
    knowledge = {
        "python": "Python is a high-level language created by Guido van Rossum in 1991.",
        "rust": "Rust is a systems language focused on safety, created by Mozilla in 2010.",
        "javascript": "JavaScript was created in 10 days by Brendan Eich in 1995.",
    }
    return knowledge.get(topic.lower(), f"Information about {topic}: widely used technology.")

@tool
def get_stats(language: str) -> str:
    """Get usage statistics for a programming language."""
    stats = {
        "python": "Used by 15M+ developers. #1 in AI/ML. Growing 25% yearly.",
        "rust": "Used by 3M+ developers. #1 in system programming satisfaction.",
        "javascript": "Used by 20M+ developers. Runs on 98% of websites.",
    }
    return stats.get(language.lower(), f"Stats for {language}: popular and growing.")


# ── PART 1: STREAMING MODE "values" (step-by-step) ───────────────────────────

print("=" * 55)
print("PART 1: stream_mode='values' (see each step)")
print("=" * 55)

agent = create_agent(
    model, tools=[search_knowledge, get_stats],
    system_prompt="You are a tech researcher. Use tools to find info. Be concise.",
)

print("Question: Tell me about Python\n")

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me about Python"}]},
    stream_mode="values",
):
    # stream_mode="values" gives full state after EACH node execution
    latest = chunk["messages"][-1]
    msg_type = type(latest).__name__

    if hasattr(latest, "tool_calls") and latest.tool_calls:
        print(f"  🔧 Agent calls: {[tc['name'] for tc in latest.tool_calls]}")
    elif hasattr(latest, "name") and latest.name:
        print(f"  📋 Tool '{latest.name}' returned: {latest.content[:80]}")
    elif latest.content and msg_type == "AIMessage":
        print(f"  ✅ Final answer: {latest.content[:150]}")


# ── PART 2: STREAMING MODE "messages" (token-by-token) ────────────────────────

print("\n" + "=" * 55)
print("PART 2: stream_mode='messages' (typing effect)")
print("=" * 55)

print("Answer: ", end="", flush=True)
for msg, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is Rust?"}]},
    stream_mode="messages",
):
    # stream_mode="messages" yields individual tokens as they're generated
    if msg.content and metadata.get("langgraph_node") == "agent":
        print(msg.content, end="", flush=True)
print()
# This creates the "typing effect" like ChatGPT!


# ── PART 3: STRUCTURED OUTPUT (Agent returns Pydantic object) ─────────────────

print("\n" + "=" * 55)
print("PART 3: Agent with structured output")
print("=" * 55)

class ResearchReport(BaseModel):
    """A structured research report."""
    topic: str = Field(description="The researched topic")
    summary: str = Field(description="Brief summary of findings")
    key_facts: List[str] = Field(description="Key facts discovered")
    confidence: float = Field(description="Confidence score 0.0-1.0")

structured_agent = create_agent(
    model,
    tools=[search_knowledge, get_stats],
    system_prompt="Research topics using tools. Provide comprehensive reports.",
    response_format=ResearchReport,  # Forces structured response
)

result = structured_agent.invoke({
    "messages": [{"role": "user", "content": "Research JavaScript for me"}]
})

# Access the structured response:
report = result["structured_response"]
print(f"Topic: {report.topic}")
print(f"Summary: {report.summary}")
print(f"Key facts: {report.key_facts}")
print(f"Confidence: {report.confidence}")


# ── PART 4: COMPARING STREAM MODES ───────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Stream modes comparison")
print("=" * 55)

print("""
stream_mode="values":
  • Shows full state after each step (node execution)
  • Good for: debugging, showing agent progress to user
  • You see: "Agent calls tool X" → "Tool returned Y" → "Final answer"

stream_mode="messages":
  • Shows individual tokens as generated
  • Good for: ChatGPT-style typing effect in UI
  • You see: word-by-word output in real-time

stream_mode="updates":
  • Shows only the CHANGES at each step
  • Good for: lightweight monitoring, logging
""")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. stream_mode="values" — see each step (tool calls, results, answer)
# 2. stream_mode="messages" — token-by-token typing effect
# 3. response_format=PydanticModel — agent returns structured data
# 4. result["structured_response"] — access the typed response
# 5. Agent uses tools THEN formats the structured output from findings
#
# NEXT: step_13_middleware.py — Intercepting agent behavior
