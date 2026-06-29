# ============================================================
# LANGCHAIN — STEP 11: Agents (create_agent — v1)
# Goal: Build an agent that reasons, calls tools, and answers.
#       Understand the ReAct loop, multi-tool queries, and
#       viewing internal agent steps.
# Run:  python step_11_agents.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: MODEL vs AGENT ───────────────────────────────────────────────────
#
# MODEL = smart person locked in a room. Answers from memory only.
# AGENT = smart person with phone, calculator, Google. Can DO things.
#
# The ReAct Loop:
#   1. REASON: "I need weather info for Tokyo"
#   2. ACT:    Call get_weather("Tokyo")
#   3. OBSERVE: "Sunny, 25°C"
#   4. REASON: "I have the answer now"
#   5. RESPOND: "The weather in Tokyo is 25°C and sunny"
#
# The agent repeats REASON → ACT → OBSERVE until it has enough info.


# ── DEFINE TOOLS ──────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {
        "london": "Cloudy, 15°C",
        "tokyo": "Sunny, 25°C",
        "paris": "Rainy, 12°C",
        "new york": "Clear, 20°C",
    }
    return weather.get(city.lower(), f"No weather data for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 2', '10 * 5'."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"Result: {eval(expression)}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_capital(country: str) -> str:
    """Get the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo", "india": "New Delhi",
                "germany": "Berlin", "australia": "Canberra"}
    return capitals.get(country.lower(), f"Unknown capital for {country}")


# ── PART 1: CREATE THE AGENT ──────────────────────────────────────────────────

print("=" * 55)
print("PART 1: Create and invoke an agent")
print("=" * 55)

agent = create_agent(
    model,                                          # The "brain"
    tools=[get_weather, calculate, get_capital],    # The "hands"
    system_prompt="You are a helpful assistant. Be concise.",
)
# create_agent() builds a ReAct agent:
#   - Model decides if it needs tools
#   - Calls tools in a loop
#   - Stops when it has enough info to answer


# ── PART 2: BASIC INVOCATION ──────────────────────────────────────────────────

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in London?"}]
})
print(f"Answer: {result['messages'][-1].content}")
# Agent internally: reason → call get_weather("London") → read result → answer


# ── PART 3: MULTI-TOOL QUERY ──────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Multi-tool query (agent chains tools)")
print("=" * 55)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the capital of Japan and the weather there?"}]
})
print(f"Answer: {result['messages'][-1].content}")
# Agent: get_capital("Japan") → "Tokyo" → get_weather("Tokyo") → "Sunny, 25°C" → answer


# ── PART 4: MATH + NO TOOLS ──────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Math tool and no-tool queries")
print("=" * 55)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 125 * 37?"}]
})
print(f"Math: {result['messages'][-1].content}")

result = agent.invoke({
    "messages": [{"role": "user", "content": "What color is the sky?"}]
})
print(f"No tool needed: {result['messages'][-1].content}")
# Agent is smart enough to NOT call tools when unnecessary.


# ── PART 5: VIEWING AGENT'S INTERNAL STEPS ────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Agent's internal steps (full trace)")
print("=" * 55)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Weather in Paris and 50 + 75?"}]
})

for msg in result["messages"]:
    msg_type = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tools_called = [tc["name"] for tc in msg.tool_calls]
        print(f"  [{msg_type}] Calls tools: {tools_called}")
    elif hasattr(msg, "name") and msg.name:
        print(f"  [ToolMessage '{msg.name}'] → {msg.content}")
    else:
        print(f"  [{msg_type}] {msg.content[:100]}")

# You'll see:
#   [HumanMessage] Weather in Paris and 50 + 75?
#   [AIMessage] Calls tools: ['get_weather', 'calculate']
#   [ToolMessage 'get_weather'] → Rainy, 12°C
#   [ToolMessage 'calculate'] → Result: 125
#   [AIMessage] The weather in Paris is rainy at 12°C, and 50 + 75 = 125.


# ── PART 6: AGENT WITH PERSISTENCE (Memory) ──────────────────────────────────

print("\n" + "=" * 55)
print("PART 6: Agent with memory (remembers across turns)")
print("=" * 55)

from langgraph.checkpoint.memory import InMemorySaver

agent_with_memory = create_agent(
    model,
    tools=[get_weather, calculate, get_capital],
    system_prompt="You are helpful. Remember what the user tells you.",
    checkpointer=InMemorySaver(),  # Enables persistence
)

config = {"configurable": {"thread_id": "bharath-session"}}

result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "My name is Bharath, I live in Bengaluru."}]},
    config=config,
)
print(f"Turn 1: {result['messages'][-1].content}")

result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "What's my name and where do I live?"}]},
    config=config,
)
print(f"Turn 2: {result['messages'][-1].content}")
# → "Your name is Bharath and you live in Bengaluru!" (REMEMBERED!)


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. create_agent(model, tools, system_prompt) — builds a ReAct agent
# 2. Agent decides WHEN and WHICH tools to call (model reasons)
# 3. Multi-tool: agent chains tools in sequence automatically
# 4. No tool needed: agent answers from knowledge when appropriate
# 5. View steps: iterate result["messages"] to see internal reasoning
# 6. Persistence: checkpointer=InMemorySaver() + thread_id for memory
#
# NEXT: step_12_agent_streaming.py — Stream agent responses in real-time
