# ============================================================
# LANGCHAIN — STEP 13: Middleware (v1 — Intercept Agent Behavior)
# Goal: Use middleware to handle tool errors, swap models dynamically,
#       and change prompts at runtime — without touching agent code.
# Run:  python step_13_middleware.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain.agents.middleware import (
    wrap_tool_call,
    wrap_model_call,
    dynamic_prompt,
)

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── CONCEPT: WHAT IS MIDDLEWARE? ──────────────────────────────────────────────
#
# Middleware = code that runs BETWEEN steps in the agent loop.
# It can inspect, modify, or block what happens.
#
#   [User input]
#        ↓
#   [wrap_model_call] ← intercept BEFORE/AFTER model thinks
#        ↓
#   [Model decides]
#        ↓
#   [wrap_tool_call]  ← intercept BEFORE/AFTER tool executes
#        ↓
#   [Tool runs]
#        ↓
#   [Model reads result, decides again]
#        ↓
#   [Final answer]
#
# You NEVER change the agent code. Just plug in middleware.


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather = {"london": "Cloudy, 15°C", "tokyo": "Sunny, 25°C"}
    return weather.get(city.lower(), f"No data for {city}")

@tool
def risky_divide(a: float, b: float) -> str:
    """Divide two numbers. Will crash if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return f"Result: {a / b}"


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE 1: @wrap_tool_call — Handle tool errors gracefully
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("MIDDLEWARE 1: @wrap_tool_call (error handling)")
print("=" * 55)

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch tool exceptions and return a friendly error message."""
    try:
        return handler(request)
        # handler(request) = run the actual tool normally
    except Exception as e:
        # Tool crashed → return error message instead of crashing agent
        return ToolMessage(
            content=f"Tool failed: {str(e)}. Please try a different approach.",
            tool_call_id=request.tool_call["id"],
        )

agent_safe = create_agent(
    model,
    tools=[risky_divide, get_weather],
    system_prompt="You are helpful. If a tool fails, explain the error to the user.",
    middleware=[handle_tool_errors],
)

result = agent_safe.invoke({
    "messages": [{"role": "user", "content": "What is 10 divided by 0?"}]
})
print(f"Answer: {result['messages'][-1].content}")
# Without middleware: agent CRASHES with ValueError
# With middleware: agent gets "Tool failed: Cannot divide by zero" and responds nicely


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE 2: @wrap_model_call — Dynamic model selection
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MIDDLEWARE 2: @wrap_model_call (model selection)")
print("=" * 55)

# CONCEPT: Use cheap model for simple queries, expensive for complex ones.
# The middleware checks message count to estimate complexity.

@wrap_model_call
def smart_model_routing(request, handler):
    """Route to different models based on complexity."""
    messages = request.messages
    # Simple heuristic: long conversations = complex = use better model
    if len(messages) > 6:
        print("  [Middleware] Complex query → using powerful model")
        # In production: return handler(request, model="openai:gpt-4o")
    else:
        print("  [Middleware] Simple query → using standard model")
        # In production: return handler(request, model="openai:gpt-4o-mini")
    return handler(request)  # For demo, use same model

agent_smart = create_agent(
    model,
    tools=[get_weather],
    system_prompt="Be helpful.",
    middleware=[smart_model_routing],
)

result = agent_smart.invoke({
    "messages": [{"role": "user", "content": "Weather in Tokyo?"}]
})
print(f"Answer: {result['messages'][-1].content}")


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE 3: @dynamic_prompt — Change system prompt at runtime
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MIDDLEWARE 3: @dynamic_prompt (personalized prompts)")
print("=" * 55)

# Change the agent's personality based on user context:
user_profiles = {
    "beginner": "Explain everything simply, use analogies, avoid jargon.",
    "expert": "Be technical and precise. Skip basics.",
}

@dynamic_prompt
def personalized_prompt(context):
    """Adjust prompt based on user's skill level."""
    level = context.get("user_level", "beginner") if context else "beginner"
    style = user_profiles.get(level, user_profiles["beginner"])
    return f"You are a helpful coding assistant. {style}"

agent_personalized = create_agent(
    model,
    tools=[get_weather],
    system_prompt="You are helpful.",  # Will be overridden by middleware
    middleware=[personalized_prompt],
)

# Same question, different user levels would get different explanations!
result = agent_personalized.invoke({
    "messages": [{"role": "user", "content": "What is a REST API?"}]
})
print(f"Answer: {result['messages'][-1].content[:150]}...")


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE 4: COMBINING MULTIPLE MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MIDDLEWARE 4: Combining multiple middleware")
print("=" * 55)

# Stack multiple middleware — they run in order:
agent_full = create_agent(
    model,
    tools=[risky_divide, get_weather],
    system_prompt="Be helpful and concise.",
    middleware=[
        handle_tool_errors,     # 1. Catch tool crashes
        smart_model_routing,    # 2. Route to right model
    ],
)

result = agent_full.invoke({
    "messages": [{"role": "user", "content": "Divide 100 by 0, then get weather in London"}]
})
print(f"Answer: {result['messages'][-1].content}")


# ── MIDDLEWARE HOOKS SUMMARY ──────────────────────────────────────────────────
#
# Hook              | When it runs              | Use case
# ------------------|---------------------------|----------------------------------
# @wrap_tool_call   | Around each tool call     | Error handling, logging, retries
# @wrap_model_call  | Around each model call    | Model routing, cost control
# @dynamic_prompt   | Before agent starts       | Personalization, context-based prompts
#
# Built-in middleware (from langchain.agents.middleware):
#   PIIMiddleware          → Redacts personal info before model sees it
#   ModelRetryMiddleware   → Auto-retry on model failures
#   ToolRetryMiddleware    → Auto-retry on tool failures
#   HumanInTheLoopMiddleware → Pause for human approval


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. @wrap_tool_call — intercept tool execution (error handling, retries)
# 2. @wrap_model_call — intercept model calls (routing, cost control)
# 3. @dynamic_prompt — change system prompt at runtime (personalization)
# 4. middleware=[...] — pass list to create_agent, runs in order
# 5. Never change agent code — just plug in middleware
# 6. Built-in: PIIMiddleware, ModelRetryMiddleware, HumanInTheLoopMiddleware
#
# NEXT: step_14_callbacks.py — Monitoring with callbacks and LangSmith
