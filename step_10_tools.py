# ============================================================
# LANGCHAIN — STEP 10: Tools and Tool Calling
# Goal: Create tools with @tool, add input validation with Pydantic,
#       understand ToolNode, and bind tools to models.
# Run:  python step_10_tools.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: BASIC TOOL (@tool decorator) ──────────────────────────────────────

print("=" * 55)
print("PART 1: Basic tool creation")
print("=" * 55)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # The DOCSTRING is CRITICAL — the model reads it to know WHEN to use this tool.
    # A clear, specific docstring = model uses tool correctly.
    weather = {
        "london": "Cloudy, 15°C, humidity 80%",
        "tokyo": "Sunny, 25°C, humidity 45%",
        "paris": "Rainy, 12°C, humidity 90%",
    }
    return weather.get(city.lower(), f"No weather data for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2', '10 * 5'."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"Result: {eval(expression)}"
        return "Error: Invalid characters"
    except Exception as e:
        return f"Error: {e}"

# Inspect tool metadata:
print(f"Tool name: {get_weather.name}")
print(f"Tool description: {get_weather.description}")
print(f"Tool args schema: {get_weather.args_schema.model_json_schema()}")

# Call directly (for testing):
print(f"Direct call: {get_weather.invoke('london')}")


# ── PART 2: ARGS SCHEMA (Pydantic validation) ────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Args schema with Pydantic validation")
print("=" * 55)

# Define strict input validation:
class SearchInput(BaseModel):
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results (1-20)")
    category: Literal["web", "news", "images"] = Field(default="web", description="Search category")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5, category: str = "web") -> str:
    """Search for information with filters."""
    return f"Found {max_results} {category} results for '{query}'"

# The model sees this schema and knows exactly what args to pass:
print(f"Schema: {search.args_schema.model_json_schema()}")
print(f"Call: {search.invoke({'query': 'Python tutorials', 'max_results': 3, 'category': 'web'})}")


# ── PART 3: DIFFERENT RETURN TYPES ───────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Tool return types")
print("=" * 55)

# Return string (simplest — agent reads it as text)
@tool
def simple_tool() -> str:
    """A simple tool returning text."""
    return "Hello from tool!"

# Return dict (structured data)
@tool
def dict_tool(city: str) -> dict:
    """Get structured weather data."""
    return {"city": city, "temp": 25, "unit": "celsius", "condition": "sunny"}

# The model interprets the return value as the tool's observation.
print(f"String return: {simple_tool.invoke({})}")
print(f"Dict return: {dict_tool.invoke({'city': 'tokyo'})}")


# ── PART 4: BIND TOOLS TO MODEL (without agent) ──────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Bind tools to model (model suggests, doesn't execute)")
print("=" * 55)

# bind_tools tells the model about available tools:
model_with_tools = model.bind_tools([get_weather, calculate])

response = model_with_tools.invoke("What's the weather in Tokyo?")
print(f"Content: {response.content}")
print(f"Tool calls: {response.tool_calls}")
# The model SUGGESTS calling get_weather("Tokyo")
# But it does NOT execute the tool — that's the agent/ToolNode's job!

response2 = model_with_tools.invoke("What is 15 * 37?")
print(f"\nContent: {response2.content}")
print(f"Tool calls: {response2.tool_calls}")

# No tools needed for general knowledge:
response3 = model_with_tools.invoke("What color is the sky?")
print(f"\nContent: {response3.content}")
print(f"Tool calls: {response3.tool_calls}")
# Empty tool_calls — model answers directly.


# ── PART 5: TOOLNODE (Auto-execute tool calls) ────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: ToolNode (executes tool calls from AIMessage)")
print("=" * 55)

from langgraph.prebuilt import ToolNode

tools = [get_weather, calculate]
tool_node = ToolNode(tools)

# ToolNode reads an AIMessage with tool_calls and executes them:
# 1. Reads response.tool_calls → [{"name": "get_weather", "args": {"city": "tokyo"}}]
# 2. Finds the matching Python function
# 3. Calls it with the args
# 4. Returns ToolMessage with the result

# Simulate: model suggests a tool call, ToolNode executes it
response = model_with_tools.invoke("Weather in Paris?")
if response.tool_calls:
    # Pass the AI's response to ToolNode for execution:
    tool_results = tool_node.invoke({"messages": [response]})
    print(f"Tool executed! Results:")
    for msg in tool_results["messages"]:
        print(f"  [{msg.name}]: {msg.content}")


# ── PART 6: tools_condition (routing helper) ──────────────────────────────────

print("\n" + "=" * 55)
print("PART 6: tools_condition (route: tools or end)")
print("=" * 55)

from langgraph.prebuilt import tools_condition

# tools_condition checks if the last message has tool_calls:
#   If yes → returns "tools" (go to ToolNode)
#   If no  → returns "__end__" (agent is done)
# Used in LangGraph conditional edges to route the agent loop.
print("tools_condition replaces manual should_continue() functions")
print("Usage: graph.add_conditional_edges('model', tools_condition)")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. @tool decorator — turns function into a tool (docstring is key!)
# 2. args_schema=PydanticModel — strict input validation
# 3. model.bind_tools() — tells model about tools (suggests, doesn't execute)
# 4. ToolNode(tools) — prebuilt node that EXECUTES tool calls from AIMessage
# 5. tools_condition — routing helper (has tool_calls? → "tools" else → "end")
# 6. Return types: str (simple), dict (structured), Command (state update)
#
# NEXT: step_11_agents.py — Building agents with create_agent
