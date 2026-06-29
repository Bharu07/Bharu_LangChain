# ============================================================
# LANGCHAIN — STEP 02: Messages (Types, Formats, Content Blocks)
# Goal: Understand message types, three input formats,
#       conversation history, and the new content_blocks API.
# Run:  python step_02_messages.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.messages import (
    SystemMessage,    # Sets the AI's role/behavior
    HumanMessage,     # User's input
    AIMessage,        # AI's response (used in history)
    ToolMessage,      # Result from a tool call
)

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: MESSAGE TYPES ─────────────────────────────────────────────────────

print("=" * 55)
print("PART 1: The four message types")
print("=" * 55)

# SystemMessage — tells the AI WHO it is and HOW to behave
# HumanMessage  — what the user says
# AIMessage     — what the AI previously said (for history)
# ToolMessage   — what a tool returned (for agents)

response = model.invoke([
    SystemMessage(content="You are a pirate. Always speak like one."),
    HumanMessage(content="What is the weather today?"),
])
print(f"Pirate AI: {response.content}")


# ── PART 2: THREE INPUT FORMATS ───────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Three ways to pass messages")
print("=" * 55)

# FORMAT 1: Plain string (simplest)
r1 = model.invoke("What is 2+2?")
print(f"Format 1 (string): {r1.content}")

# FORMAT 2: Message objects (most explicit)
r2 = model.invoke([
    SystemMessage(content="Be concise."),
    HumanMessage(content="What is 2+2?"),
])
print(f"Format 2 (objects): {r2.content}")

# FORMAT 3: Dicts (shorthand — used in agents)
r3 = model.invoke([
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
])
print(f"Format 3 (dicts): {r3.content}")
# All three produce the same result!


# ── PART 3: CONVERSATION HISTORY ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Manual conversation history")
print("=" * 55)

# Models have NO memory! Each .invoke() is independent.
# To make the AI "remember", pass previous messages manually:

history = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="My name is Bharath and I work at Infosys."),
    AIMessage(content="Nice to meet you, Bharath! How can I help you today?"),
    HumanMessage(content="Where do I work?"),  # Tests memory
]

response = model.invoke(history)
print(f"AI remembers: {response.content}")
# Output: "You work at Infosys." — because we passed the full history!

# Without history:
response2 = model.invoke("Where do I work?")
print(f"Without history: {response2.content}")
# Output: "I don't know where you work." — no context!


# ── PART 4: CONTENT BLOCKS (v1 — Provider-Agnostic) ──────────────────────────

print("\n" + "=" * 55)
print("PART 4: content_blocks (unified response format)")
print("=" * 55)

# THE PROBLEM:
#   Different models return responses in DIFFERENT formats:
#
#   OpenAI GPT-4o returns:
#     response.content = "Paris"  (just a plain string)
#
#   Anthropic Claude returns:
#     response.content = [{"type": "text", "text": "Paris"},
#                         {"type": "thinking", "thinking": "..."}]
#     (a list with text + reasoning blocks)
#
#   Google Gemini returns:
#     response.content = "Paris"  (string, but tool calls are separate)
#
#   If you write code that does response.content[0]["text"],
#   it BREAKS when you switch from Claude to GPT-4o (because GPT returns a string).
#
# THE SOLUTION — content_blocks:
#   response.content_blocks gives you a STANDARD format
#   that works the SAME for ALL providers:
#     [{"type": "text", "text": "Paris"}]
#
#   Even if the underlying model returns a plain string, LangChain wraps it
#   into this standard block format for you.
#
# WHY THIS MATTERS:
#   You write your parsing code ONCE using content_blocks,
#   then freely swap providers (OpenAI → Claude → Gemini) without
#   changing a single line of your response-handling code.
#   This is the whole point of LangChain's abstraction layer.

response = model.invoke("What is the capital of France? One word.")

print(f"Raw .content: {response.content}")
# .content — the raw format (string for OpenAI, list for Claude)
# Works fine for simple text, but breaks if you switch providers.

print(f".content_blocks: {response.content_blocks}")
# .content_blocks — ALWAYS a list of typed dicts, regardless of provider.
# This is the SAFE way to parse responses.

# Access text from content_blocks (same code for ALL providers):
for block in response.content_blocks:
    if block["type"] == "text":
        print(f"  Text block: {block['text']}")
        # Every provider will have at least one "text" block.

    elif block["type"] == "reasoning":
        print(f"  Reasoning: {block['reasoning']}")
        # Only some models (Claude with extended thinking, o1) return this.
        # If you switch to GPT-4o, this block simply won't appear — no crash.

    elif block["type"] == "tool_call":
        print(f"  Tool call: {block['name']}")
        # When the model wants to call a function/tool.

# KEY TAKEAWAY:
# You do NOT need to change your code when switching models.
# content_blocks normalizes everything. Missing block types are
# simply absent — your if/elif just skips them gracefully.


# ── PART 5: MULTIMODAL (Images) ───────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Sending images (multimodal)")
print("=" * 55)

# Models like GPT-4o can understand images:
multimodal_msg = HumanMessage(content=[
    {"type": "text", "text": "Describe this image in one sentence."},
    {"type": "image_url", "image_url": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png"
    }},
])

response = model.invoke([multimodal_msg])
print(f"Image description: {response.content}")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. Four message types: System, Human, AI, Tool
# 2. Three input formats: string, objects, dicts (all equivalent)
# 3. Models have NO memory — pass history manually
# 4. content_blocks = unified response format for all providers
# 5. Multimodal: send images via {"type": "image_url", ...}
#
# NEXT: step_03_streaming_and_batch.py — Real-time output and parallel requests
