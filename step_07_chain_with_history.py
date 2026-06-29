# ============================================================
# LANGCHAIN — STEP 07: Chain with History (Conversation Memory)
# Goal: Give chains memory so they remember previous messages.
#       Multiple sessions with isolated histories.
# Run:  python step_07_chain_with_history.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: THE PROBLEM — Chains are stateless ───────────────────────────────

print("=" * 55)
print("PART 1: The problem — chains have no memory")
print("=" * 55)

simple_chain = ChatPromptTemplate.from_template("{input}") | model | StrOutputParser()

print(simple_chain.invoke({"input": "My name is Bharath"}))
print(simple_chain.invoke({"input": "What's my name?"}))
# → "I don't know your name" — each invoke is independent!


# ── PART 2: THE SOLUTION — RunnableWithMessageHistory ─────────────────────────

print("\n" + "=" * 55)
print("PART 2: Adding memory with RunnableWithMessageHistory")
print("=" * 55)

# Step 1: Prompt with a history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember everything the user tells you."),
    MessagesPlaceholder("history"),  # Previous messages go here
    ("human", "{input}"),            # Current message
])

chain = prompt | model | StrOutputParser()

# Step 2: Session store (holds history per session_id)
store = {}

def get_session_history(session_id: str):
    """Get or create history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Step 3: Wrap the chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # Which key has the user's current message
    history_messages_key="history",  # Which placeholder to fill with history
)

# Step 4: Use it — same session_id = same conversation
config = {"configurable": {"session_id": "user-bharath"}}

print(chain_with_history.invoke({"input": "My name is Bharath and I love Python."}, config=config))
print(chain_with_history.invoke({"input": "What's my name and what do I love?"}, config=config))
# → "Your name is Bharath and you love Python!" — IT REMEMBERS!


# ── PART 3: MULTIPLE SESSIONS (Isolated Histories) ───────────────────────────

print("\n" + "=" * 55)
print("PART 3: Multiple sessions (separate users)")
print("=" * 55)

# Alice's session
alice_config = {"configurable": {"session_id": "alice"}}
chain_with_history.invoke({"input": "I'm Alice, I like JavaScript."}, config=alice_config)

# Bob's session
bob_config = {"configurable": {"session_id": "bob"}}
chain_with_history.invoke({"input": "I'm Bob, I prefer Rust."}, config=bob_config)

# They don't mix!
alice_answer = chain_with_history.invoke({"input": "What's my name and language?"}, config=alice_config)
bob_answer = chain_with_history.invoke({"input": "What's my name and language?"}, config=bob_config)

print(f"Alice's session: {alice_answer}")
print(f"Bob's session: {bob_answer}")
# Each session is completely isolated.


# ── PART 4: VIEWING STORED HISTORY ────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Inspecting stored history")
print("=" * 55)

# You can look at what's stored:
history = get_session_history("user-bharath")
print(f"Messages stored for 'user-bharath': {len(history.messages)}")
for msg in history.messages:
    print(f"  [{type(msg).__name__}] {msg.content[:60]}...")


# ── PART 5: PRE-LOADING HISTORY ───────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Pre-loading conversation history")
print("=" * 55)

# Resume a previous conversation by pre-populating history:
from langchain.messages import HumanMessage, AIMessage

store["resumed-session"] = ChatMessageHistory()
store["resumed-session"].add_user_message("I'm working on a Flask project")
store["resumed-session"].add_ai_message("Great! Flask is a lightweight Python web framework. How can I help?")

resume_config = {"configurable": {"session_id": "resumed-session"}}
result = chain_with_history.invoke(
    {"input": "What framework am I using?"},
    config=resume_config
)
print(f"Resumed: {result}")
# → "You're using Flask!" — pre-loaded history works!


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. Chains are stateless by default — each invoke() is independent
# 2. RunnableWithMessageHistory wraps a chain to add memory
# 3. MessagesPlaceholder("history") — slot where past messages go
# 4. session_id isolates different conversations
# 5. get_session_history returns a ChatMessageHistory per session
# 6. Pre-load history to resume conversations
#
# NEXT: step_08_fallbacks.py — Error handling, retries, and fallbacks
