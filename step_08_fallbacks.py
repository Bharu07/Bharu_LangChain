# ============================================================
# LANGCHAIN — STEP 08: Fallbacks and Error Handling
# Goal: Handle model failures gracefully with fallback models,
#       retries, and custom error handling.
# Run:  python step_08_fallbacks.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: MODEL FALLBACKS ──────────────────────────────────────────────────

print("=" * 55)
print("PART 1: Model fallbacks (try backup on failure)")
print("=" * 55)

# If primary model fails (rate limit, outage), try backup:
primary = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)
backup = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)
# In real use, backup would be a different provider:
# backup = init_chat_model("anthropic:claude-sonnet-4-6")

model_with_fallback = primary.with_fallbacks([backup])
# If primary raises an exception → automatically tries backup
# If backup also fails → raises the error

response = model_with_fallback.invoke("Hello!")
print(f"Response: {response.content}")
print("(If primary failed, backup would have answered)")


# ── PART 2: CHAIN-LEVEL FALLBACKS ────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Chain-level fallbacks")
print("=" * 55)

# Fallbacks work at the chain level too:
prompt = ChatPromptTemplate.from_template("Explain {topic} briefly.")

primary_chain = prompt | primary | StrOutputParser()
backup_chain = prompt | backup | StrOutputParser()

safe_chain = primary_chain.with_fallbacks([backup_chain])
result = safe_chain.invoke({"topic": "neural networks"})
print(f"Result: {result[:100]}...")


# ── PART 3: RETRIES (for transient failures) ──────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Retries (auto-retry on transient failures)")
print("=" * 55)

# Rate limits and timeouts are transient — retrying often works:
reliable_model = model.with_retry(
    stop_after_attempt=3,              # Try up to 3 times
    wait_exponential_multiplier=1,     # Wait 1s, 2s, 4s between retries
)
# If first call hits rate limit → waits 1s → retries
# If second fails → waits 2s → retries
# If third fails → raises error

response = reliable_model.invoke("What is Python?")
print(f"Reliable response: {response.content}")


# ── PART 4: COMBINING RETRIES + FALLBACKS ─────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Retries + Fallbacks combined")
print("=" * 55)

# Best practice: retry first, then fall back:
bulletproof_model = (
    model
    .with_retry(stop_after_attempt=2)    # Try twice
    .with_fallbacks([backup])            # Then try backup
)
# Flow: primary(attempt 1) → primary(attempt 2) → backup
response = bulletproof_model.invoke("Hello!")
print(f"Bulletproof: {response.content}")


# ── PART 5: CUSTOM ERROR HANDLING IN CHAINS ───────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Custom error handling with RunnableLambda")
print("=" * 55)

# Sometimes you want custom logic for handling parse errors:
def safe_json_parse(text: str) -> dict:
    """Try to parse JSON, return error dict on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Instead of crashing, return a structured error
        return {"error": "Failed to parse JSON", "raw_text": text[:100]}

json_prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with keys 'name' and 'age' for: {person}"
)

chain = json_prompt | model | StrOutputParser() | RunnableLambda(safe_json_parse)

result = chain.invoke({"person": "Bharath, 26 years old"})
print(f"Parsed: {result}")
# If model returns valid JSON → parsed dict
# If model returns garbage → {"error": "...", "raw_text": "..."}


# ── PART 6: GRACEFUL DEGRADATION PATTERN ──────────────────────────────────────

print("\n" + "=" * 55)
print("PART 6: Graceful degradation pattern")
print("=" * 55)

# Real-world pattern: try structured output, fall back to plain text
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str
    key_points: list[str]

# Primary: structured output
structured_chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | model.with_structured_output(Summary)
)

# Fallback: plain text (always works)
text_chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | model
    | StrOutputParser()
)

safe_summary = structured_chain.with_fallbacks([text_chain])
result = safe_summary.invoke({"text": "Python was created by Guido van Rossum in 1991."})
print(f"Result type: {type(result).__name__}")
print(f"Result: {result}")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. model.with_fallbacks([backup]) — try alternative on failure
# 2. chain.with_fallbacks([backup_chain]) — works at chain level too
# 3. model.with_retry(stop_after_attempt=3) — retry transient failures
# 4. Combine: .with_retry().with_fallbacks() — retry THEN fallback
# 5. RunnableLambda(safe_fn) — custom error handling in chains
# 6. Pattern: structured → plain text fallback for reliability
#
# NEXT: step_09_rag_pipeline.py — Complete RAG pipeline
