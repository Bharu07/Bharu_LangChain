# ============================================================
# LANGCHAIN — STEP 03: Streaming and Batch Processing
# Goal: Get real-time token output (typing effect) and
#       process multiple requests in parallel for speed.
# Run:  python step_03_streaming_and_batch.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: invoke() vs stream() ─────────────────────────────────────────────

print("=" * 55)
print("PART 1: invoke() — waits for complete response")
print("=" * 55)

# invoke() blocks until the ENTIRE response is generated
response = model.invoke("Write a haiku about coding.")
print(response.content)
# You wait 2-3 seconds seeing nothing, then get the full text at once.


print("\n" + "=" * 55)
print("PART 2: stream() — real-time token output")
print("=" * 55)

# stream() yields tokens AS THEY ARE GENERATED
print("Streaming: ", end="", flush=True)
for chunk in model.stream("Write a haiku about coding."):
    print(chunk.content, end="", flush=True)
    # Each chunk is a small piece (word or partial word)
    # end="" = no newline between chunks
    # flush=True = print immediately, don't buffer
print()  # newline at end

# WHY STREAMING?
#   - Better UX (user sees response building up)
#   - Feels faster (even though total time is the same)
#   - Like ChatGPT's typing effect


# ── PART 3: COLLECTING A STREAM ───────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Collecting stream into single message")
print("=" * 55)

# Sometimes you want streaming for display BUT also need the full message
full_response = None
for chunk in model.stream("What is AI in one sentence?"):
    if full_response is None:
        full_response = chunk
    else:
        full_response = full_response + chunk
    # LangChain AIMessageChunks support + operator to merge

print(f"Full response: {full_response.content}")
print(f"Token usage: {full_response.usage_metadata}")


# ── PART 4: BATCH PROCESSING ─────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: batch() — multiple requests in parallel")
print("=" * 55)

# Without batch (SLOW — sequential, one at a time):
#   for q in questions:
#       model.invoke(q)  # Waits for each to finish before starting next

# With batch (FAST — all run in parallel):
questions = [
    "What is Python? One sentence.",
    "What is JavaScript? One sentence.",
    "What is Rust? One sentence.",
]

import time
start = time.time()
responses = model.batch(questions)
elapsed = time.time() - start

for q, r in zip(questions, responses):
    print(f"  Q: {q}")
    print(f"  A: {r.content}\n")

print(f"Batch time: {elapsed:.2f}s (all 3 ran in parallel!)")


# ── PART 5: BATCH WITH CONCURRENCY LIMIT ──────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Batch with rate limit control")
print("=" * 55)

# If your API has rate limits, control parallelism:
responses = model.batch(
    questions,
    config={"max_concurrency": 2}  # Max 2 requests at a time
)
print(f"Rate-limited batch: {len(responses)} responses received")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. invoke() = blocks until complete (simple, good for backend)
# 2. stream() = yields tokens in real-time (good for UX/frontend)
# 3. Chunks can be merged with + operator
# 4. batch() = run multiple requests in parallel (much faster)
# 5. max_concurrency = control rate limits
#
# NEXT: step_04_prompt_templates.py — Reusable prompt templates