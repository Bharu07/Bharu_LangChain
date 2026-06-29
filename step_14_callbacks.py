# ============================================================
# LANGCHAIN — STEP 14: Callbacks and LangSmith
# Goal: Monitor what's happening inside chains/agents with callbacks.
#       Understand LangSmith for production tracing.
# Run:  python step_14_callbacks.py
# ============================================================

from dotenv import load_dotenv
import os
import time
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: CUSTOM CALLBACK (Timing) ─────────────────────────────────────────

print("=" * 55)
print("PART 1: Custom callback — measure LLM latency")
print("=" * 55)

class TimingCallback(BaseCallbackHandler):
    """Measures how long each LLM call takes."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        print(f"  ⏱️  LLM call started...")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"  ⏱️  LLM call finished in {elapsed:.2f}s")

# Use with invoke:
chain = ChatPromptTemplate.from_template("{q}") | model | StrOutputParser()
result = chain.invoke(
    {"q": "What is Python?"},
    config={"callbacks": [TimingCallback()]}
)
print(f"  Answer: {result[:80]}")


# ── PART 2: LOGGING CALLBACK ─────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Logging callback — see what's happening")
print("=" * 55)

class LoggingCallback(BaseCallbackHandler):
    """Logs every step of chain/agent execution."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "Unknown")
        print(f"  📝 Chain '{name}' started")

    def on_chain_end(self, outputs, **kwargs):
        print(f"  📝 Chain finished")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"  🤖 Calling LLM...")

    def on_llm_end(self, response, **kwargs):
        print(f"  🤖 LLM responded")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown")
        print(f"  🔧 Tool '{name}' called with: {input_str[:50]}")

    def on_tool_end(self, output, **kwargs):
        print(f"  🔧 Tool returned: {str(output)[:50]}")

result = chain.invoke(
    {"q": "Explain REST APIs briefly."},
    config={"callbacks": [LoggingCallback()]}
)
print(f"  Answer: {result[:80]}")


# ── PART 3: COST TRACKING CALLBACK ───────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Cost tracking callback")
print("=" * 55)

class CostTracker(BaseCallbackHandler):
    """Track token usage and estimate costs."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response, **kwargs):
        self.call_count += 1
        # Token usage varies by provider — check response structure
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)

    def report(self):
        total = self.total_input_tokens + self.total_output_tokens
        # GPT-4o pricing (approximate): $5/1M input, $15/1M output
        cost = (self.total_input_tokens * 5 + self.total_output_tokens * 15) / 1_000_000
        print(f"  Calls: {self.call_count}")
        print(f"  Tokens: {self.total_input_tokens} input + {self.total_output_tokens} output = {total}")
        print(f"  Estimated cost: ${cost:.4f}")

tracker = CostTracker()

# Run multiple queries with same tracker:
for q in ["What is AI?", "What is ML?", "What is DL?"]:
    chain.invoke({"q": q}, config={"callbacks": [tracker]})

tracker.report()


# ── PART 4: MULTIPLE CALLBACKS ────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Multiple callbacks combined")
print("=" * 55)

# Stack multiple callbacks:
result = chain.invoke(
    {"q": "What is Docker?"},
    config={"callbacks": [TimingCallback(), LoggingCallback()]}
)
print(f"  Answer: {result[:80]}")


# ── PART 5: LANGSMITH (Production Tracing) ────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: LangSmith integration")
print("=" * 55)

print("""
LangSmith = Production monitoring for LLM applications.

Setup (add to .env):
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_pt_xxxxx
  LANGSMITH_PROJECT=my-project

Once set, ALL chains/agents are automatically traced:
  • Full conversation history
  • Each tool call with inputs/outputs
  • Token usage and latency per step
  • Error traces with full context
  • Evaluation scores over time

Dashboard at: https://smith.langchain.com

Benefits over custom callbacks:
  • Zero code changes (just env vars)
  • Visual trace explorer
  • Team collaboration
  • A/B testing different prompts
  • Dataset management for testing
""")


# ── CALLBACK HOOKS REFERENCE ─────────────────────────────────────────────────
#
# Hook                  | When it fires
# ----------------------|------------------------------------------
# on_llm_start          | Before calling the model
# on_llm_end            | After model responds
# on_llm_error          | When model call fails
# on_chain_start        | Before a chain starts
# on_chain_end          | After a chain finishes
# on_tool_start         | Before a tool executes
# on_tool_end           | After a tool returns
# on_retriever_start    | Before retriever searches
# on_retriever_end      | After retriever returns docs


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. BaseCallbackHandler — override hooks to monitor execution
# 2. on_llm_start/end — track model calls (timing, cost)
# 3. on_tool_start/end — track tool usage
# 4. config={"callbacks": [...]} — pass callbacks to any chain/agent
# 5. Stack multiple callbacks for combined monitoring
# 6. LangSmith — production tracing with zero code changes (just env vars)
#
# NEXT: step_15_real_world_chain.py — Complete real-world example
