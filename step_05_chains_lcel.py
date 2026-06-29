# ============================================================
# LANGCHAIN — STEP 05: Chains (LCEL — LangChain Expression Language)
# Goal: Master the pipe operator, RunnablePassthrough, RunnableLambda,
#       RunnableParallel, RunnableBranch — the building blocks of chains.
# Run:  python step_05_chains_lcel.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
)

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)
parser = StrOutputParser()


# ── PART 1: BASIC CHAIN (prompt | model | parser) ─────────────────────────────

print("=" * 55)
print("PART 1: Basic LCEL chain")
print("=" * 55)

prompt = ChatPromptTemplate.from_template("Explain {topic} in 2 sentences.")
chain = prompt | model | parser

result = chain.invoke({"topic": "recursion"})
print(f"Result: {result}")

# Everything built with | supports these for free:
#   chain.invoke()  — single execution
#   chain.stream()  — token-by-token
#   chain.batch()   — parallel execution


# ── PART 2: RunnablePassthrough (pass data through) ───────────────────────────

print("\n" + "=" * 55)
print("PART 2: RunnablePassthrough")
print("=" * 55)

# RunnablePassthrough passes its input through unchanged.
# Used to create dict-shaped inputs for prompts:

prompt2 = ChatPromptTemplate.from_template(
    "The topic is: {topic}\nExplain in {style} style."
)

chain2 = (
    {"topic": RunnablePassthrough(), "style": lambda x: "simple"}
    | prompt2 | model | parser
)
print(chain2.invoke("machine learning"))
# RunnablePassthrough() passes "machine learning" as-is to "topic"
# lambda returns "simple" for "style"


# .assign() — add new keys while keeping all existing ones:
chain3 = RunnablePassthrough.assign(
    upper_topic=lambda x: x["topic"].upper(),
    word_count=lambda x: len(x["topic"].split()),
)
result = chain3.invoke({"topic": "neural networks"})
print(f"Assign result: {result}")
# {"topic": "neural networks", "upper_topic": "NEURAL NETWORKS", "word_count": 2}


# ── PART 3: RunnableLambda (custom Python functions in chain) ─────────────────

print("\n" + "=" * 55)
print("PART 3: RunnableLambda")
print("=" * 55)

# Wrap ANY Python function as a chain step:
def add_emoji(text: str) -> str:
    """Add a rocket emoji to the end."""
    return text + " 🚀"

def word_count(text: str) -> str:
    """Count words in the text."""
    count = len(text.split())
    return f"{text}\n[Word count: {count}]"

chain4 = prompt | model | parser | RunnableLambda(add_emoji)
print(chain4.invoke({"topic": "APIs"}))

# Chain multiple lambdas:
chain5 = prompt | model | parser | RunnableLambda(add_emoji) | RunnableLambda(word_count)
print(chain5.invoke({"topic": "REST"}))


# ── PART 4: RunnableParallel (run multiple chains simultaneously) ─────────────

print("\n" + "=" * 55)
print("PART 4: RunnableParallel")
print("=" * 55)

# Run multiple chains at the same time on the same input:
summary_prompt = ChatPromptTemplate.from_template("Summarize {topic} in 1 sentence.")
translate_prompt = ChatPromptTemplate.from_template("Translate '{topic}' to French.")

parallel_chain = RunnableParallel(
    summary=summary_prompt | model | parser,
    french=translate_prompt | model | parser,
    original=RunnablePassthrough(),
)

result = parallel_chain.invoke({"topic": "artificial intelligence"})
print(f"Summary: {result['summary']}")
print(f"French: {result['french']}")
print(f"Original input: {result['original']}")
# Both summary and translation run IN PARALLEL — faster!


# ── PART 5: RunnableBranch (conditional routing) ──────────────────────────────

print("\n" + "=" * 55)
print("PART 5: RunnableBranch (if/else in chains)")
print("=" * 55)

# Route to different chains based on conditions:
code_prompt = ChatPromptTemplate.from_template(
    "Explain this code concept: {topic}. Include a code example."
)
math_prompt = ChatPromptTemplate.from_template(
    "Solve this math problem: {topic}. Show step by step."
)
general_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply."
)

branch = RunnableBranch(
    (lambda x: "code" in x["topic"].lower() or "python" in x["topic"].lower(),
     code_prompt | model | parser),
    (lambda x: any(w in x["topic"].lower() for w in ["math", "calculate", "equation"]),
     math_prompt | model | parser),
    general_prompt | model | parser,  # Default fallback
)

print("Code topic:")
print(branch.invoke({"topic": "Python decorators code"}))
print("\nMath topic:")
print(branch.invoke({"topic": "Calculate 15% of 200"}))
print("\nGeneral topic:")
print(branch.invoke({"topic": "climate change"}))


# ── PART 6: DICT SYNTAX (shorthand for RunnableParallel) ──────────────────────

print("\n" + "=" * 55)
print("PART 6: Dict syntax (common pattern)")
print("=" * 55)

# This dict is actually a RunnableParallel:
rag_style_chain = (
    {"context": lambda x: "Python is a programming language", "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Context: {context}\n\nAnswer: {question}")
    | model
    | parser
)
print(rag_style_chain.invoke("What is Python?"))
# This pattern is used everywhere in RAG chains.


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. prompt | model | parser — basic LCEL chain
# 2. RunnablePassthrough() — pass input unchanged, .assign() to add keys
# 3. RunnableLambda(fn) — wrap any function as a chain step
# 4. RunnableParallel(a=..., b=...) — run branches simultaneously
# 5. RunnableBranch((cond, chain), ..., default) — conditional routing
# 6. Dict syntax {} — shorthand for RunnableParallel
#
# NEXT: step_06_output_parsers.py — Structured output (JSON, Pydantic)
