# ============================================================
# LANGCHAIN — STEP 15: Real-World Chain (Customer Support Email)
# Goal: Combine everything into a production-ready chain.
#       Analyze ticket → Route → Generate response → Stream.
# Run:  python step_15_real_world_chain.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ══════════════════════════════════════════════════════════════════════════════
# THE PIPELINE:
#
#   Customer Ticket
#        ↓
#   [1. ANALYZE] → TicketAnalysis (structured output)
#        ↓
#   [2. ROUTE] → Based on category (billing/technical/general)
#        ↓
#   [3. GENERATE] → Specialized response per category
#        ↓
#   [4. STREAM] → Real-time output
# ══════════════════════════════════════════════════════════════════════════════


# ── STEP 1: ANALYZE TICKET (Structured Output) ───────────────────────────────

class TicketAnalysis(BaseModel):
    """Structured analysis of a customer support ticket."""
    category: Literal["billing", "technical", "general"] = Field(
        description="Ticket category"
    )
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        description="Customer's emotional tone"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent is this"
    )
    summary: str = Field(description="One-line summary of the issue")

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support ticket analyzer. Classify the ticket accurately."),
    ("human", "Analyze this customer ticket:\n\n{ticket}")
])

analyze_chain = analyze_prompt | model.with_structured_output(TicketAnalysis)


# ── STEP 2: ROUTE BY CATEGORY (Specialized Prompts) ──────────────────────────

billing_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a billing support specialist. Be empathetic and solution-oriented.
Always offer specific next steps. If the customer is angry, acknowledge their frustration first.
Mention relevant policies when applicable."""),
    ("human", "Customer ticket (sentiment: {sentiment}, urgency: {urgency}):\n{ticket}\n\nWrite a professional response email.")
])

technical_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical support specialist. Be clear and step-by-step.
Provide troubleshooting steps. If you can't solve it, offer escalation.
Use numbered steps for clarity."""),
    ("human", "Customer ticket (sentiment: {sentiment}, urgency: {urgency}):\n{ticket}\n\nWrite a helpful response email.")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support representative. Be friendly and helpful.
Answer their question clearly and offer additional assistance."""),
    ("human", "Customer ticket (sentiment: {sentiment}, urgency: {urgency}):\n{ticket}\n\nWrite a friendly response email.")
])


# ── STEP 3: FULL PIPELINE ────────────────────────────────────────────────────

def process_ticket(ticket: str) -> dict:
    """Analyze ticket and prepare for routing."""
    analysis = analyze_chain.invoke({"ticket": ticket})
    return {
        "ticket": ticket,
        "category": analysis.category,
        "sentiment": analysis.sentiment,
        "urgency": analysis.urgency,
        "summary": analysis.summary,
    }

def route_to_specialist(info: dict) -> str:
    """Route to the right prompt based on category."""
    category = info["category"]
    if category == "billing":
        chain = billing_prompt | model | StrOutputParser()
    elif category == "technical":
        chain = technical_prompt | model | StrOutputParser()
    else:
        chain = general_prompt | model | StrOutputParser()
    return chain.invoke(info)


# ── RUN THE PIPELINE ──────────────────────────────────────────────────────────

tickets = [
    """Subject: CHARGED TWICE!!!
I was charged $49.99 TWICE for my subscription this month. This is unacceptable!
I've been a loyal customer for 3 years and this is how you treat me?
Fix this immediately or I'm canceling everything.""",

    """Subject: Can't login after update
Hi, since the latest app update (v3.2.1) I can't log in anymore.
I've tried resetting my password and clearing the cache but nothing works.
I'm on Android 14, Samsung Galaxy S24. Please help!""",

    """Subject: Question about team plans
Hello, I'm interested in upgrading from the individual plan to a team plan
for my company (about 15 people). Could you tell me about pricing and
what features are included? Thanks!""",
]

print("=" * 60)
print("CUSTOMER SUPPORT EMAIL GENERATOR")
print("=" * 60)

for i, ticket in enumerate(tickets, 1):
    print(f"\n{'─' * 60}")
    print(f"TICKET {i}")
    print(f"{'─' * 60}")

    # Step 1: Analyze
    info = process_ticket(ticket)
    print(f"📊 Category: {info['category']} | Sentiment: {info['sentiment']} | Urgency: {info['urgency']}")
    print(f"📝 Summary: {info['summary']}")

    # Step 2 & 3: Route and Generate
    print(f"\n📧 RESPONSE:")
    response = route_to_specialist(info)
    print(response)


# ── BONUS: STREAMING THE RESPONSE ─────────────────────────────────────────────

print(f"\n{'═' * 60}")
print("BONUS: Streaming response in real-time")
print(f"{'═' * 60}")

# For the last ticket, stream the response:
info = process_ticket(tickets[2])
print(f"Category: {info['category']}")
print(f"\nStreaming response:\n")

response_chain = general_prompt | model | StrOutputParser()
for chunk in response_chain.stream(info):
    print(chunk, end="", flush=True)
print()


# ── WHAT WE LEARNED (Full Pipeline) ──────────────────────────────────────────
#
# This real-world chain combines:
#   ✅ Structured output (TicketAnalysis Pydantic model)
#   ✅ Conditional routing (RunnableBranch / manual routing)
#   ✅ Multiple specialized prompts (billing/technical/general)
#   ✅ Streaming for real-time output
#   ✅ Chain composition (prompt | model | parser)
#
# PRODUCTION ADDITIONS you'd add:
#   • Fallbacks (if primary model fails)
#   • Callbacks (cost tracking, latency monitoring)
#   • LangSmith tracing
#   • RAG (retrieve relevant help articles)
#   • Human-in-the-loop (review before sending)
#
# ═══════════════════════════════════════════════════════════════════════════════
# END OF LANGCHAIN CODES — All 15 steps cover:
#   01: Models (init_chat_model)
#   02: Messages (types, content_blocks)
#   03: Streaming and Batch
#   04: Prompt Templates
#   05: Chains (LCEL, Runnables)
#   06: Output Parsers (structured output)
#   07: Chain with History (memory)
#   08: Fallbacks and Error Handling
#   09: RAG Pipeline
#   10: Tools (@tool, ToolNode)
#   11: Agents (create_agent)
#   12: Agent Streaming + Structured Output
#   13: Middleware (v1)
#   14: Callbacks + LangSmith
#   15: Real-World Chain (this file)
# ═══════════════════════════════════════════════════════════════════════════════
