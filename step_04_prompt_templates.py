# ============================================================
# LANGCHAIN — STEP 04: Prompt Templates
# Goal: Create reusable prompt templates with variables.
#       Use with chains for clean, maintainable code.
# Run:  python step_04_prompt_templates.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate, MessagesPlaceholder

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: BASIC TEMPLATE ───────────────────────────────────────────────────

print("=" * 55)
print("PART 1: Basic prompt template with variables")
print("=" * 55)

# WHY TEMPLATES?
#   Without: You rebuild the prompt string every time
#   With: Define once, reuse with different variables

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who explains things in {style}."),
    ("human", "{question}")
])
# {role}, {style}, {question} are VARIABLES filled at runtime.
# Nothing is sent to the model yet — this is just a template.

# Use it:
messages = prompt.invoke({
    "role": "Python tutor",
    "style": "simple terms with analogies",
    "question": "What is a list comprehension?"
})
response = model.invoke(messages)
print(f"Answer: {response.content}")


# ── PART 2: TEMPLATE + PIPE = CHAIN ──────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Template in a chain (pipe operator)")
print("=" * 55)

from langchain_core.output_parsers import StrOutputParser

# Chain: prompt → model → extract text
chain = prompt | model | StrOutputParser()

# Reuse with different inputs:
result = chain.invoke({
    "role": "JavaScript expert",
    "style": "one-liners",
    "question": "What is a closure?"
})
print(f"JS Expert: {result}")

result = chain.invoke({
    "role": "data scientist",
    "style": "technical terms",
    "question": "What is overfitting?"
})
print(f"Data Scientist: {result}")


# ── PART 3: SIMPLER TEMPLATE (from_template) ─────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Simple template shorthand")
print("=" * 55)

# When you just need a single user message with a variable:
simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}.")
chain2 = simple | model | StrOutputParser()

print(chain2.invoke({"text": "Hello world", "language": "French"}))
print(chain2.invoke({"text": "Good morning", "language": "Japanese"}))


# ── PART 4: FEW-SHOT PROMPTING ────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Few-shot prompting (teaching by example)")
print("=" * 55)

# Give the model EXAMPLES of what you want before asking:
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "fast", "output": "slow"},
    {"input": "hot", "output": "cold"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You give the opposite of each word. Just the opposite, nothing else."),
    few_shot,  # Inserts all examples here
    ("human", "{input}"),
])

chain3 = final_prompt | model | StrOutputParser()
print(f"Opposite of 'bright': {chain3.invoke({'input': 'bright'})}")
print(f"Opposite of 'tall': {chain3.invoke({'input': 'tall'})}")


# ── PART 5: MESSAGES PLACEHOLDER (for dynamic history) ────────────────────────

print("\n" + "=" * 55)
print("PART 5: MessagesPlaceholder (insert message lists)")
print("=" * 55)

# Used when you need to insert a dynamic list of messages (like chat history)
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),  # Insert previous messages here
    ("human", "{input}"),
])

from langchain.messages import HumanMessage, AIMessage

messages = prompt_with_history.invoke({
    "history": [
        HumanMessage(content="My favorite language is Python"),
        AIMessage(content="Python is great! How can I help?"),
    ],
    "input": "What's my favorite language?",
})
response = model.invoke(messages)
print(f"With history: {response.content}")
# The model remembers because we passed history via the placeholder!


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. ChatPromptTemplate.from_messages() — template with variables
# 2. Pipe operator: prompt | model | parser — creates a reusable chain
# 3. from_template() — simpler shorthand for single-message prompts
# 4. FewShotChatMessagePromptTemplate — teach by example
# 5. MessagesPlaceholder — insert dynamic message lists (for history)
#
# NEXT: step_05_chains_lcel.py — LCEL chains (the | operator in depth)
