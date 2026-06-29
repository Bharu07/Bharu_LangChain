# ============================================================
# LANGCHAIN — STEP 06: Output Parsers and Structured Output
# Goal: Get models to return structured data (JSON, Pydantic objects)
#       instead of plain text.
# Run:  python step_06_output_parsers.py
# ============================================================

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)


# ── PART 1: StrOutputParser (basic text extraction) ───────────────────────────

print("=" * 55)
print("PART 1: StrOutputParser")
print("=" * 55)

# Without parser: model.invoke() returns AIMessage object
response = model.invoke("What is Python?")
print(f"Without parser: {type(response).__name__} → {response.content[:50]}...")

# With parser: just the text string
chain = ChatPromptTemplate.from_template("{q}") | model | StrOutputParser()
result = chain.invoke({"q": "What is Python? One sentence."})
print(f"With parser: {type(result).__name__} → {result}")


# ── PART 2: JsonOutputParser (get JSON from model) ────────────────────────────

print("\n" + "=" * 55)
print("PART 2: JsonOutputParser")
print("=" * 55)

# Define the expected structure with Pydantic:
class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")

parser = JsonOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie information from the user's text.\n{format_instructions}"),
    ("human", "{query}")
])

chain = prompt | model | parser

result = chain.invoke({
    "query": "The Dark Knight came out in 2008, it's a superhero action movie",
    "format_instructions": parser.get_format_instructions(),
    # get_format_instructions() tells the model EXACTLY what JSON format to return
})
print(f"Parsed: {result}")
print(f"Title: {result['title']}, Year: {result['year']}, Genre: {result['genre']}")


# ── PART 3: with_structured_output (RECOMMENDED — modern approach) ────────────

print("\n" + "=" * 55)
print("PART 3: with_structured_output (best approach)")
print("=" * 55)

# This is the PREFERRED way in LangChain v1.
# The model returns a validated Pydantic object directly.

class MovieInfo(BaseModel):
    """Information about a movie."""
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    director: str = Field(description="Director name")

structured_model = model.with_structured_output(MovieInfo)
# This wraps the model to ALWAYS return a MovieInfo object.

result = structured_model.invoke("Tell me about Inception directed by Christopher Nolan in 2010")
print(f"Type: {type(result).__name__}")
print(f"Title: {result.title}")
print(f"Year: {result.year}")
print(f"Genre: {result.genre}")
print(f"Director: {result.director}")


# ── PART 4: COMPLEX STRUCTURED OUTPUT ─────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Complex structured output (nested objects)")
print("=" * 55)

class Ingredient(BaseModel):
    name: str
    amount: str

class Recipe(BaseModel):
    """A cooking recipe."""
    name: str = Field(description="Recipe name")
    prep_time: str = Field(description="Preparation time")
    ingredients: List[Ingredient] = Field(description="List of ingredients")
    steps: List[str] = Field(description="Cooking steps")

recipe_model = model.with_structured_output(Recipe)
result = recipe_model.invoke("Give me a simple recipe for scrambled eggs")
print(f"Recipe: {result.name}")
print(f"Prep time: {result.prep_time}")
print(f"Ingredients: {[f'{i.amount} {i.name}' for i in result.ingredients]}")
print(f"Steps: {result.steps}")


# ── PART 5: STRUCTURED OUTPUT IN CHAINS ───────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: Structured output in chains")
print("=" * 55)

class CodeReview(BaseModel):
    """Code review feedback."""
    quality: str = Field(description="Overall quality: good/fair/poor")
    issues: List[str] = Field(description="List of issues found")
    suggestions: List[str] = Field(description="Improvement suggestions")

review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior code reviewer. Analyze the code and provide structured feedback."),
    ("human", "Review this code:\n{code}")
])

review_chain = review_prompt | model.with_structured_output(CodeReview)

result = review_chain.invoke({
    "code": """
def calculate(x, y, op):
    if op == 'add': return x + y
    if op == 'sub': return x - y
    eval(f'{x} {op} {y}')
"""
})
print(f"Quality: {result.quality}")
print(f"Issues: {result.issues}")
print(f"Suggestions: {result.suggestions}")


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. StrOutputParser() — extract plain text from AIMessage
# 2. JsonOutputParser(pydantic_object=...) — parse JSON with format instructions
# 3. model.with_structured_output(Schema) — BEST approach, returns Pydantic object
# 4. Pydantic models define the expected structure with Field descriptions
# 5. Works perfectly in chains: prompt | structured_model
#
# NEXT: step_07_chain_with_history.py — Conversation memory in chains
