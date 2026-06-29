# ============================================================
# LANGCHAIN — STEP 09: RAG Pipeline (Retrieval Augmented Generation)
# Goal: Build a complete RAG system — load documents, split, embed,
#       store in vector DB, retrieve, and generate answers.
# Run:  python step_09_rag_pipeline.py
# ============================================================

# ── INSTALL ──────────────────────────────────────────────────────────────────
# pip install -U langchain langchain-openai langchain-community faiss-cpu python-dotenv

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")


# ══════════════════════════════════════════════════════════════════════════════
# THE RAG PIPELINE:
#
#   ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐
#   │ 1. LOAD  │ → │ 2. SPLIT│ → │ 3. EMBED│ → │ 4. STORE    │
#   │(documents)│   │(chunks) │   │(vectors)│   │(vector DB)  │
#   └──────────┘   └─────────┘   └─────────┘   └──────┬──────┘
#                                                       │
#   ┌──────────┐   ┌─────────┐   ┌─────────────────────┘
#   │ 7. ANSWER│ ← │ 6.PROMPT│ ← │ 5. RETRIEVE (search)
#   │  (LLM)   │   │(context)│   │    (user question)
#   └──────────┘   └─────────┘   └─────────────────────
# ══════════════════════════════════════════════════════════════════════════════


# ── STEP 1: LOAD DOCUMENTS ───────────────────────────────────────────────────

print("=" * 60)
print("BUILDING THE RAG PIPELINE")
print("=" * 60)

# In production: PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader
# Here using inline Documents to keep it self-contained:
documents = [
    Document(
        page_content="""
Company Leave Policy:
- Annual leave: 20 days per year for full-time employees.
- Sick leave: 12 days per year (medical certificate needed after 3 days).
- Parental leave: 16 weeks primary caregiver, 4 weeks secondary.
- Unused annual leave: max 5 days carry forward to next year.
""",
        metadata={"source": "handbook.pdf", "section": "leave"},
    ),
    Document(
        page_content="""
Remote Work Policy:
- Eligible after 3-month probation period.
- Up to 3 days remote per week. Tuesday/Thursday in-office required.
- International remote: max 4 weeks/year, needs HR approval.
- Equipment: company laptop + $500/year home office allowance.
- Internet: $50/month for 3+ days remote workers.
""",
        metadata={"source": "handbook.pdf", "section": "remote_work"},
    ),
    Document(
        page_content="""
Expense Policy:
- Flights: economy domestic, business for international 6+ hours.
- Hotels: $200/night domestic, $300/night international.
- Meals: $50/day during travel. Alcohol: max 2 drinks on company.
- Submit expenses within 30 days. After 60 days = no reimbursement.
- Under $500: manager approval. $500-$2000: department head. Over $2000: VP.
""",
        metadata={"source": "handbook.pdf", "section": "expenses"},
    ),
]
print(f"[Step 1] Loaded {len(documents)} documents")


# ── STEP 2: SPLIT INTO CHUNKS ────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Max characters per chunk
    chunk_overlap=50,     # Overlap to preserve context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],  # Split priorities
)
# WHY SPLIT?
#   Documents are often too long to fit in a single prompt.
#   Splitting into chunks lets us retrieve only RELEVANT parts.

chunks = splitter.split_documents(documents)
print(f"[Step 2] Split into {len(chunks)} chunks (size=300, overlap=50)")


# ── STEP 3 & 4: EMBED AND STORE ──────────────────────────────────────────────

# Embeddings convert text → numerical vectors (lists of numbers)
# Similar text → similar vectors → we can find relevant chunks!
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"[Step 3-4] Embedded {len(chunks)} chunks and stored in FAISS")


# ── STEP 5: CREATE RETRIEVER ─────────────────────────────────────────────────

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# k=3 means: return top 3 most similar chunks for any query
print(f"[Step 5] Retriever ready (top-3 similarity search)")


# ── STEP 6: BUILD THE RAG CHAIN ──────────────────────────────────────────────

def format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant. Answer ONLY from the provided context.
If the answer is not in the context, say "I don't have that information."
Be specific and concise."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)
# How this works:
#   1. "question" = user's input passed through unchanged
#   2. "context" = user's input → retriever (finds similar chunks) → format_docs
#   3. Both go into the prompt template
#   4. Model generates answer from context
#   5. Parser extracts text

print(f"[Step 6] RAG chain built!\n")


# ── STEP 7: ASK QUESTIONS! ────────────────────────────────────────────────────

print("=" * 60)
print("ASKING QUESTIONS")
print("=" * 60)

questions = [
    "How many days of annual leave do I get?",
    "Can I work from another country?",
    "What's the hotel limit for international travel?",
    "What's the company's stock price?",  # Not in documents!
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")


# ── BONUS: STREAMING RAG ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BONUS: Streaming RAG response")
print("=" * 60)

print("Q: What is the remote work equipment policy?")
print("A: ", end="", flush=True)
for chunk in rag_chain.stream("What is the remote work equipment policy?"):
    print(chunk, end="", flush=True)
print()


# ── BONUS: SAVE AND LOAD VECTOR STORE ─────────────────────────────────────────

# Save to disk (reuse without re-embedding)
vectorstore.save_local("my_faiss_index")

# Load from disk later:
# loaded_store = FAISS.load_local("my_faiss_index", embeddings,
#                                   allow_dangerous_deserialization=True)


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. RAG = Retrieval Augmented Generation (give LLM external knowledge)
# 2. Pipeline: Load → Split → Embed → Store → Retrieve → Prompt → Answer
# 3. RecursiveCharacterTextSplitter — chunk with overlap
# 4. FAISS.from_documents() — create vector store from chunks
# 5. vectorstore.as_retriever(k=3) — similarity search interface
# 6. RAG chain: {context: retriever | format, question: passthrough} | prompt | model
# 7. The model answers ONLY from context — no hallucination!
#
# NEXT: step_10_tools.py — Creating tools for agents