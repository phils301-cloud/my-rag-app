"""
upload_docs.py — Upload your document to Pinecone
==================================================
Run this ONCE before starting the app.

Usage:
    python upload_docs.py --file your_document.pdf

Supports: PDF, TXT, DOCX
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# ── Load API keys from .env ────────────────────────────────────
load_dotenv()

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX]):
    print("\n ERROR: Missing keys in .env file.")
    print("  Make sure your .env has:")
    print("    OPENAI_API_KEY=sk-...")
    print("    PINECONE_API_KEY=...")
    print("    PINECONE_INDEX=story111\n")
    sys.exit(1)

# ── Imports ────────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
except ImportError as e:
    print(f"\n ERROR: Missing package — {e}")
    print("  Run: pip install langchain langchain-community langchain-openai langchain-pinecone langchain-text-splitters pinecone-client pypdf docx2txt python-dotenv\n")
    sys.exit(1)

# ── Parse arguments ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Upload a document to Pinecone")
parser.add_argument("--file", required=True, help="Path to your document (PDF, TXT, or DOCX)")
parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk (default: 500)")
parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks (default: 50)")
args = parser.parse_args()

# ── Check file exists ──────────────────────────────────────────
if not os.path.exists(args.file):
    print(f"\n ERROR: File not found: {args.file}")
    print("  Make sure the file is in the same folder or provide the full path.\n")
    sys.exit(1)

# ── Load document ──────────────────────────────────────────────
print(f"\n Loading document: {args.file}")

ext = args.file.lower().split(".")[-1]

if ext == "pdf":
    loader = PyPDFLoader(args.file)
elif ext == "txt":
    loader = TextLoader(args.file, encoding="utf-8")
elif ext == "docx":
    loader = Docx2txtLoader(args.file)
else:
    print(f"\n ERROR: Unsupported file type .{ext}")
    print("  Supported: .pdf  .txt  .docx\n")
    sys.exit(1)

documents = loader.load()
print(f" Loaded {len(documents)} page(s)")

# ── Split into chunks ──────────────────────────────────────────
print(f" Splitting into chunks (size={args.chunk_size}, overlap={args.chunk_overlap})...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap
)
chunks = splitter.split_documents(documents)
print(f" Created {len(chunks)} chunks")

# ── Set up embeddings ──────────────────────────────────────────
print(" Setting up OpenAI embeddings (text-embedding-3-small, 512 dimensions)...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

# ── Upload to Pinecone ─────────────────────────────────────────
print(f" Uploading to Pinecone index: {PINECONE_INDEX}")
print(" This may take 1-2 minutes depending on document size...")

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=PINECONE_INDEX
)

print(f"\n SUCCESS! {len(chunks)} chunks uploaded to Pinecone index '{PINECONE_INDEX}'")
print(" Verify: go to app.pinecone.io > your index > vector count should be > 0")
print(" You can now run: docker run --env-file .env -p 7860:7860 rag-app\n")
