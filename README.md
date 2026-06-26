# CiteMind

CiteMind is an AI-powered research paper assistant that helps users upload academic papers, ask questions, and receive citation-grounded answers.

The project uses Retrieval-Augmented Generation, also known as RAG, to search through uploaded PDFs and generate answers based only on the provided documents.

## Why I Built This

I am building CiteMind to learn practical AI engineering concepts such as document processing, embeddings, vector databases, RAG pipelines, backend APIs, frontend development, and deployment.


## Core Features

- Upload research paper PDFs
- Extract and chunk text from documents
- Generate embeddings for document chunks
- Store chunks in a vector database
- Ask questions about uploaded papers
- Generate answers using RAG
- Show source citations from the paper
- Evaluate answer quality
- Detect possible hallucinations
- Provide a clean frontend chat interface
- Deploy with Docker

## Tech Stack

- Python
- FastAPI
- LangChain or LlamaIndex
- FAISS or ChromaDB
- OpenAI API
- React or Next.js
- Docker
- GitHub

## Project Status

Currently in development.

## Planned Milestones

1. Project setup
2. FastAPI backend
3. PDF upload endpoint
4. PDF text extraction
5. Text chunking
6. Embedding generation
7. Vector database integration
8. RAG question answering
9. Source citation display
10. Frontend chat interface
11. Evaluation dataset
12. Docker deployment