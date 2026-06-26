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

## Running the Backend Locally

Follow these steps to run the CiteMind backend on your local machine.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn backend.app.main:app --reload
```

### 5. Open the backend in your browser

```text
http://127.0.0.1:8000
```

### 6. Open the API documentation

```text
http://127.0.0.1:8000/docs
```

## Current Backend Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Returns a welcome message |
| `/health` | GET | Checks if the backend is running |
| `/upload-pdf` | POST | Uploads a research paper PDF |

## PDF Upload Feature

CiteMind now supports uploading PDF files through the FastAPI backend.

To test the upload endpoint:

1. Run the backend server.
2. Open `http://127.0.0.1:8000/docs`.
3. Expand the `/upload-pdf` endpoint.
4. Click `Try it out`.
5. Upload a PDF file.
6. Click `Execute`.

Uploaded files are stored locally in `backend/uploads/`, which is ignored by GitHub.

## Text Chunking for RAG

CiteMind now supports splitting extracted PDF text into smaller overlapping chunks.

Chunking is an important step in Retrieval-Augmented Generation because large documents cannot be passed directly into an AI model all at once. Instead, the document is broken into smaller sections that can later be embedded, stored in a vector database, and retrieved based on a user's question.

The current chunking system uses:

- Chunk size: 1000 characters
- Overlap: 200 characters

The overlap helps preserve context between nearby chunks.## Text Chunking for RAG

CiteMind now supports splitting extracted PDF text into smaller overlapping chunks.

Chunking is an important step in Retrieval-Augmented Generation because large documents cannot be passed directly into an AI model all at once. Instead, the document is broken into smaller sections that can later be embedded, stored in a vector database, and retrieved based on a user's question.

The current chunking system uses:

- Chunk size: 1000 characters
- Overlap: 200 characters

The overlap helps preserve context between nearby chunks.

## Embedding Generation

CiteMind now supports generating embeddings for text chunks using the `all-MiniLM-L6-v2` sentence-transformers model.

Embeddings convert text into numerical vectors. These vectors will later be stored in a vector database and used to retrieve the most relevant document chunks when a user asks a question.

Current embedding setup:

- Model: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Input: chunked PDF text
- Output: vector representation for each chunk

## Vector Database Storage

CiteMind now stores embedded document chunks in a local ChromaDB vector database.

This allows the app to persist document chunks and their embeddings so they can later be retrieved during question answering.

Current vector database setup:

- Vector DB: ChromaDB
- Storage mode: Local persistent storage
- Collection name: `citemind_papers`
- Stored data: chunk text, embeddings, filename, chunk ID, and character positions

The local database is stored in `backend/chroma_db/`, which is ignored by GitHub.

## Similarity Search Retrieval

CiteMind now supports retrieving relevant document chunks from the vector database.

When a user asks a question, CiteMind:

1. Converts the question into an embedding.
2. Searches ChromaDB for the most similar document chunks.
3. Returns the top matching chunks with metadata.
4. Includes filename, chunk ID, character positions, and similarity distance.

This is the retrieval part of Retrieval-Augmented Generation.

## Grounded Answer Generation

CiteMind now supports a basic RAG-style answer endpoint through `/ask`.

When a user asks a question, CiteMind:

1. Converts the question into an embedding.
2. Searches the vector database for relevant chunks.
3. Uses the retrieved chunks to generate a grounded answer.
4. Returns the answer with source metadata.

The current answer generator is intentionally conservative. It only uses retrieved document text and does not use an external LLM yet. This helps reduce hallucination and keeps the first RAG pipeline simple and transparent.