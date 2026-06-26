from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from backend.app.services.pdf_service import extract_text_from_pdf
from backend.app.services.chunking_service import chunk_text
from backend.app.services.embedding_service import (
    generate_embeddings,
    generate_query_embedding,
)
from backend.app.services.vector_store_service import (
    store_chunks_in_vector_db,
    search_similar_chunks,
    get_collection_count,
)

app = FastAPI(
    title="CiteMind API",
    description="AI-powered research paper assistant using RAG",
    version="0.1.0"
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {
        "message": "Welcome to CiteMind",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CiteMind API"
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    extracted_data = extract_text_from_pdf(str(file_path))
    chunks = chunk_text(extracted_data["full_text"])
    embedded_chunks = generate_embeddings(chunks)

    vector_store_result = store_chunks_in_vector_db(
        embedded_chunks=embedded_chunks,
        filename=file.filename
    )

    text_preview = extracted_data["full_text"][:1000]
    chunk_preview = chunks[:3]

    embedding_dimension = 0
    if embedded_chunks:
        embedding_dimension = len(embedded_chunks[0]["embedding"])

    return {
        "message": "PDF uploaded, text extracted, chunked, embedded, and stored successfully",
        "filename": file.filename,
        "file_path": str(file_path),
        "total_pages": extracted_data["total_pages"],
        "total_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": embedding_dimension,
        "vector_store": vector_store_result,
        "total_chunks_in_vector_db": get_collection_count(),
        "text_preview": text_preview,
        "chunk_preview": chunk_preview
    }


@app.post("/search")
def search_documents(request: SearchRequest):
    if get_collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document chunks found. Please upload a PDF first."
        )

    query_embedding = generate_query_embedding(request.question)

    retrieved_chunks = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=request.top_k
    )

    return {
        "question": request.question,
        "top_k": request.top_k,
        "retrieved_chunks": retrieved_chunks
    }


@app.get("/vector-store/stats")
def vector_store_stats():
    return {
        "collection_name": "citemind_papers",
        "total_chunks": get_collection_count()
    }