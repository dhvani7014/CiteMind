from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.app.services.pdf_service import extract_text_from_pdf
from backend.app.services.chunking_service import chunk_text
from backend.app.services.embedding_service import generate_embeddings

app = FastAPI(
    title="CiteMind API",
    description="AI-powered research paper assistant using RAG",
    version="0.1.0"
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

    text_preview = extracted_data["full_text"][:1000]
    chunk_preview = chunks[:3]

    embedding_dimension = 0
    if embedded_chunks:
        embedding_dimension = len(embedded_chunks[0]["embedding"])

    return {
        "message": "PDF uploaded, text extracted, chunked, and embedded successfully",
        "filename": file.filename,
        "file_path": str(file_path),
        "total_pages": extracted_data["total_pages"],
        "total_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": embedding_dimension,
        "text_preview": text_preview,
        "chunk_preview": chunk_preview
    }