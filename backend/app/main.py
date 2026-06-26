from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.app.services.pdf_service import extract_text_from_pdf

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

    text_preview = extracted_data["full_text"][:1000]

    return {
        "message": "PDF uploaded and text extracted successfully",
        "filename": file.filename,
        "file_path": str(file_path),
        "total_pages": extracted_data["total_pages"],
        "text_preview": text_preview
    }