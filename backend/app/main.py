from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
    clear_vector_store,
    get_document_intro_chunks,
)
from backend.app.services.answer_service import generate_grounded_answer
from backend.app.services.llm_service import generate_llm_answer


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


app = FastAPI(
    title="CiteMind API",
    description="AI-powered research paper assistant using RAG",
    version="0.1.0",
    docs_url=None,
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ACTIVE_DOCUMENT = {
    "filename": None
}


class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_chunks": get_collection_count(),
        },
    )


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def api_docs(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="docs.html",
        context={
            "total_chunks": get_collection_count(),
        },
    )


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def user_app(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="app.html",
        context={
            "total_chunks": get_collection_count(),
        },
    )


@app.get("/stats", response_class=HTMLResponse, include_in_schema=False)
def stats_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="stats.html",
        context={
            "total_chunks": get_collection_count(),
            "active_document": ACTIVE_DOCUMENT["filename"],
            "collection_name": "citemind_papers",
        },
    )


@app.get("/swagger", response_class=HTMLResponse, include_in_schema=False)
def custom_swagger_ui():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CiteMind Swagger Console</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
            <link rel="stylesheet" href="/static/swagger.css">
        </head>
        <body>
            <div class="swagger-header">
                <div>
                    <h1>CiteMind API Playground</h1>
                    <p>Test the RAG backend endpoints for PDF upload, vector search, and citation-grounded answers.</p>
                </div>
                <a href="/docs">Back to API Console</a>
            </div>

            <div id="swagger-ui"></div>

            <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
            <script>
                window.onload = function() {
                    SwaggerUIBundle({
                        url: "/openapi.json",
                        dom_id: "#swagger-ui",
                        deepLinking: true,
                        layout: "BaseLayout",
                        docExpansion: "none",
                        defaultModelsExpandDepth: -1,
                        displayRequestDuration: true,
                        filter: true,
                        tryItOutEnabled: true
                    });
                };
            </script>
        </body>
        </html>
        """
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CiteMind API",
        "active_document": ACTIVE_DOCUMENT["filename"],
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
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
        filename=file.filename,
    )

    ACTIVE_DOCUMENT["filename"] = file.filename

    text_preview = extracted_data["full_text"][:1000]
    chunk_preview = chunks[:3]

    embedding_dimension = 0
    if embedded_chunks:
        embedding_dimension = len(embedded_chunks[0]["embedding"])

    return {
        "message": "PDF uploaded, text extracted, chunked, embedded, and stored successfully",
        "filename": file.filename,
        "active_document": ACTIVE_DOCUMENT["filename"],
        "file_path": str(file_path),
        "total_pages": extracted_data["total_pages"],
        "total_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": embedding_dimension,
        "vector_store": vector_store_result,
        "total_chunks_in_vector_db": get_collection_count(),
        "text_preview": text_preview,
        "chunk_preview": chunk_preview,
    }


@app.post("/search")
def search_documents(request: SearchRequest):
    if get_collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document chunks found. Please upload a PDF first.",
        )

    if ACTIVE_DOCUMENT["filename"] is None:
        raise HTTPException(
            status_code=400,
            detail="No active document found. Please upload a PDF first.",
        )

    query_embedding = generate_query_embedding(request.question)

    retrieved_chunks = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=request.top_k,
        filename=ACTIVE_DOCUMENT["filename"],
    )

    return {
        "question": request.question,
        "top_k": request.top_k,
        "active_document": ACTIVE_DOCUMENT["filename"],
        "retrieved_chunks": retrieved_chunks,
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    if get_collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document chunks found. Please upload a PDF first.",
        )

    if ACTIVE_DOCUMENT["filename"] is None:
        raise HTTPException(
            status_code=400,
            detail="No active document found. Please upload a PDF first.",
        )

    query_embedding = generate_query_embedding(request.question)

    retrieved_chunks = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=request.top_k,
        filename=ACTIVE_DOCUMENT["filename"],
    )

    question_lower = request.question.lower()

    context_boost_keywords = [
        "main idea",
        "main point",
        "summary",
        "summarize",
        "summarise",
        "abstract",
        "overview",
        "what is this paper about",
        "what does this paper discuss",
        "what is the paper saying",
        "explain this paper",
        "paper breakdown",
        "break down this paper",
        "high level explanation",
        "simple explanation",
        "easy explanation",
        "author",
        "authors",
        "who wrote",
        "who is the author",
        "title",
        "paper title",
        "publication",
        "published",
        "conference",
        "journal",
        "year",
        "date",
        "affiliation",
        "university",
        "institution",
        "problem",
        "problem solved",
        "research problem",
        "research question",
        "motivation",
        "why was this paper written",
        "why is this important",
        "gap",
        "research gap",
        "objective",
        "goal",
        "purpose",
        "contribution",
        "contributions",
        "main contribution",
        "novelty",
        "what is new",
        "innovation",
        "claim",
        "claims",
        "key idea",
        "key ideas",
        "method",
        "methods",
        "methodology",
        "approach",
        "technique",
        "algorithm",
        "architecture",
        "model",
        "framework",
        "pipeline",
        "system design",
        "how does it work",
        "dataset",
        "data",
        "training data",
        "test data",
        "experiment",
        "experiments",
        "experimental setup",
        "evaluation",
        "metrics",
        "baseline",
        "comparison",
        "benchmark",
        "performance",
        "accuracy",
        "error rate",
        "results",
        "findings",
        "discussion",
        "conclusion",
        "limitations",
        "weakness",
        "weaknesses",
        "future work",
        "next steps",
        "open problems",
        "figure",
        "figures",
        "table",
        "tables",
        "equation",
        "formula",
        "diagram",
        "graph",
        "chart",
        "important",
        "important points",
        "important concepts",
        "key points",
        "key concepts",
        "main concepts",
        "study",
        "study guide",
        "quiz",
        "exam",
        "test",
        "revision",
        "review",
        "notes",
        "prepare",
        "prep",
        "learn",
        "teach me",
        "for my quiz",
        "for my exam",
        "for my test",
        "what should i remember",
        "what should i know",
        "possible questions",
        "interview questions",
        "practice questions",
        "strengths",
        "pros",
        "advantages",
        "disadvantages",
        "critique",
        "critical analysis",
        "impact",
        "significance",
        "real world use",
        "applications",
        "use cases",
        "related work",
        "references",
        "citation",
        "citations",
        "prior work",
        "previous work",
        "compare",
        "comparison with",
    ]

    if any(keyword in question_lower for keyword in context_boost_keywords):
        intro_chunks = get_document_intro_chunks(
            filename=ACTIVE_DOCUMENT["filename"],
            limit=3,
        )

        combined_chunks = intro_chunks + retrieved_chunks

        unique_chunks = []
        seen_chunk_ids = set()

        for chunk in combined_chunks:
            metadata = chunk.get("metadata", {})
            chunk_id = metadata.get("chunk_id")

            if chunk_id not in seen_chunk_ids:
                unique_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

        retrieved_chunks = unique_chunks[: request.top_k + 3]

    llm_result = generate_llm_answer(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
    )

    answer_result = generate_grounded_answer(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
    )

    if llm_result:
        final_answer = llm_result["answer"]
        answer_mode = "llm"
        llm_provider = llm_result["llm_provider"]
        llm_model = llm_result["llm_model"]
    else:
        final_answer = answer_result["answer"]
        answer_mode = "local_fallback"
        llm_provider = None
        llm_model = None

    return {
        "question": request.question,
        "top_k": request.top_k,
        "active_document": ACTIVE_DOCUMENT["filename"],
        "answer": final_answer,
        "answer_mode": answer_mode,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "citations": answer_result["citations"],
        "sources": answer_result["sources"],
        "retrieved_chunks": retrieved_chunks,
    }


@app.delete("/vector-store/clear")
def clear_stored_chunks():
    ACTIVE_DOCUMENT["filename"] = None
    return clear_vector_store()


@app.get("/vector-store/stats")
def vector_store_stats():
    return {
        "collection_name": "citemind_papers",
        "total_chunks": get_collection_count(),
        "active_document": ACTIVE_DOCUMENT["filename"],
    }