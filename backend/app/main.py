from fastapi import FastAPI

app = FastAPI(
    title="CiteMind API",
    description="AI-powered research paper assistant using RAG",
    version="0.1.0"
)

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