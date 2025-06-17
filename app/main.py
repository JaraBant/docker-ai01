from fastapi import FastAPI, UploadFile, File
from app.rag_engine import PDFRAGEngine

app = FastAPI()
engine = None

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open("uploaded/uploaded.pdf", "wb") as f:
        f.write(contents)
    global engine
    engine = PDFRAGEngine("uploaded/uploaded.pdf")
    return {"status": "PDF uploaded and indexed"}

@app.get("/ask/")
def ask(question: str):
    if not engine:
        return {"error": "No PDF uploaded"}
    answer = engine.query(question)
    return {"answer": answer}
