from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# the functon that handels explaination and answering
from paper_processor import process_pdf, answer_question

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Welcome to the API"}

# Route to accept a PDF and return summary or status
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Read file content as bytes
        content = await file.read()
        
        # Process the PDF content
        result = process_pdf(content)
        
        return {"summary": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Route to accept a query and return the answer
class QuestionRequest(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question(payload: QuestionRequest):
    answer = answer_question(payload.query)
    return {"answer": answer}
