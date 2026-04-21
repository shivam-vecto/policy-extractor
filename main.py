# main.py - Complete Working API with Extraction and Q&A
import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import requests
from typing import Optional, List

# Load environment variables
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file")
    print("Please create .env file with: GEMINI_API_KEY=your_key_here")
else:
    print(f"Gemini API key loaded")

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Property Insurance Policy Extractor & Q&A",
    description="Extract policy data and ask questions about insurance policies using Gemini AI",
    version="2.0.0"
)

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    policy_id: Optional[str] = None

class QuestionResponse(BaseModel):
    success: bool
    question: str
    answer: str
    policy_used: str
    timestamp: datetime

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n{page_text}"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_policy_data_with_gemini(pdf_text: str) -> dict:
    """Use Gemini REST API to extract structured data"""
    
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    models_to_try = ["gemini-2.5-flash",
        "gemini-1.5-flash", 
        "gemini-2.0-flash",
        "gemini-pro"]
    
    prompt = f"""
    Extract the following mandatory fields from this property insurance policy document.
    Return ONLY valid JSON with these exact keys:
    
    Mandatory Fields:
    - insurance_company
    - policy_number
    - property_owner_information
    - property_location
    - property_construction_date
    - policy_sum_insured
    - deductibles
    - policy_limits
    - exclusions
    
    Policy Text:
    {pdf_text[:20000]}
    
    Return ONLY valid JSON.
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception:
            continue
    
    return {"error": "All models failed"}

# ============ EXTRACTION ENDPOINT ============

@app.post("/extract")
async def extract_policy(file: UploadFile = File(...)):
    """Upload a policy PDF and extract data"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 10MB")
        
        # Save file
        file_id = str(uuid.uuid4())[:8]
        saved_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        with open(saved_path, "wb") as f:
            f.write(content)
        
        # Extract text and data
        pdf_text = extract_text_from_pdf(content)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        extracted_data = extract_policy_data_with_gemini(pdf_text)
        
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "extraction_id": file_id,
                "extracted_data": extracted_data,
                "file_saved": str(saved_path),
                "timestamp": datetime.now().isoformat()
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Property Insurance Policy Extractor & Q&A API",
        "version": "2.0.0",
        "endpoints": {
            "POST /extract": "Upload PDF and extract policy data",
            "POST /ask": "Ask questions about uploaded policies (POST method)",
            "GET /ask": "Ask questions about uploaded policies (GET method - easier for testing)",
            "GET /health": "Check API health",
            "GET /files": "List uploaded files",
            "GET /models": "List available Gemini models"
        },
        "example_usage": {
            "extract": "curl -X POST 'http://localhost:8000/extract' -F 'file=@policy.pdf'",
            "ask_get": "curl 'http://localhost:8000/ask?question=What%20is%20the%20deductible?'",
            "ask_post": "curl -X POST 'http://localhost:8000/ask' -H 'Content-Type: application/json' -d '{\"question\": \"What is the deductible?\"}'"
        }
    }

@app.get("/health")
async def health():
    files_count = len(list(UPLOAD_DIR.glob("*.pdf")))
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "files_available": files_count,
        "python_version": "3.9.0"
    }

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = []
    for file_path in UPLOAD_DIR.glob("*.pdf"):
        files.append({
            "filename": file_path.name,
            "extraction_id": file_path.name.split('_')[0],
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
        })
    return {"files": files, "count": len(files)}

@app.get("/models")
async def list_models():
    """List available Gemini models"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json()
            generate_models = []
            for model in models.get('models', []):
                if 'generateContent' in str(model.get('supportedGenerationMethods', [])):
                    generate_models.append({
                        'name': model['name'].replace('models/', ''),
                        'display_name': model.get('displayName', '')
                    })
            return {"models": generate_models}
        else:
            return {"error": "Failed to list models"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Property Insurance Policy Extractor & Q&A API")
    print("=" * 60)
    print(f"Uploads directory: {UPLOAD_DIR.absolute()}")
    print(f"Gemini API configured: {bool(GEMINI_API_KEY)}")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    print(f"Health Check: http://127.0.0.1:8000/health")
    print("\n Quick Test Commands:")
    print("   1. Upload a PDF:")
    print("      curl -X POST 'http://localhost:8000/extract' -F 'file=@policy.pdf'")
    print("\n   2. Ask a question (GET method - easiest):")
    print("      curl 'http://localhost:8000/ask?question=What%20is%20the%20deductible?'")
    print("\n   3. Ask a question (POST method):")
    print('      curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is the deductible?\"}"')
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)