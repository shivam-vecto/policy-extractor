# main.py - Working with your available Gemini models
import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import requests

# Load environment variables
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("=" * 50)
    print("WARNING: GEMINI_API_KEY not found in .env file")
    print("Please create .env file with: GEMINI_API_KEY=your_key_here")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    print("=" * 50)
else:
    print(f"Gemini API key loaded")

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Policy PDF Extractor",
    description="Extract data from policy PDFs using Gemini AI",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    upload_dir_exists: bool
    python_version: str

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
    """Use Gemini REST API to extract structured data from policy text"""
    
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    # Use models that are confirmed working
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-flash-latest"
    ]
    
    prompt = f"""
    Extract the following mandatory fields from this property insurance policy document.
    Return ONLY valid JSON with these exact keys (use null if information not found):
    
    Mandatory Fields to Extract:
    
    1. Insurance Company: Name of the insurance company providing coverage
    2. Policy Number: Unique policy identifier number
    3. Property Owner Information: Name(s) of property owner(s)/insured party including any co-owners
    4. Property Location: Complete physical address of the insured property (street, city, state, zip)
    5. Property Construction Date: Year or date when the property was built/constructed
    6. Policy Sum Insured: Total coverage amount for the property
    7. Deductibles: Amount the insured must pay before coverage applies (include perils if specified)
    8. Policy Limits: Maximum coverage amounts for different coverage types
    9. Exclusions: Specific situations, perils, or properties NOT covered by the policy
    
    Format the output as JSON with exactly these keys:
    - insurance_company
    - policy_number
    - property_owner_information
    - property_location
    - property_construction_date
    - policy_sum_insured
    - deductibles
    - policy_limits
    - exclusions
    
    Policy Document Text:
    {pdf_text[:20000]}
    
    Return ONLY valid JSON. Example format:
    {{
        "insurance_company": "State Farm Insurance",
        "policy_number": "SF-123456789",
        "property_owner_information": "John A. Smith and Mary B. Smith",
        "property_location": "123 Main Street, Anytown, CA 90210",
        "property_construction_date": "1995",
        "policy_sum_insured": "$500,000",
        "deductibles": "$1,000 for standard perils, 2% for wind/hail",
        "policy_limits": "Dwelling: $500,000, Contents: $250,000, Liability: $300,000",
        "exclusions": "Flood, Earthquake, Earth Movement, Wear and Tear, Mold, Neglect"
    }}
    
    Make sure to extract as much detail as possible from the policy document.
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,  # Lower temperature for more precise extraction
            "maxOutputTokens": 8192
        }
    }
    
    # Try each model until one works
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
                        extracted_json = json.loads(json_match.group())
                        extracted_json['_model_used'] = model_name
                        return extracted_json
                    else:
                        return {"error": "Could not parse JSON", "raw_text": text_response[:500], "model_used": model_name}
                else:
                    return {"error": "No response from Gemini", "model_used": model_name}
                    
        except Exception as e:
            continue
    
    return {"error": "All Gemini models failed. Please check your API key and permissions."}

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
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(content)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Extract data using Gemini
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
        "message": "Policy PDF Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "Upload PDF and extract policy data",
            "GET /health": "Check API health",
            "GET /models": "List available models"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        gemini_configured=bool(GEMINI_API_KEY),
        upload_dir_exists=UPLOAD_DIR.exists(),
        python_version="3.9.0"
    )

@app.get("/models")
async def list_models():
    """List all available Gemini models"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json()
            # Filter for generateContent models
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

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            })
    return {"files": files, "count": len(files)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🚀 Starting Policy PDF Extractor API")
    print("=" * 50)
    print(f"📁 Uploads directory: {UPLOAD_DIR.absolute()}")
    print(f"🔑 Gemini API configured: {bool(GEMINI_API_KEY)}")
    print(f"📚 API Documentation: http://127.0.0.1:8000/docs")
    print(f"🩺 Health Check: http://127.0.0.1:8000/health")
    print(f"📋 Available Models: http://127.0.0.1:8000/models")
    print("=" * 50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)