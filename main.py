# main.py - Universal Insurance Policy Extractor (Handles All Policy Types)
import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import requests

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env file")

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Universal Insurance Policy Extractor API",
    description="Extract data from any type of insurance policy PDF (Health, Home, Auto, Life, etc.)",
    version="3.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def detect_policy_type(pdf_text: str) -> Dict[str, Any]:
    """Detect the type of insurance policy"""
    text_lower = pdf_text.lower()
    
    # Define patterns for different policy types
    patterns = {
        "health": {
            "keywords": ["health insurance", "medical", "deductible", "copay", "coinsurance", "ppo", "hmo", "prescription", "out-of-pocket"],
            "fields": ["deductible", "copay", "coinsurance", "out_of_pocket_max", "primary_care", "specialist", "emergency"]
        },
        "homeowners": {
            "keywords": ["homeowners", "dwelling", "property", "hazard", "personal property", "other structures", "loss of use"],
            "fields": ["dwelling_coverage", "personal_property", "deductible", "property_address", "year_built"]
        },
        "auto": {
            "keywords": ["auto", "car", "vehicle", "collision", "comprehensive", "liability", "uninsured motorist"],
            "fields": ["vehicle", "vin", "collision_deductible", "liability_coverage", "uninsured_motorist"]
        },
        "life": {
            "keywords": ["life insurance", "death benefit", "term life", "whole life", "beneficiary", "face amount"],
            "fields": ["death_benefit", "premium", "beneficiary", "term_years", "cash_value"]
        },
        "renters": {
            "keywords": ["renters", "rental", "tenant", "personal property", "loss of use"],
            "fields": ["personal_property", "loss_of_use", "liability", "rental_address"]
        },
        "disability": {
            "keywords": ["disability", "income protection", "elimination period", "benefit period", "own occupation"],
            "fields": ["monthly_benefit", "elimination_period", "benefit_period", "occupation"]
        },
        "dental": {
            "keywords": ["dental", "orthodontic", "cleaning", "crown", "root canal", "fluoride"],
            "fields": ["annual_maximum", "preventive_coverage", "basic_coverage", "major_coverage"]
        }
    }
    
    # Score each policy type
    scores = {}
    for policy_type, info in patterns.items():
        score = sum(1 for keyword in info["keywords"] if keyword in text_lower)
        scores[policy_type] = score
    
    # Get the best match
    best_match = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
    
    return {
        "policy_type": best_match,
        "confidence": scores[best_match] if best_match != "unknown" else 0,
        "possible_types": [{"type": t, "score": s} for t, s in scores.items() if s > 0]
    }

def create_extraction_prompt(pdf_text: str, policy_type: str) -> str:
    """Create a tailored extraction prompt based on policy type"""
    
    base_prompt = f"""Extract ALL insurance policy information from this document and return as a clean JSON object.

Document type: {policy_type.upper()} INSURANCE POLICY

Document text:
{pdf_text[:30000]}

"""
    
    if policy_type == "health":
        return base_prompt + """
Extract these SPECIFIC health insurance fields:

POLICY INFO:
- insurance_company
- policy_number
- group_number

POLICYHOLDER:
- policyholder_name
- date_of_birth
- gender
- address

COVERAGE PERIOD:
- effective_date
- expiration_date

PLAN DETAILS:
- plan_type (PPO, HMO, EPO, POS)
- network_name

COST SHARING:
- annual_deductible_individual
- annual_deductible_family
- out_of_pocket_max_individual
- out_of_pocket_max_family
- coinsurance_percentage

COPAYMENTS:
- primary_care_copay
- specialist_copay
- emergency_copay
- urgent_care_copay
- telehealth_copay

PREMIUM:
- monthly_premium
- annual_premium

PRESCRIPTION DRUGS:
- generic_copay
- preferred_brand_copay
- non_preferred_brand_copay
- specialty_drug_coverage

OTHER COVERAGES:
- preventive_care_coverage
- mental_health_coverage
- maternity_coverage
- vision_coverage
- dental_coverage

Return as JSON with proper nested structure for copayments and prescription drugs.
"""
    
    elif policy_type == "homeowners":
        return base_prompt + """
Extract these SPECIFIC homeowners insurance fields:

PROPERTY INFO:
- insurance_company
- policy_number
- named_insured
- property_address
- year_built
- square_footage
- construction_type

COVERAGES:
- dwelling_coverage (amount)
- other_structures_coverage (amount)
- personal_property_coverage (amount)
- loss_of_use_coverage (amount)
- personal_liability_coverage (amount)
- medical_payments_coverage (amount)

DEDUCTIBLES:
- standard_deductible
- wind_hail_deductible (include percentage)
- water_backup_deductible

POLICY PERIOD:
- effective_date
- expiration_date

PREMIUM:
- annual_premium
- payment_plan

MORTGAGE INFO:
- mortgagee

Return as JSON with numeric values (remove $ and commas).
"""
    
    elif policy_type == "auto":
        return base_prompt + """
Extract these SPECIFIC auto insurance fields:

POLICY INFO:
- insurance_company
- policy_number
- named_insured
- vehicle_year
- vehicle_make
- vehicle_model
- vin (Vehicle Identification Number)

COVERAGES:
- bodily_injury_liability
- property_damage_liability
- collision_coverage
- comprehensive_coverage
- uninsured_motorist_coverage
- medical_payments_coverage

DEDUCTIBLES:
- collision_deductible
- comprehensive_deductible

DATES:
- effective_date
- expiration_date

PREMIUM:
- annual_premium
- monthly_premium

Return as JSON with numeric values (remove $ and commas).
"""
    
    elif policy_type == "life":
        return base_prompt + """
Extract these SPECIFIC life insurance fields:

POLICY INFO:
- insurance_company
- policy_number
- policy_type (Term, Whole, Universal, Variable)

INSURED:
- insured_name
- date_of_birth
- tobacco_use (Yes/No)

COVERAGE:
- death_benefit_amount
- riders (list any riders)

BENEFICIARY:
- primary_beneficiary
- contingent_beneficiary

PREMIUM:
- annual_premium
- monthly_premium
- premium_guarantee_period

DATES:
- effective_date
- expiration_date (if term policy)

Return as JSON.
"""
    
    else:
        # Generic prompt for unknown policy types
        return base_prompt + """
Extract ALL information from this insurance policy and organize it into a structured JSON.

Include ANY fields you find such as:
- Insurance company names
- Policy numbers
- Names (policyholder, insured, beneficiary)
- Dates (effective, expiration, issue)
- Coverage amounts and limits
- Deductibles
- Premium amounts
- Addresses
- Any other policy details

Create field names that are descriptive and use snake_case.
Extract all numeric values as numbers (remove $, commas, %).
Return ONLY the JSON object with all extracted data.
"""

def extract_with_gemini(pdf_text: str, policy_type: Optional[str] = None) -> Dict[str, Any]:
    """Extract data using Gemini AI"""
    
    if not GEMINI_API_KEY:
        # Use regex extraction as fallback
        return extract_universal_regex(pdf_text)
    
    # Detect policy type if not provided
    if not policy_type:
        detection = detect_policy_type(pdf_text)
        policy_type = detection["policy_type"]
        print(f"Detected policy type: {policy_type} (confidence: {detection['confidence']})")
    
    # Create tailored prompt
    prompt = create_extraction_prompt(pdf_text, policy_type)
    
    # Add strict formatting instructions
    prompt += """

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON - no markdown, no explanatory text, no code blocks
2. Remove all currency symbols ($, €, £) and commas from numbers
3. Convert percentages to numeric values (e.g., "20%" -> 20)
4. Use consistent date format: YYYY-MM-DD where possible
5. If a field isn't found, omit it from the JSON
6. Ensure all JSON property names are in double quotes
7. Use nested objects for related fields (e.g., "copayments": {"primary_care": 25})"""

    models_to_try = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-2.0-flash"
    ]
    
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4096,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        try:
            print(f"🔄 Trying {model_name}...")
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{[\s\S]*\}', text_response)
                    if json_match:
                        json_str = json_match.group()
                        
                        # Clean and parse JSON
                        try:
                            extracted = json.loads(json_str)
                            # Add metadata
                            extracted["_metadata"] = {
                                "policy_type": policy_type,
                                "extraction_method": "gemini",
                                "model_used": model_name,
                                "extraction_timestamp": datetime.now().isoformat()
                            }
                            print(f"✅ Successfully extracted {len(extracted)} fields")
                            return extracted
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON parse error: {e}")
                            continue
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error with {model_name}: {str(e)}")
            continue
    
    # Fallback to regex
    print("🔄 Falling back to universal regex extraction...")
    return extract_universal_regex(pdf_text)

def extract_universal_regex(pdf_text: str) -> Dict[str, Any]:
    """Universal regex extraction that works with any policy"""
    
    extracted = {
        "_metadata": {
            "extraction_method": "regex_fallback",
            "extraction_timestamp": datetime.now().isoformat()
        }
    }
    
    # Common patterns across all policy types
    patterns = {
        # Basic information
        "insurance_company": r"(?:INSURANCE\s+COMPANY|Insurance\s+Company|Company)[:\s]+([^\n]+)",
        "policy_number": r"(?:POLICY\s+NUMBER|Policy\s+Number|Policy\s+#|Policy\s+No)[:\s]+([A-Z0-9\-]+)",
        "group_number": r"(?:GROUP\s+NUMBER|Group\s+Number|Group\s+#)[:\s]+([A-Z0-9\-]+)",
        
        # Names
        "policyholder_name": r"(?:Name|Policyholder|Insured|Named\s+Insured)[:\s]+([^\n]+)",
        "insured_name": r"(?:Insured|Member)[:\s]+([^\n]+)",
        
        # Dates
        "effective_date": r"(?:Effective\s+Date|Coverage\s+From|Start\s+Date)[:\s]+([^\n]+)",
        "expiration_date": r"(?:Expiration\s+Date|Coverage\s+To|End\s+Date)[:\s]+([^\n]+)",
        
        # Address
        "address": r"(?:Address|Mailing\s+Address)[:\s]+([^\n]+)",
        
        # Financial
        "annual_premium": r"(?:Annual\s+Premium|Yearly\s+Premium)[:\s]*\$?([\d,]+\.?\d*)",
        "monthly_premium": r"(?:Monthly\s+Premium)[:\s]*\$?([\d,]+\.?\d*)",
        "deductible": r"(?:Deductible)[:\s]*\$?([\d,]+\.?\d*)",
        
        # Health-specific
        "copay": r"(?:Copay|Copayment)[:\s]*\$?([\d,]+)",
        "coinsurance": r"(?:Coinsurance)[:\s]*(\d+)%",
        
        # Coverage amounts
        "coverage_amount": r"(?:Coverage|Limit|Benefit)[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        
        # Dates of birth
        "date_of_birth": r"(?:Date\s+of\s+Birth|DOB|Birth\s+Date)[:\s]+([^\n]+)",
        
        # Phone numbers
        "phone": r"(?:Phone|Contact|Claims)[:\s]+([\d\-\(\)\+]+)"
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, pdf_text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            # Clean up value
            value = re.sub(r'\s+', ' ', value)
            # Convert to number if it looks like one
            if field in ["annual_premium", "monthly_premium", "deductible", "copay", "coverage_amount"]:
                value = re.sub(r'[^\d.]', '', value)
                try:
                    value = float(value) if '.' in value else int(value)
                except:
                    pass
            extracted[field] = value
    
    # Add policy type detection
    detection = detect_policy_type(pdf_text)
    extracted["_detected_policy_type"] = detection["policy_type"]
    
    # Add raw text preview for debugging
    if len(extracted) <= 2:  # If we barely extracted anything
        extracted["_raw_text_preview"] = pdf_text[:500]
        extracted["_message"] = "Limited data extracted. Consider using Gemini API for better results."
    
    return extracted

@app.post("/extract")
async def extract_policy(
    file: UploadFile = File(...),
    policy_type: Optional[str] = Form(None)
):
    """
    Upload any insurance policy PDF and extract all data
    
    Parameters:
    - file: PDF file to extract
    - policy_type: Optional hint about policy type (health, homeowners, auto, life, renters, disability, dental)
    """
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file
        content = await file.read()
        
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 15MB")
        
        # Generate unique ID
        extraction_id = str(uuid.uuid4())[:8]
        
        # Save file
        saved_path = UPLOAD_DIR / f"{extraction_id}_{file.filename}"
        with open(saved_path, "wb") as f:
            f.write(content)
        
        # Extract text from PDF
        print(f"📄 Extracting text from {file.filename}...")
        pdf_text = extract_text_from_pdf(content)
        
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        print(f"📝 Extracted {len(pdf_text)} characters")
        
        # Detect policy type if not provided
        if not policy_type:
            detection = detect_policy_type(pdf_text)
            policy_type = detection["policy_type"]
            print(f"🔍 Detected policy type: {policy_type}")
        
        # Extract data
        extracted_data = extract_with_gemini(pdf_text, policy_type)
        
        # Save result
        result_path = UPLOAD_DIR / f"{extraction_id}_result.json"
        result = {
            "extraction_id": extraction_id,
            "filename": file.filename,
            "policy_type_hint": policy_type,
            "extracted_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        # Prepare response for frontend
        return JSONResponse(
            content={
                "success": True,
                "extraction_id": extraction_id,
                "filename": file.filename,
                "policy_type": extracted_data.get("_metadata", {}).get("policy_type", policy_type),
                "extracted_data": extracted_data,
                "fields_extracted": len([k for k in extracted_data.keys() if not k.startswith("_")]),
                "timestamp": datetime.now().isoformat()
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Universal Insurance Policy Extractor",
        "version": "3.0.0",
        "description": "Extract data from any type of insurance policy PDF",
        "supported_policy_types": ["health", "homeowners", "auto", "life", "renters", "disability", "dental"],
        "endpoints": {
            "POST /extract": "Upload and extract data from any policy PDF",
            "POST /extract-with-type": "Upload with policy type hint for better accuracy",
            "GET /health": "Check API health",
            "GET /policy-types": "List supported policy types"
        },
        "frontend_integration": {
            "example_html": "/static/upload.html",
            "cors_enabled": True,
            "response_format": "JSON"
        }
    }

@app.post("/extract-with-type")
async def extract_policy_with_type(
    file: UploadFile = File(...),
    policy_type: str = Form(...)
):
    """Extract policy with explicit type hint for better accuracy"""
    return await extract_policy(file, policy_type)

@app.get("/policy-types")
async def get_policy_types():
    """Get list of supported policy types with their expected fields"""
    return {
        "supported_types": {
            "health": {
                "description": "Health insurance policies",
                "key_fields": ["deductible", "copay", "coinsurance", "out_of_pocket_max", "premium"]
            },
            "homeowners": {
                "description": "Homeowners insurance policies",
                "key_fields": ["dwelling_coverage", "personal_property", "deductible", "property_address"]
            },
            "auto": {
                "description": "Auto insurance policies",
                "key_fields": ["vehicle_info", "liability_coverage", "deductible", "premium"]
            },
            "life": {
                "description": "Life insurance policies",
                "key_fields": ["death_benefit", "beneficiary", "premium", "policy_type"]
            },
            "renters": {
                "description": "Renters insurance policies",
                "key_fields": ["personal_property", "liability", "loss_of_use"]
            },
            "disability": {
                "description": "Disability insurance policies",
                "key_fields": ["monthly_benefit", "elimination_period", "benefit_period"]
            },
            "dental": {
                "description": "Dental insurance policies",
                "key_fields": ["annual_maximum", "preventive_coverage", "basic_coverage"]
            }
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    files_count = len(list(UPLOAD_DIR.glob("*.pdf")))
    return {
        "status": "healthy",
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "files_processed": files_count,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/extractions")
async def list_extractions(limit: int = 50):
    """List recent extractions"""
    extractions = []
    for file_path in sorted(UPLOAD_DIR.glob("*_result.json"), reverse=True)[:limit]:
        with open(file_path, "r") as f:
            data = json.load(f)
            extractions.append({
                "extraction_id": data["extraction_id"],
                "filename": data["filename"],
                "policy_type": data.get("extracted_data", {}).get("_metadata", {}).get("policy_type", "unknown"),
                "fields_extracted": len(data.get("extracted_data", {})),
                "timestamp": data["timestamp"]
            })
    return {
        "count": len(extractions),
        "extractions": extractions
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🏢 UNIVERSAL INSURANCE POLICY EXTRACTOR API v3.0")
    print("=" * 70)
    print(f"✅ Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured (using regex fallback)'}")
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"🌐 CORS Enabled: Yes (for frontend integration)")
    print("\n📋 SUPPORTED POLICY TYPES:")
    print("   • Health Insurance")
    print("   • Homeowners Insurance")
    print("   • Auto Insurance")
    print("   • Life Insurance")
    print("   • Renters Insurance")
    print("   • Disability Insurance")
    print("   • Dental Insurance")
    print("\n💡 TEST WITH CURL:")
    print("   # Auto-detect policy type")
    print("   curl -X POST http://localhost:8000/extract -F 'file=@health_policy.pdf'")
    print("\n   # Specify policy type for better accuracy")
    print("   curl -X POST http://localhost:8000/extract-with-type -F 'file=@health_policy.pdf' -F 'policy_type=health'")
    print("\n🖥️  API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)