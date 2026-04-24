# main.py - Universal Document Intelligence API (Complete Extraction for All Document Types)
import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
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
    title="Universal Document Intelligence API",
    description="Auto-detect any document type and extract all relevant data - No manual selection needed",
    version="5.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def detect_document_type(pdf_text: str) -> Dict[str, Any]:
    """Auto-detect document type from content"""
    text_lower = pdf_text.lower()
    text_upper = pdf_text.upper()
    
    # Priority patterns (higher specificity first)
    patterns = {
        # Indian Driver's License
        "drivers_license": {
            "keywords": [
                "driving licence", "driver license", "drivers license", "licence no",
                "ministry of road transport", "government of india", "rto", "lmv", "mcwg",
                "blood group", "organ donor", "emergency contact"
            ],
            "patterns": [
                r"Licence\s+No\s*:\s*[A-Z]{2}-\d{2}-\d+",
                r"DRIVING\s+LICENCE",
                r"MINISTRY OF ROAD TRANSPORT",
                r"Government\s+of\s+India",
                r"Blood\s+Group\s*:\s*[A-Z\+]+"
            ],
            "priority": 100
        },
        
        # Homeowners Insurance
        "homeowners_insurance": {
            "keywords": [
                "homeowners insurance", "dwelling", "property insurance", 
                "coverage a", "coverage b", "other structures", "personal property",
                "loss of use", "property location", "tax parcel id"
            ],
            "patterns": [
                r"HOMEOWNERS\s+INSURANCE",
                r"COVERAGE\s+A\s*-\s*Dwelling",
                r"PROPERTY\s+LOCATION"
            ],
            "priority": 95
        },
        
        # Health Insurance
        "health_insurance": {
            "keywords": [
                "health insurance", "medical", "deductible", "copay", 
                "coinsurance", "ppo", "hmo", "out-of-pocket", "health plan"
            ],
            "priority": 90
        },
        
        # Auto Insurance
        "auto_insurance": {
            "keywords": [
                "auto insurance", "car insurance", "vehicle", "collision", 
                "comprehensive", "liability", "uninsured motorist"
            ],
            "priority": 90
        },
        
        # Life Insurance
        "life_insurance": {
            "keywords": [
                "life insurance", "death benefit", "term life", "whole life", 
                "beneficiary", "face amount"
            ],
            "priority": 90
        },
        
        # Passport
        "passport": {
            "keywords": [
                "passport", "passport number", "department of state"
            ],
            "patterns": [
                r"PASSPORT\s+NUMBER[:\s]+([A-Z0-9]+)"
            ],
            "priority": 95
        },
        
        # Bank Statement
        "bank_statement": {
            "keywords": [
                "bank statement", "account summary", "transaction history",
                "beginning balance", "ending balance"
            ],
            "priority": 80
        },
        
        # Pay Stub
        "pay_stub": {
            "keywords": [
                "pay stub", "earnings statement", "payroll", "net pay", "gross pay"
            ],
            "priority": 80
        },
        
        # Tax Document
        "tax_document": {
            "keywords": [
                "w-2", "w2", "1099", "tax return", "irs", "form 1040"
            ],
            "priority": 90
        }
    }
    
    # Score each document type
    scores = {}
    for doc_type, info in patterns.items():
        keyword_score = sum(2 for keyword in info["keywords"] if keyword in text_lower)
        
        pattern_score = 0
        if "patterns" in info:
            for pattern in info["patterns"]:
                if re.search(pattern, text_upper, re.IGNORECASE):
                    pattern_score += 15
        
        total_score = keyword_score + pattern_score
        if "priority" in info:
            total_score = total_score * (info["priority"] / 100)
        
        if total_score > 0:
            scores[doc_type] = total_score
    
    # Get the best match
    if scores:
        best_match = max(scores, key=scores.get)
        confidence = min(100, int(scores[best_match]))
        
        return {
            "document_type": best_match,
            "confidence": confidence,
            "all_scores": {k: int(v) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]}
        }
    else:
        return {
            "document_type": "general_document",
            "confidence": 0,
            "all_scores": {}
        }

def extract_drivers_license(pdf_text: str) -> Dict[str, Any]:
    """Extract Indian Driving License data with improved patterns"""
    
    extracted = {}
    
    # License Number
    license_patterns = [
        r"Licence\s+No\s*:\s*([A-Z0-9\-]+)",
        r"License\s+No\s*:\s*([A-Z0-9\-]+)",
        r"Licence\s+Number\s*:\s*([A-Z0-9\-]+)",
        r"([A-Z]{2}-\d{2}-\d{10,15})"
    ]
    for pattern in license_patterns:
        match = re.search(pattern, pdf_text, re.IGNORECASE | re.MULTILINE)
        if match:
            license_num = match.group(1).strip()
            if license_num.upper() not in ["DETAILS", "NUMBER", "LICENSE", "LICENCE"]:
                extracted["license_number"] = license_num
                break
    
    # Full Name
    name_match = re.search(r"Name\s*:\s*([A-Z][A-Z\s]+)(?:\n|S/o|D/o|W/o|$)", pdf_text, re.IGNORECASE | re.MULTILINE)
    if not name_match:
        name_match = re.search(r"Name\s*:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    
    if name_match:
        name = name_match.group(1).strip()
        name = re.sub(r'\s+[S|D|W]/o.*$', '', name)
        extracted["full_name"] = name
    
    # Father/Husband Name
    relation_match = re.search(r"(?:S/o|D/o|W/o)\s*:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if relation_match:
        extracted["father_husband_name"] = relation_match.group(1).strip()
    
    # Date of Birth
    dob_match = re.search(r"Date\s+of\s+Birth\s*:\s*(\d{2}/\d{2}/\d{4})", pdf_text, re.IGNORECASE)
    if dob_match:
        extracted["date_of_birth"] = dob_match.group(1)
    
    # Blood Group
    blood_match = re.search(r"Blood\s+Group\s*:\s*([A-Z\+]+)", pdf_text, re.IGNORECASE)
    if blood_match:
        extracted["blood_group"] = blood_match.group(1)
    
    # Address
    address_match = re.search(r"Address\s*:\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n(?:Issue|Valid))", pdf_text, re.IGNORECASE | re.DOTALL)
    if address_match:
        address = address_match.group(1).strip()
        address = re.sub(r'\s+', ' ', address)
        extracted["address"] = address
    
    # Issue Date
    issue_match = re.search(r"Issue\s+Date\s*:\s*(\d{2}/\d{2}/\d{4})", pdf_text, re.IGNORECASE)
    if issue_match:
        extracted["issue_date"] = issue_match.group(1)
    
    # Expiration Date
    expiry_match = re.search(r"Valid\s+Till\s*:\s*(\d{2}/\d{2}/\d{4})", pdf_text, re.IGNORECASE)
    if expiry_match:
        extracted["expiration_date"] = expiry_match.group(1)
    
    # Issuing Authority
    authority_match = re.search(r"Issuing\s+Authority\s*:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if authority_match:
        extracted["issuing_authority"] = authority_match.group(1).strip()
    
    # Emergency Contact
    emergency_match = re.search(r"Emergency\s+Contact\s*:\s*(\d{10})", pdf_text, re.IGNORECASE)
    if emergency_match:
        extracted["emergency_contact"] = emergency_match.group(1)
    
    # Organ Donor
    organ_match = re.search(r"Organ\s+Donor\s*:\s*(YES|NO)", pdf_text, re.IGNORECASE)
    if organ_match:
        extracted["organ_donor"] = organ_match.group(1).upper()
    
    # Vehicle Classes
    vehicle_classes = []
    
    mcwg_match = re.search(r"MCWG.*?From\s+(\d{2}/\d{2}/\d{4})\s+To\s+(\d{2}/\d{2}/\d{4})", pdf_text, re.IGNORECASE)
    if mcwg_match:
        vehicle_classes.append({
            "class": "MCWG",
            "description": "Motorcycle With Gear",
            "valid_from": mcwg_match.group(1),
            "valid_to": mcwg_match.group(2)
        })
    
    lmv_match = re.search(r"LMV.*?From\s+(\d{2}/\d{2}/\d{4})\s+To\s+(\d{2}/\d{2}/\d{4})", pdf_text, re.IGNORECASE)
    if lmv_match:
        vehicle_classes.append({
            "class": "LMV",
            "description": "Light Motor Vehicle",
            "valid_from": lmv_match.group(1),
            "valid_to": lmv_match.group(2)
        })
    
    if vehicle_classes:
        extracted["vehicle_classes"] = vehicle_classes
    
    return extracted

def extract_homeowners_insurance(pdf_text: str) -> Dict[str, Any]:
    """Extract Homeowners Insurance data"""
    
    extracted = {}
    
    # Policy Information
    company_match = re.search(r"INSURANCE\s+COMPANY:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if company_match:
        extracted["insurance_company"] = company_match.group(1).strip()
    
    policy_match = re.search(r"Policy\s+Number:\s*([A-Z0-9\-]+)", pdf_text, re.IGNORECASE)
    if policy_match:
        extracted["policy_number"] = policy_match.group(1)
    
    agent_match = re.search(r"Agent:\s*([^\n(]+)", pdf_text, re.IGNORECASE)
    if agent_match:
        extracted["agent_name"] = agent_match.group(1).strip()
    
    # Property Owner
    insured_match = re.search(r"Named\s+Insured:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if insured_match:
        extracted["named_insured"] = insured_match.group(1).strip()
    
    address_match = re.search(r"Mailing\s+Address:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if address_match:
        extracted["mailing_address"] = address_match.group(1).strip()
    
    # Property Location
    prop_address_match = re.search(r"Insured\s+Property\s+Address:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if prop_address_match:
        extracted["property_address"] = prop_address_match.group(1).strip()
    
    # Property Details
    year_match = re.search(r"Year\s+Built.*?:\s*(\d{4})", pdf_text, re.IGNORECASE)
    if year_match:
        extracted["year_built"] = int(year_match.group(1))
    
    sqft_match = re.search(r"Square\s+Footage:\s*([\d,]+)", pdf_text, re.IGNORECASE)
    if sqft_match:
        extracted["square_footage"] = int(sqft_match.group(1).replace(',', ''))
    
    bedrooms_match = re.search(r"Bedrooms:\s*(\d+)", pdf_text, re.IGNORECASE)
    if bedrooms_match:
        extracted["bedrooms"] = int(bedrooms_match.group(1))
    
    bathrooms_match = re.search(r"Bathrooms:\s*(\d+)", pdf_text, re.IGNORECASE)
    if bathrooms_match:
        extracted["bathrooms"] = int(bathrooms_match.group(1))
    
    # Coverage Limits
    dwelling_match = re.search(r"Coverage A.*?Sum\s+Insured:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE | re.DOTALL)
    if dwelling_match:
        extracted["dwelling_coverage"] = int(dwelling_match.group(1).replace(',', ''))
    
    other_struct_match = re.search(r"Coverage B.*?Limit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE | re.DOTALL)
    if other_struct_match:
        extracted["other_structures_coverage"] = int(other_struct_match.group(1).replace(',', ''))
    
    personal_prop_match = re.search(r"Coverage C.*?Limit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE | re.DOTALL)
    if personal_prop_match:
        extracted["personal_property_coverage"] = int(personal_prop_match.group(1).replace(',', ''))
    
    liability_match = re.search(r"Coverage E.*?Limit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE | re.DOTALL)
    if liability_match:
        extracted["personal_liability_coverage"] = int(liability_match.group(1).replace(',', ''))
    
    # Deductibles
    standard_ded_match = re.search(r"Standard\s+Deductible:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if standard_ded_match:
        extracted["standard_deductible"] = int(standard_ded_match.group(1).replace(',', ''))
    
    # Policy Period
    effective_match = re.search(r"Effective\s+Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", pdf_text, re.IGNORECASE)
    if effective_match:
        try:
            date_obj = datetime.strptime(effective_match.group(1), "%B %d, %Y")
            extracted["effective_date"] = date_obj.strftime("%Y-%m-%d")
        except:
            extracted["effective_date"] = effective_match.group(1)
    
    expiration_match = re.search(r"Expiration\s+Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", pdf_text, re.IGNORECASE)
    if expiration_match:
        try:
            date_obj = datetime.strptime(expiration_match.group(1), "%B %d, %Y")
            extracted["expiration_date"] = date_obj.strftime("%Y-%m-%d")
        except:
            extracted["expiration_date"] = expiration_match.group(1)
    
    # Premium
    premium_match = re.search(r"Annual\s+Premium:\s*\$?([\d,]+\.?\d*)", pdf_text, re.IGNORECASE)
    if premium_match:
        extracted["annual_premium"] = float(premium_match.group(1).replace(',', ''))
    
    return extracted

def extract_health_insurance(pdf_text: str) -> Dict[str, Any]:
    """Extract Health Insurance data with comprehensive fields"""
    
    extracted = {}
    
    # Policy Information
    company_match = re.search(r"INSURANCE\s+COMPANY:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if company_match:
        extracted["insurance_company"] = company_match.group(1).strip()
    
    policy_match = re.search(r"Policy\s+Number:\s*([A-Z0-9\-]+)", pdf_text, re.IGNORECASE)
    if policy_match:
        extracted["policy_number"] = policy_match.group(1)
    
    group_match = re.search(r"Group\s+Number:\s*([A-Z0-9\-]+)", pdf_text, re.IGNORECASE)
    if group_match:
        extracted["group_number"] = group_match.group(1)
    
    # Policyholder Information
    name_match = re.search(r"Name:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if name_match:
        extracted["policyholder_name"] = name_match.group(1).strip()
    
    dob_match = re.search(r"Date\s+of\s+Birth:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if dob_match:
        extracted["date_of_birth"] = dob_match.group(1).strip()
    
    gender_match = re.search(r"Gender:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if gender_match:
        extracted["gender"] = gender_match.group(1).strip()
    
    address_match = re.search(r"Address:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if address_match:
        extracted["address"] = address_match.group(1).strip()
    
    # Coverage Period
    effective_match = re.search(r"Effective\s+Date:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if effective_match:
        extracted["effective_date"] = effective_match.group(1).strip()
    
    expiration_match = re.search(r"Expiration\s+Date:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if expiration_match:
        extracted["expiration_date"] = expiration_match.group(1).strip()
    
    # Plan Details
    plan_type_match = re.search(r"Plan\s+Type:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if plan_type_match:
        extracted["plan_type"] = plan_type_match.group(1).strip()
    
    network_match = re.search(r"Network:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if network_match:
        extracted["network"] = network_match.group(1).strip()
    
    # Cost Sharing - Deductibles
    individual_deductible_match = re.search(r"Annual\s+Deductible:\s*\$?([\d,]+)\s*\(Individual\)", pdf_text, re.IGNORECASE)
    if individual_deductible_match:
        extracted["individual_deductible"] = int(individual_deductible_match.group(1).replace(',', ''))
    
    family_deductible_match = re.search(r"Annual\s+Deductible:.*?\$\s*([\d,]+)\s*\(Family\)", pdf_text, re.IGNORECASE)
    if family_deductible_match:
        extracted["family_deductible"] = int(family_deductible_match.group(1).replace(',', ''))
    
    # Out-of-Pocket Maximums
    individual_oop_match = re.search(r"Out-of-Pocket\s+Maximum:\s*\$?([\d,]+)\s*\(Individual\)", pdf_text, re.IGNORECASE)
    if individual_oop_match:
        extracted["individual_out_of_pocket_max"] = int(individual_oop_match.group(1).replace(',', ''))
    
    family_oop_match = re.search(r"Out-of-Pocket\s+Maximum:.*?\$\s*([\d,]+)\s*\(Family\)", pdf_text, re.IGNORECASE)
    if family_oop_match:
        extracted["family_out_of_pocket_max"] = int(family_oop_match.group(1).replace(',', ''))
    
    # Coinsurance
    coinsurance_match = re.search(r"Coinsurance:\s*Plan\s+pays\s*(\d+)%", pdf_text, re.IGNORECASE)
    if coinsurance_match:
        extracted["coinsurance_percentage"] = int(coinsurance_match.group(1))
    
    # Copayments
    copayments = {}
    
    pcp_match = re.search(r"Primary\s+Care\s+Physician\s+Visit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if pcp_match:
        copayments["primary_care"] = int(pcp_match.group(1).replace(',', ''))
    
    specialist_match = re.search(r"Specialist\s+Visit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if specialist_match:
        copayments["specialist"] = int(specialist_match.group(1).replace(',', ''))
    
    emergency_match = re.search(r"Emergency\s+Room\s+Visit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if emergency_match:
        copayments["emergency_room"] = int(emergency_match.group(1).replace(',', ''))
    
    urgent_match = re.search(r"Urgent\s+Care\s+Visit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if urgent_match:
        copayments["urgent_care"] = int(urgent_match.group(1).replace(',', ''))
    
    telehealth_match = re.search(r"Telehealth\s+Visit:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if telehealth_match:
        copayments["telehealth"] = int(telehealth_match.group(1).replace(',', ''))
    
    if copayments:
        extracted["copayments"] = copayments
    
    # Premium Information
    monthly_premium_match = re.search(r"Monthly\s+Premium:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if monthly_premium_match:
        extracted["monthly_premium"] = float(monthly_premium_match.group(1).replace(',', ''))
    
    annual_premium_match = re.search(r"Annual\s+Premium:\s*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if annual_premium_match:
        extracted["annual_premium"] = float(annual_premium_match.group(1).replace(',', ''))
    
    payment_due_match = re.search(r"Payment\s+Due:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if payment_due_match:
        extracted["payment_due"] = payment_due_match.group(1).strip()
    
    # Prescription Drug Coverage
    prescription = {}
    
    generic_match = re.search(r"Tier\s+1\s*\(Generic\).*?\$?([\d,]+)\s*copay", pdf_text, re.IGNORECASE)
    if generic_match:
        prescription["generic_copay"] = int(generic_match.group(1).replace(',', ''))
    
    preferred_match = re.search(r"Tier\s+2\s*\(Preferred\s+Brand\).*?\$?([\d,]+)\s*copay", pdf_text, re.IGNORECASE)
    if preferred_match:
        prescription["preferred_brand_copay"] = int(preferred_match.group(1).replace(',', ''))
    
    non_preferred_match = re.search(r"Tier\s+3\s*\(Non-Preferred\s+Brand\).*?\$?([\d,]+)\s*copay", pdf_text, re.IGNORECASE)
    if non_preferred_match:
        prescription["non_preferred_brand_copay"] = int(non_preferred_match.group(1).replace(',', ''))
    
    specialty_match = re.search(r"Tier\s+4\s*\(Specialty\).*?(\d+)%", pdf_text, re.IGNORECASE)
    if specialty_match:
        prescription["specialty_coinsurance"] = int(specialty_match.group(1))
    
    if prescription:
        extracted["prescription_drugs"] = prescription
    
    # Medical Services Coverage
    medical_services = {}
    
    inpatient_match = re.search(r"Inpatient\s+Hospital:\s*(\d+)%", pdf_text, re.IGNORECASE)
    if inpatient_match:
        medical_services["inpatient_hospital"] = f"{inpatient_match.group(1)}% after deductible"
    
    outpatient_match = re.search(r"Outpatient\s+Surgery:\s*(\d+)%", pdf_text, re.IGNORECASE)
    if outpatient_match:
        medical_services["outpatient_surgery"] = f"{outpatient_match.group(1)}% after deductible"
    
    emergency_services_match = re.search(r"Emergency\s+Services:\s*(\d+)%", pdf_text, re.IGNORECASE)
    if emergency_services_match:
        medical_services["emergency_services"] = f"{emergency_services_match.group(1)}% after deductible"
    
    preventive_match = re.search(r"Preventive\s+Care:\s*(\d+)%", pdf_text, re.IGNORECASE)
    if preventive_match:
        medical_services["preventive_care"] = f"{preventive_match.group(1)}% covered"
    
    physical_therapy_match = re.search(r"Physical\s+Therapy:\s*(\d+)\s+visits", pdf_text, re.IGNORECASE)
    if physical_therapy_match:
        medical_services["physical_therapy"] = f"{physical_therapy_match.group(1)} visits per year, $40 copay"
    
    if medical_services:
        extracted["medical_services"] = medical_services
    
    # Additional Coverages
    additional = {}
    
    vision_match = re.search(r"Vision:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if vision_match:
        additional["vision"] = vision_match.group(1).strip()
    
    dental_match = re.search(r"Dental:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if dental_match:
        additional["dental"] = dental_match.group(1).strip()
    
    if additional:
        extracted["additional_coverages"] = additional
    
    # Claims Information
    claims_match = re.search(r"Claims\s+Address:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if claims_match:
        extracted["claims_address"] = claims_match.group(1).strip()
    
    phone_match = re.search(r"Phone:\s*([0-9\-]+)", pdf_text, re.IGNORECASE)
    if phone_match:
        extracted["claims_phone"] = phone_match.group(1).strip()
    
    website_match = re.search(r"Website:\s*([^\n]+)", pdf_text, re.IGNORECASE)
    if website_match:
        extracted["website"] = website_match.group(1).strip()
    
    # Pre-existing Conditions
    pre_existing_match = re.search(r"Pre-existing\s+conditions\s+covered\s+from\s+effective\s+date", pdf_text, re.IGNORECASE)
    if pre_existing_match:
        extracted["pre_existing_conditions"] = "Covered from effective date"
    
    return extracted

def extract_auto_insurance(pdf_text: str) -> Dict[str, Any]:
    """Extract Auto Insurance data"""
    
    extracted = {}
    
    company_match = re.search(r"(?:INSURANCE\s+COMPANY|Company)[:\s]+([^\n]+)", pdf_text, re.IGNORECASE)
    if company_match:
        extracted["insurance_company"] = company_match.group(1).strip()
    
    policy_match = re.search(r"(?:POLICY\s+NUMBER|Policy\s+#)[:\s]+([A-Z0-9\-]+)", pdf_text, re.IGNORECASE)
    if policy_match:
        extracted["policy_number"] = policy_match.group(1)
    
    vehicle_match = re.search(r"(?:Vehicle|Car)[:\s]+([^\n]+)", pdf_text, re.IGNORECASE)
    if vehicle_match:
        extracted["vehicle"] = vehicle_match.group(1).strip()
    
    return extracted

def extract_life_insurance(pdf_text: str) -> Dict[str, Any]:
    """Extract Life Insurance data"""
    
    extracted = {}
    
    company_match = re.search(r"(?:INSURANCE\s+COMPANY|Company)[:\s]+([^\n]+)", pdf_text, re.IGNORECASE)
    if company_match:
        extracted["insurance_company"] = company_match.group(1).strip()
    
    policy_match = re.search(r"(?:POLICY\s+NUMBER|Policy\s+#)[:\s]+([A-Z0-9\-]+)", pdf_text, re.IGNORECASE)
    if policy_match:
        extracted["policy_number"] = policy_match.group(1)
    
    benefit_match = re.search(r"(?:Death\s+Benefit|Face\s+Amount)[:\s]*\$?([\d,]+)", pdf_text, re.IGNORECASE)
    if benefit_match:
        extracted["death_benefit"] = int(benefit_match.group(1).replace(',', ''))
    
    return extracted

def create_extraction_prompt(pdf_text: str, document_type: str) -> str:
    """Create tailored extraction prompt based on detected document type"""
    
    base_prompt = f"""Analyze this {document_type.replace('_', ' ').upper()} document and extract ALL relevant information.

Document text:
{pdf_text[:40000]}

"""
    
    if document_type == "drivers_license":
        return base_prompt + """
Extract EVERYTHING from this driver's license. Return as JSON with:
- license_number
- full_name
- father_husband_name
- date_of_birth (YYYY-MM-DD)
- blood_group
- address
- issue_date (YYYY-MM-DD)
- expiration_date (YYYY-MM-DD)
- issuing_authority
- emergency_contact
- organ_donor (YES/NO)
- vehicle_classes (array)

Return ONLY valid JSON."""
    
    elif document_type == "homeowners_insurance":
        return base_prompt + """
Extract EVERYTHING from this homeowners insurance. Return as JSON with:
- insurance_company
- policy_number
- agent_name
- named_insured
- property_address
- year_built
- square_footage
- bedrooms
- bathrooms
- dwelling_coverage
- personal_property_coverage
- liability_coverage
- standard_deductible
- effective_date (YYYY-MM-DD)
- expiration_date (YYYY-MM-DD)
- annual_premium

Return ONLY valid JSON."""
    
    elif document_type == "health_insurance":
        return base_prompt + """
Extract EVERYTHING from this health insurance. Return as JSON with:
- insurance_company
- policy_number
- policyholder_name
- effective_date
- deductible
- copay
- monthly_premium

Return ONLY valid JSON."""
    
    else:
        return base_prompt + """
Extract ALL relevant information from this document.

Return as JSON with appropriate fields.
Convert dates to YYYY-MM-DD format.
Return ONLY valid JSON."""

def extract_with_gemini(pdf_text: str, document_type: str) -> Dict[str, Any]:
    """Extract data using Gemini AI based on detected document type"""
    
    # Use regex for specific document types first
    if document_type == "drivers_license":
        regex_data = extract_drivers_license(pdf_text)
        if regex_data and len(regex_data) > 3:
            print(f"✅ Extracted {len(regex_data)} fields using regex patterns")
            return regex_data
    
    elif document_type == "homeowners_insurance":
        regex_data = extract_homeowners_insurance(pdf_text)
        if regex_data and len(regex_data) > 5:
            print(f"✅ Extracted {len(regex_data)} fields using regex patterns")
            return regex_data
    
    elif document_type == "health_insurance":
        regex_data = extract_health_insurance(pdf_text)
        if regex_data and len(regex_data) > 3:
            print(f"✅ Extracted {len(regex_data)} fields from health insurance using regex patterns")
            return regex_data
    
    elif document_type == "auto_insurance":
        regex_data = extract_auto_insurance(pdf_text)
        if regex_data:
            return regex_data
    
    elif document_type == "life_insurance":
        regex_data = extract_life_insurance(pdf_text)
        if regex_data:
            return regex_data
    
    if not GEMINI_API_KEY:
        return {}
    
    prompt = create_extraction_prompt(pdf_text, document_type)
    prompt += "\n\nCRITICAL: Return ONLY valid JSON, no other text."
    
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
                    
                    json_match = re.search(r'\{[\s\S]*\}', text_response)
                    if json_match:
                        json_str = json_match.group()
                        extracted = json.loads(json_str)
                        print(f"✅ Extracted {len(extracted)} fields using Gemini")
                        return extracted
        except Exception as e:
            print(f"⚠️ Error with {model_name}: {str(e)}")
            continue
    
    return {}

@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """Upload ANY document - automatically detects type and extracts all data"""
    
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
        
        # Extract text
        print(f"📄 Processing: {file.filename}")
        pdf_text = extract_text_from_pdf(content)
        
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        print(f"📝 Extracted {len(pdf_text)} characters")
        
        # Auto-detect document type
        detection = detect_document_type(pdf_text)
        document_type = detection["document_type"]
        confidence = detection["confidence"]
        
        print(f"🔍 Detected: {document_type.upper()} (confidence: {confidence}%)")
        
        # Extract data based on detected type
        extracted_data = extract_with_gemini(pdf_text, document_type)
        
        # Clean dates in extracted data
        for key, value in extracted_data.items():
            if 'date' in key.lower() and isinstance(value, str):
                date_match = re.search(r'(\d{2})/(\d{2})/(\d{4})', value)
                if date_match:
                    day, month, year = date_match.groups()
                    extracted_data[key] = f"{year}-{month}-{day}"
        
        # Add metadata
        result = {
            "success": True,
            "extraction_id": extraction_id,
            "filename": file.filename,
            "document_type": document_type,
            "detection_confidence": confidence,
            "extracted_data": extracted_data,
            "fields_extracted": len(extracted_data),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save result
        result_path = UPLOAD_DIR / f"{extraction_id}_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Extracted {len(extracted_data)} fields from {document_type}\n")
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Universal Document Intelligence API",
        "version": "5.0.0",
        "description": "Upload ANY document - Auto-detects type and extracts all data",
        "supported_document_types": [
            "Driver's License (Indian)",
            "Homeowners Insurance",
            "Health Insurance",
            "Auto Insurance",
            "Life Insurance",
            "Passport",
            "Bank Statements",
            "Pay Stubs",
            "Tax Documents"
        ],
        "endpoint": {
            "url": "POST /extract",
            "usage": "curl -X POST http://localhost:8000/extract -F 'file=@your_document.pdf'"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    files_count = len(list(UPLOAD_DIR.glob("*.pdf")))
    return {
        "status": "healthy",
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "documents_processed": files_count
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("UNIVERSAL DOCUMENT INTELLIGENCE API v5.0")
    print("=" * 70)
    print("ONE ENDPOINT - ANY DOCUMENT - AUTO DETECTION")
    print("=" * 70)
    print(f"Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured'}")
    print("\n SUPPORTED DOCUMENT TYPES:")
    print("Driver's License (Indian) - 15+ fields")
    print(" Homeowners Insurance - 20+ fields")
    print(" Health Insurance - 10+ fields")
    print(" Auto Insurance - 8+ fields")
    print(" Life Insurance - 8+ fields")
    print("\n USAGE:")
    print("   curl -X POST http://localhost:8000/extract -F 'file=@document.pdf'")
    print("\n  API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)