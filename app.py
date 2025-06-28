from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_file
import fitz  # PyMuPDF for PDF handling
import requests
import tiktoken
import random
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
import re
from datetime import datetime, timedelta, date, timezone
import stripe
from flask_session import Session
import difflib
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
from openai import OpenAI  # Added for OpenAI API
from typing import Dict, Any, List
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(16).hex()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = app.secret_key
Session(app)

# Stripe configuration (replace with your actual keys)
stripe.api_key = 'sk_test_your_stripe_secret_key'
STRIPE_WEBHOOK_SECRET = 'whsec_your_webhook_secret'

# OCR.Space API configuration (replace with your actual key)
OCR_SPACE_API_KEY = "YOUR_OCR_SPACE_API_KEY"
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory for uploaded files
UPLOAD_FOLDER = "Uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# JSON file paths
USERS_FILE = "users.json"
DOCUMENTS_FILE = "documents.json"
CONTACT_MESSAGES_FILE = "contact_messages.json"
PROPERTIES_FILE = "properties.json"
PRODUCTS_FILE = "products.json"
EVENTS_FILE = "events.json"
REPORTS_FILE = "reports.json"
LEGISLATIVE_UPDATES_FILE = "legislative_updates.json"

# Initialize JSON files
for file in [USERS_FILE, DOCUMENTS_FILE, CONTACT_MESSAGES_FILE, PROPERTIES_FILE,
             PRODUCTS_FILE, EVENTS_FILE, REPORTS_FILE, LEGISLATIVE_UPDATES_FILE]:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump([], f)

# Initialize default data for JSON files
def initialize_json_files():
    default_products = [
        {"name": "Leasehold Analysis", "icon": "ri-file-chart-line", "color": "blue", "lastUpdated": "June 8, 2025", "status": "Active"},
        {"name": "Compare Leaseholds", "icon": "ri-scales-line", "color": "indigo", "lastUpdated": "June 5, 2025", "status": "Active"},
        {"name": "Service Charge Analysis", "icon": "ri-money-dollar-circle-line", "color": "orange", "lastUpdated": "June 10, 2025", "status": "Active"},
        {"name": "Premium Consultation", "icon": "ri-user-voice-line", "color": "gray", "lastUpdated": "Expert advice on demand", "status": "Locked"}
    ]
    if os.path.getsize(PRODUCTS_FILE) == 0:
        with open(PRODUCTS_FILE, 'w') as f:
            json.dump(default_products, f, indent=4)

    default_events = [
        {"date": "June 15, 2025", "title": "Scheduled Maintenance", "description": "Annual building inspection at Riverside Apartments", "color": "blue"},
        {"date": "June 20, 2025", "title": "Legislative Update", "description": "New regulations on service charge transparency", "color": "purple"},
        {"date": "June 28, 2025", "title": "Report Renewal", "description": "Your leasehold analysis report will be updated", "color": "yellow"}
    ]
    if os.path.getsize(EVENTS_FILE) == 0:
        with open(EVENTS_FILE, 'w') as f:
            json.dump(default_events, f, indent=4)

    default_reports = [
        {"user_email": None, "icon": "ri-file-chart-line", "color": "blue", "title": "Riverside Apartments Analysis", "date": "June 8, 2025"},
        {"user_email": None, "icon": "ri-scales-line", "color": "indigo", "title": "Riverside vs. Oakwood Comparison", "date": "June 5, 2025"},
        {"user_email": None, "icon": "ri-money-dollar-circle-line", "color": "orange", "title": "Q2 Service Charge Analysis", "date": "June 10, 2025"}
    ]
    if os.path.getsize(REPORTS_FILE) == 0:
        with open(REPORTS_FILE, 'w') as f:
            json.dump(default_reports, f, indent=4)

    default_updates = [
        {
            "icon": "ri-government-line",
            "color": "red",
            "title": "Leasehold Reform Act 2025",
            "description": "New legislation affecting ground rent calculations and lease extensions.",
            "date": "June 1, 2025"
        },
        {
            "icon": "ri-government-line",
            "color": "green",
            "title": "Service Charge Transparency Rules",
            "description": "New requirements for landlords to provide itemized breakdowns.",
            "date": "May 25, 2025"
        }
    ]
    if os.path.getsize(LEGISLATIVE_UPDATES_FILE) == 0:
        with open(LEGISLATIVE_UPDATES_FILE, 'w') as f:
            json.dump(default_updates, f, indent=4)

initialize_json_files()
def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate the number of tokens in the input text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error estimating tokens: {e}")
        return len(text) 
def clean_json_response(response_text: str) -> str:
    """Clean up OpenAI response to extract valid JSON."""
    try:
        # Remove leading/trailing whitespace and newlines
        response_text = response_text.strip()
        # Remove markdown code blocks (e.g., ```json ... ```)
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        # Remove any non-JSON lines before or after the JSON object
        response_text = re.sub(r'^[^\{]*', '', response_text)
        response_text = re.sub(r'[^\}]*$', '', response_text)
        # Ensure it starts with { and ends with }
        if not response_text:
            return '{}'
        if not response_text.startswith('{'):
            response_text = '{' + response_text[response_text.find('"'):] if '"' in response_text else '{}'
        if not response_text.endswith('}'):
            response_text = response_text[:response_text.rfind('}') + 1] if '}' in response_text else '{}'
        # Validate JSON before returning
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            logger.error(f"Cleaned response is still invalid JSON: {response_text[:200]}...")
            return '{}'
    except Exception as e:
        logger.error(f"Error cleaning JSON response: {e}")
        return '{}'

# Simulated LLM analysis for lease documents

def analyze_lease_document(text: str, max_retries: int = 3, base_retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Analyzes lease documents using OpenAI API to extract and categorize clauses,
    lease information, risk assessments, and recommendations.

    Args:
        text (str): Extracted text from the uploaded lease document.
        max_retries (int): Maximum number of retries for API calls.
        base_retry_delay (float): Base delay between retries in seconds (exponential backoff).

    Returns:
        Dict[str, Any]: Detailed analysis including clauses, lease information,
                        risk assessment, recommendations, and error details if applicable.
    """
    if not text or not text.strip():
        logger.error("No valid text provided for lease analysis")
        return {
            "document_summary": {
                "clauses": [],
                "priority_action_items": ["Upload a valid lease document with readable text."]
            },
            "lease_information": {
                "lease_term": "Unknown",
                "parties_involved": "Unknown",
                "ground_rent": "Unknown",
                "property_address": "Unknown",
                "commencement_date": "Unknown"
            },
            "risk_assessment": {
                "risk_percentage": 0,
                "overall_risk": "Unknown",
                "description": "No valid document text provided for analysis.",
                "risk_items": [
                    {"title": "Invalid Input", "description": "No readable text provided for analysis.", "severity": "high", "reference": "N/A"}
                ]
            },
            "analysis": {
                "positive_elements": [],
                "concerning_clauses": [
                    {"title": "Input Error", "description": "No valid text provided for analysis.", "reference": "N/A", "clause_text": ""}
                ],
                "recommended_removals": [],
                "renegotiation_points": [],
                "missing_elements": [],
                "modern_standards": []
            },
            "error": "No valid text provided for lease analysis"
        }

    # Truncate input text to avoid exceeding context window (approx 12,000 tokens for gpt-4o)
    max_input_tokens = 12000
    input_tokens = estimate_tokens(text)
    if input_tokens > max_input_tokens:
        logger.warning(f"Input text exceeds {max_input_tokens} tokens ({input_tokens} tokens). Truncating.")
        encoding = tiktoken.encoding_for_model("gpt-4o")
        text = encoding.decode(encoding.encode(text)[:max_input_tokens])

    # Define fallback response with specific error details
    fallback_response = {
        "document_summary": {
            "clauses": [],
            "priority_action_items": ["Unable to analyze the lease document due to processing error."]
        },
        "lease_information": {
            "lease_term": "Unknown",
            "parties_involved": "Unknown",
            "ground_rent": "Unknown",
            "property_address": "Unknown",
            "commencement_date": "Unknown"
        },
        "risk_assessment": {
            "risk_percentage": 0,
            "overall_risk": "Unknown",
            "description": "Analysis failed due to an internal processing error.",
            "risk_items": [
                {"title": "Processing Error", "description": "Failed to process the document.", "severity": "high", "reference": "N/A"}
            ]
        },
        "analysis": {
            "positive_elements": [],
            "concerning_clauses": [
                {"title": "Processing Error", "description": "Unable to process the document.", "reference": "N/A", "clause_text": ""}
            ],
            "recommended_removals": [],
            "renegotiation_points": [],
            "missing_elements": [],
            "modern_standards": []
        },
        "error": "Internal processing error"
    }

    try:
        # Refined prompt to enforce strict JSON output
        prompt = """
You are an expert in leasehold property management and lease document analysis. Given the text of a lease document, analyze it to extract key information, identify clauses, assess risks, and provide recommendations. Follow these instructions precisely:

1. **Document Summary**:
   - Extract 5-10 distinct clauses or sections from the document text.
   - Categorize each clause as Positive, Concerning, Remove, or Renegotiate.
   - Assign a reference (e.g., "Clause 4.2" or "Section 3") from the document or infer if not explicit.
   - Provide 2-3 priority action items based on the analysis (e.g., "Consult a lawyer about ground rent").

2. **Lease Information**:
   - Extract key details: lease term (e.g., "99 years"), parties involved (lessee and lessor), ground rent (amount and frequency), property address, and commencement date.
   - If any details are missing, mark as "Unknown".

3. **Risk Assessment**:
   - Calculate a risk percentage (0-100) based on concerning clauses (e.g., escalating ground rent = +20%, ambiguous terms = +10%).
   - Assign an overall risk level: Low (<50), Medium (50-75), High (>75).
   - Provide a description summarizing the risk profile.
   - List 2-5 risk items with title, description, severity (low/medium/high), and reference.

4. **Analysis**:
   - **Positive Elements**: Identify clauses that benefit the lessee (e.g., fixed ground rent).
   - **Concerning Clauses**: Highlight clauses that pose risks (e.g., escalating ground rent).
   - **Recommended Removals**: Suggest clauses to remove (e.g., outdated provisions).
   - **Renegotiation Points**: Suggest clauses to renegotiate (e.g., unclear service charges).
   - **Missing Elements**: Identify missing modern provisions (e.g., right to manage).
   - **Modern Standards**: Suggest clauses to align with current standards (e.g., ground rent cap at 0.1% of property value).
   - For each item, provide a title, description, reference, and (for concerning clauses) the clause text.

**Output Requirements**:
- Return ONLY a valid JSON object with the following structure:
{
  "document_summary": {
    "clauses": [{"text": str, "category": str, "reference": str}, ...],
    "priority_action_items": [str, ...]
  },
  "lease_information": {
    "lease_term": str,
    "parties_involved": str,
    "ground_rent": str,
    "property_address": str,
    "commencement_date": str
  },
  "risk_assessment": {
    "risk_percentage": int,
    "overall_risk": str,
    "description": str,
    "risk_items": [{"title": str, "description": str, "severity": str, "reference": str}, ...]
  },
  "analysis": {
    "positive_elements": [{"title": str, "description": str, "reference": str}, ...],
    "concerning_clauses": [{"title": str, "description": str, "reference": str, "clause_text": str}, ...],
    "recommended_removals": [{"title": str, "description": str, "reference": str}, ...],
    "renegotiation_points": [{"title": str, "description": str, "reference": str}, ...],
    "missing_elements": [{"title": str, "description": str, "reference": str}, ...],
    "modern_standards": [{"title": str, "description": str, "reference": str}, ...]
  }
}
- Do NOT include any text outside the JSON object (e.g., no newlines, comments, markdown, or explanations).
- Ensure the JSON is syntactically correct with no trailing commas or unclosed brackets.
- If the input text is empty or invalid, include an error description in `risk_assessment.risk_items` and `analysis.concerning_clauses`.
- Base the analysis on the provided document text. Do not fabricate data unless explicitly instructed to mark missing values as "Unknown".

Document text:
{text}
"""

        # Estimate total tokens needed
        prompt_tokens = estimate_tokens(prompt.format(text=text))
        max_output_tokens = 8000
        total_tokens = prompt_tokens + max_output_tokens

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                # Call OpenAI API
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a precise and analytical assistant for leasehold property management. Return only a valid JSON object, with no additional text, newlines, or markdown."},
                        {"role": "user", "content": prompt.format(text=text)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Lowered temperature for stricter adherence to instructions
                    max_tokens=max_output_tokens
                )

                # Parse API response
                response_content = response.choices[0].message.content
                logger.debug(f"Raw OpenAI response (attempt {attempt + 1}): {response_content[:2000]}...")  # Log up to 2000 chars
                cleaned_response = clean_json_response(response_content)
                if cleaned_response == '{}':
                    logger.warning(f"Cleaned response is empty JSON (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        retry_delay = base_retry_delay * (2 ** attempt)
                        logger.info(f"Retrying API call after {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        **fallback_response,
                        "risk_assessment": {
                            **fallback_response["risk_assessment"],
                            "description": "API returned an empty or invalid JSON response.",
                            "risk_items": [
                                {"title": "Empty JSON Response", "description": f"API returned invalid or empty JSON: {response_content[:100]}...", "severity": "high", "reference": "N/A"}
                            ]
                        },
                        "analysis": {
                            **fallback_response["analysis"],
                            "concerning_clauses": [
                                {"title": "Empty JSON Response", "description": f"API returned invalid or empty JSON: {response_content[:100]}...", "reference": "N/A", "clause_text": ""}
                            ]
                        },
                        "error": f"Empty or invalid JSON response: {response_content[:100]}..."
                    }

                try:
                    analysis_result = json.loads(cleaned_response)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Invalid JSON in cleaned OpenAI response (attempt {attempt + 1}/{max_retries}): {cleaned_response[:500]}...")
                    if attempt < max_retries - 1:
                        retry_delay = base_retry_delay * (2 ** attempt)
                        logger.info(f"Retrying API call after {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        **fallback_response,
                        "risk_assessment": {
                            **fallback_response["risk_assessment"],
                            "description": f"Failed to parse API response: {str(json_err)}.",
                            "risk_items": [
                                {"title": "JSON Parsing Error", "description": f"Invalid JSON in API response: {cleaned_response[:100]}...", "severity": "high", "reference": "N/A"}
                            ]
                        },
                        "analysis": {
                            **fallback_response["analysis"],
                            "concerning_clauses": [
                                {"title": "JSON Parsing Error", "description": f"API returned invalid JSON: {cleaned_response[:100]}...", "reference": "N/A", "clause_text": ""}
                            ]
                        },
                        "error": f"JSON parsing error: {str(json_err)}"
                    }

                # Validate response structure
                required_keys = ["document_summary", "lease_information", "risk_assessment", "analysis"]
                missing_keys = [key for key in required_keys if key not in analysis_result]
                if missing_keys:
                    logger.error(f"Incomplete OpenAI response (attempt {attempt + 1}/{max_retries}): missing keys {missing_keys}")
                    if attempt < max_retries - 1:
                        retry_delay = base_retry_delay * (2 ** attempt)
                        logger.info(f"Retrying API call after {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        **fallback_response,
                        "risk_assessment": {
                            **fallback_response["risk_assessment"],
                            "description": f"API response missing required keys: {', '.join(missing_keys)}.",
                            "risk_items": [
                                {"title": "Incomplete Response", "description": f"Missing required keys: {', '.join(missing_keys)}.", "severity": "high", "reference": "N/A"}
                            ]
                        },
                        "analysis": {
                            **fallback_response["analysis"],
                            "concerning_clauses": [
                                {"title": "Incomplete Response", "description": f"Missing required keys: {', '.join(missing_keys)}.", "reference": "N/A", "clause_text": ""}
                            ]
                        },
                        "error": f"Incomplete response: missing keys {', '.join(missing_keys)}"
                    }

                # Validate extracted data
                if not analysis_result["document_summary"]["clauses"]:
                    logger.warning("No clauses extracted from lease document")
                    analysis_result["document_summary"]["priority_action_items"].append("No lease clauses identified. Verify document content.")
                    analysis_result["risk_assessment"]["risk_items"].append({
                        "title": "No Clauses Found",
                        "description": "No lease clauses were identified in the document. Please verify the document content.",
                        "severity": "medium",
                        "reference": "N/A"
                    })
                    analysis_result["analysis"]["concerning_clauses"].append({
                        "title": "No Clauses Extracted",
                        "description": "The document may not contain relevant lease information.",
                        "reference": "N/A",
                        "clause_text": ""
                    })

                return analysis_result

            except Exception as api_err:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {api_err}")
                if attempt < max_retries - 1:
                    retry_delay = base_retry_delay * (2 ** attempt)
                    logger.info(f"Retrying API call after {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return {
                    **fallback_response,
                    "risk_assessment": {
                        **fallback_response["risk_assessment"],
                        "description": f"Failed to process document due to API error: {str(api_err)}.",
                        "risk_items": [
                            {"title": "API Error", "description": f"OpenAI API failed: {str(api_err)}.", "severity": "high", "reference": "N/A"}
                        ]
                    },
                    "analysis": {
                        **fallback_response["analysis"],
                        "concerning_clauses": [
                            {"title": "API Error", "description": f"OpenAI API failed: {str(api_err)}.", "reference": "N/A", "clause_text": ""}
                        ]
                    },
                    "error": f"API error: {str(api_err)}"
                }

        # If all retries fail
        logger.error("All retry attempts failed")
        return {
            **fallback_response,
            "risk_assessment": {
                **fallback_response["risk_assessment"],
                "description": "All retry attempts to process the document failed.",
                "risk_items": [
                    {"title": "Retry Limit Exceeded", "description": "All retry attempts to process the document failed.", "severity": "high", "reference": "N/A"}
                ]
            },
            "analysis": {
                **fallback_response["analysis"],
                "concerning_clauses": [
                    {"title": "Retry Limit Exceeded", "description": "All retry attempts to process the document failed.", "reference": "N/A", "clause_text": ""}
                ]
            },
            "error": "All retry attempts failed"
        }

    except Exception as e:
        logger.error(f"Unexpected error in analyze_lease_document: {e}")
        return {
            **fallback_response,
            "risk_assessment": {
                **fallback_response["risk_assessment"],
                "description": f"Unexpected error during analysis: {str(e)}.",
                "risk_items": [
                    {"title": "Unexpected Error", "description": f"An unexpected error occurred: {str(e)}.", "severity": "high", "reference": "N/A"}
                ]
            },
            "analysis": {
                **fallback_response["analysis"],
                "concerning_clauses": [
                    {"title": "Unexpected Error", "description": f"An unexpected error occurred: {str(e)}.", "reference": "N/A", "clause_text": ""}
                ]
            },
            "error": f"Unexpected error: {str(e)}"
        }
# Compare two lease documents
def compare_leases(original_text, proposed_text):
    original_lines = original_text.split('\n')
    proposed_lines = proposed_text.split('\n')
    differ = difflib.Differ()
    diff = list(differ.compare(original_lines, proposed_lines))

    added = []
    removed = []
    unchanged = []
    modified = []

    for line in diff:
        if line.startswith('+ '):
            added.append(line[2:].strip())
        elif line.startswith('- '):
            removed.append(line[2:].strip())
        elif line.startswith('  '):
            unchanged.append(line[2:].strip())

    for orig_line in removed[:]:
        for prop_line in added[:]:
            if difflib.SequenceMatcher(None, orig_line, prop_line).ratio() > 0.8:
                modified.append(prop_line)
                removed.remove(orig_line)
                added.remove(prop_line)
                break

    original_analysis = analyze_lease_document(original_text)
    proposed_analysis = analyze_lease_document(proposed_text)

    try:
        overall_risk_value = (original_analysis['risk_assessment']['risk_percentage'] + proposed_analysis['risk_assessment']['risk_percentage']) / 2
        overall_risk = "Low" if overall_risk_value < 50 else "Medium" if overall_risk_value <= 75 else "High"

        financial_impact = "Low"
        for item in original_analysis['risk_assessment']['risk_items'] + proposed_analysis['risk_assessment']['risk_items']:
            if item['severity'] == "high" and "rent" in item['title'].lower():
                financial_impact = "High"
                break
            elif item['severity'] == "medium":
                financial_impact = "Medium"

        legal_compliance = "Low"
        concerning_count = sum(1 for clause in original_analysis['document_summary']['clauses'] + proposed_analysis['document_summary']['clauses'] if clause['category'] in ["Concerning", "Renegotiate"])
        if concerning_count > 3:
            legal_compliance = "High"
        elif concerning_count > 1:
            legal_compliance = "Medium"
    except Exception as e:
        logger.error(f"Error computing risk assessment: {e}")
        overall_risk = "Unknown"
        financial_impact = "Unknown"
        legal_compliance = "Unknown"

    compliance_checks = [
        {
            "name": "Leasehold Reform Act Compliance",
            "status": "Compliant" if concerning_count == 0 else "Warning",
            "icon": "ri-checkbox-circle-line" if concerning_count == 0 else "ri-alert-line"
        },
        {
            "name": "Ground Rent Regulation",
            "status": "Non-Compliant" if any("ground rent" in item['title'].lower() for item in proposed_analysis['risk_assessment']['risk_items']) else "Compliant",
            "icon": "ri-close-circle-line" if any("ground rent" in item['title'].lower() for item in proposed_analysis['risk_assessment']['risk_items']) else "ri-checkbox-circle-line"
        },
        {
            "name": "Service Charge Transparency",
            "status": "Warning" if any("service charge" in item['title'].lower() for item in proposed_analysis['risk_assessment']['risk_items']) else "Compliant",
            "icon": "ri-alert-line" if any("service charge" in item['title'].lower() for item in proposed_analysis['risk_assessment']['risk_items']) else "ri-checkbox-circle-line"
        }
    ]

    try:
        base_premium = 150000
        local_avg_premium = f"£{base_premium:,.0f}"
        proposed_premium = f"£{base_premium * (1 + overall_risk_value / 100):,.0f}"
        extension_length = proposed_analysis['lease_information']['lease_term'] or "90 years"
    except Exception as e:
        logger.error(f"Error computing market benchmarks: {e}")
        local_avg_premium = "£150,000"
        proposed_premium = "£150,000"
        extension_length = "90 years"

    missing_provisions = [
        {
            "title": "Sustainability & Energy Efficiency",
            "description": "Current market trends show increased focus on environmental standards and energy performance.",
            "recommendation": "Include provisions for green improvements and clear cost-sharing arrangements for energy efficiency upgrades."
        },
        {
            "title": "Digital Infrastructure Rights",
            "description": "Modern leases commonly include provisions for high-speed internet installation and smart home technology.",
            "recommendation": "Add clauses permitting installation of fiber optic cables and smart home systems without requiring separate consent."
        },
        {
            "title": "Electric Vehicle Infrastructure",
            "description": "Growing demand for EV charging facilities in residential properties.",
            "recommendation": "Include rights to install EV charging points and framework for shared charging infrastructure."
        },
        {
            "title": "Building Safety Provisions",
            "description": "Recent legislation emphasizes enhanced building safety requirements and transparent reporting.",
            "recommendation": "Add specific clauses addressing building safety compliance, fire safety measures, and related cost arrangements."
        },
        {
            "title": "Flexible Living Arrangements",
            "description": "Modern leases increasingly accommodate home-based working and short-term letting platforms.",
            "recommendation": "Include clear provisions for home office use and regulated short-term letting rights with appropriate safeguards."
        },
        {
            "title": "Service Charge Transparency",
            "description": "Best practice now includes detailed service charge consultation and reporting requirements.",
            "recommendation": "Include provisions for digital service charge reporting, consultation processes, and reserve fund management."
        }
    ]

    positive_improvements = [
        {
            "title": "Enhanced Property Management Rights",
            "description": "The new lease provides clearer guidelines for property management and maintenance responsibilities.",
            "value_added": "Better defined responsibilities reduce potential disputes and ensure proper maintenance."
        },
        {
            "title": "Structured Fee Framework",
            "description": "Introduction of clear, capped administration fees for assignments and alterations provides transparency.",
            "value_added": "Predictable costs for future transactions and property improvements."
        },
        {
            "title": "Improved Access Protocol",
            "description": "New 48-hour notice requirement for landlord access ensures better privacy and planning.",
            "value_added": "Enhanced tenant privacy while maintaining necessary property management access."
        },
        {
            "title": "Structural Maintenance Clarity",
            "description": "Clear definition of structural repair responsibilities and cost-sharing arrangements.",
            "value_added": "Better long-term property maintenance and value preservation."
        }
    ]

    negotiation_areas = [
        {
            "title": "Clear Entry Rights",
            "description": "The new lease includes well-defined landlord entry rights with 48-hour notice period.",
            "benefit": "Provides clarity and protection for both parties regarding property access."
        },
        {
            "title": "Transparent Fee Structure",
            "description": "Clear administration fees for assignments and alterations with defined caps.",
            "benefit": "Provides certainty and prevents unexpected costs in the future."
        },
        {
            "title": "Reduced Extension Period",
            "description": "The proposed extension is only 70 years instead of the statutory 90 years.",
            "recommendation": "Request the full 90-year extension as provided by the Leasehold Reform Act."
        },
        {
            "title": "Increased Ground Rent",
            "description": "The proposed ground rent is higher than the original lease and exceeds market standards.",
            "recommendation": "Negotiate for a peppercorn (£0) ground rent, which is standard for lease extensions."
        },
        {
            "title": "Service Charge Increases",
            "description": "The service charge cap has increased from RPI+2% to RPI+3.5%.",
            "recommendation": "Maintain the original cap of RPI+2% or negotiate for RPI only."
        },
        {
            "title": "Additional Administration Fees",
            "description": "New administration fees for assignment, subletting, and alterations have been introduced.",
            "recommendation": "Request removal of these fees or cap them at a lower rate with inflation protection."
        },
        {
            "title": "Maintenance Responsibilities",
            "description": "Expanded tenant responsibilities for structural repairs without clear cost limits.",
            "recommendation": "Clarify cost-sharing arrangements and establish reasonable cost limits."
        }
    ]

    return {
        "total_differences": len(added) + len(removed) + len(modified),
        "added_clauses": len(added),
        "removed_clauses": len(removed),
        "modified_clauses": len(modified),
        "clause_changes": {
            "unchanged": unchanged,
            "added": added,
            "removed": removed,
            "modified": modified
        },
        "risk_assessment": {
            "overall_risk": overall_risk,
            "financial_impact": financial_impact,
            "legal_compliance": legal_compliance
        },
        "compliance_checks": compliance_checks,
        "market_benchmarks": {
            "local_avg_premium": local_avg_premium,
            "proposed_premium": proposed_premium,
            "extension_length": extension_length
        },
        "recommendations": [
            f"Address {financial_impact} financial risks identified in the proposed lease.",
            "Consult a legal advisor regarding compliance issues.",
            "Review added clauses for alignment with modern standards."
        ],
        "missing_provisions": missing_provisions,
        "positive_improvements": positive_improvements,
        "negotiation_areas": negotiation_areas,
        "original_analysis": original_analysis,
        "proposed_analysis": proposed_analysis
    }

# Updated analyze_service_charge function with error handling
def analyze_service_charge(text: str, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Analyzes service charge documents using OpenAI API to extract and categorize clauses,
    financial data, and compliance issues.

    Args:
        text (str): Extracted text from the uploaded service charge document.
        max_retries (int): Maximum number of retries for API calls.
        retry_delay (float): Delay between retries in seconds.

    Returns:
        Dict[str, Any]: Detailed analysis including clauses, financial breakdowns,
                        compliance checks, and discrepancies.
    """
    if not text or not text.strip():
        raise ValueError("No valid text provided for analysis")

    # Truncate input text to avoid exceeding context window
    max_input_length = 10000
    text = text[:max_input_length]

    # Define fallback response
    fallback_response = {
        "analysis_summary": [
            {
                "title": "Analysis Failure",
                "description": "Unable to analyze the document due to processing error.",
                "color": "red",
                "icon": "alert-line",
                "reference": "N/A"
            }
        ],
        "service_charge_breakdown": {
            "total": 0.0,
            "categories": [],
            "year": datetime.now().year
        },
        "budget_data": {
            "total_budget": 0.0,
            "spent_to_date": 0.0,
            "remaining": 0.0,
            "variance": "0%",
            "monthly_data": []
        },
        "year_end_vs_actual": [],
        "discrepancies": [
            {"title": "Analysis Error", "description": "Failed to process the document. Please try again or contact support."}
        ],
        "compliance_checks": {
            "status": "locked",
            "details": []
        },
        "extracted_clauses": []
    }

    try:
        # Prepare prompt for OpenAI API
        prompt = """
        You are an expert in leasehold property management and service charge analysis. Given the text of a service charge document (e.g., invoice, budget, or statement), perform the following tasks:

        1. **Extract Clauses**:
           - Identify 5-10 distinct clauses or sections related to service charges.
           - Categorize each clause into one of: Maintenance, Utilities, Management Fees, Insurance, Cleaning, Security, or Other.
           - Assign a unique clause ID (e.g., SC001).
           - Extract any monetary amounts (in GBP, e.g., £1,200.50) associated with each clause.
           - Assess compliance status (Compliant, Non-Compliant, Warning) based on terms like "Section 20", "unaccounted", or "missing notice".
           - Provide a reference (e.g., "Para 45" or "Section 3.2").

        2. **Financial Breakdown**:
           - Summarize financial data by category, including total amount and percentage of total.
           - Calculate the overall total service charge.

        3. **Budget Data**:
           - Estimate monthly budget data for 12 months (Jan-Dec) based on the total service charge.
           - Include actual spending (slightly varied, e.g., ±10%) and calculate remaining budget.
           - Provide a variance percentage compared to the previous year (estimate if not specified).

        4. **Year-End vs Actual**:
           - Compare estimated vs. actual costs for each category (use extracted amounts as actual, estimate ±5% for estimated).

        5. **Compliance Checks**:
           - Evaluate compliance with regulations (e.g., Section 20, transparency requirements).
           - For each check, provide a name, description, status (Compliant/Warning/Non-Compliant), color (green/yellow/red), icon (e.g., checkbox-circle-line), and sub-details with progress (0-100%).

        6. **Discrepancies**:
           - Identify potential issues (e.g., unaccounted costs, missing notices, overbilling).
           - Provide a title and description for each discrepancy.

        7. **Analysis Summary**:
           - Summarize key findings (e.g., high-cost clauses, compliance issues, positive aspects).
           - Include a title, description, color (green/yellow/red/orange), icon, and reference for each finding.

        Return the response as a complete, valid JSON object with the following structure:
        {
          "analysis_summary": [{"title": str, "description": str, "color": str, "icon": str, "reference": str}, ...],
          "service_charge_breakdown": {"total": float, "categories": [{"name": str, "amount": float, "percentage": float}, ...], "year": int},
          "budget_data": {"total_budget": float, "spent_to_date": float, "remaining": float, "variance": str, "monthly_data": [{"month": str, "budget": float, "actual": float}, ...]},
          "year_end_vs_actual": [{"category": str, "estimated": float, "actual": float}, ...],
          "discrepancies": [{"title": str, "description": str}, ...],
          "compliance_checks": {"status": str, "details": [{"name": str, "description": str, "color": str, "icon": str, "sub_details": [{"name": str, "status": str, "progress": int, "color": str}, ...]}, ...]},
          "extracted_clauses": [{"clause_id": str, "text": str, "category": str, "amount": float, "compliance_status": str, "reference": str}, ...]
        }

        Ensure the JSON is complete, properly formatted, and contains no syntax errors. If the document lacks sufficient detail, infer reasonable values based on typical service charge documents, but prioritize extracted data.

        Document text:
        {text}
        """

        # Retry loop for API calls
        for attempt in range(max_retries):
            try:
                # Call OpenAI API
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a precise and analytical assistant for leasehold property management."},
                        {"role": "user", "content": prompt.format(text=text)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=6000
                )

                # Parse API response
                response_content = response.choices[0].message.content
                try:
                    analysis_result = json.loads(response_content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Invalid JSON in OpenAI response: {response_content}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying API call (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return fallback_response

                # Validate response structure
                required_keys = [
                    "analysis_summary", "service_charge_breakdown", "budget_data",
                    "year_end_vs_actual", "discrepancies", "compliance_checks", "extracted_clauses"
                ]
                if not all(key in analysis_result for key in required_keys):
                    logger.error(f"Incomplete OpenAI response: missing keys {set(required_keys) - set(analysis_result.keys())}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying API call (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return fallback_response

                # Add user-specific compliance status
                user_email = session.get('user_email')
                users = load_users()
                user = next((u for u in users if u['email'] == user_email), None)
                analysis_result["compliance_checks"]["status"] = (
                    'unlocked' if user and user.get('subscription', {}).get('plan') == 'Premium' else 'locked'
                )
                if analysis_result["compliance_checks"]["status"] == "locked":
                    analysis_result["compliance_checks"]["details"] = []

                # Ensure year is current
                analysis_result["service_charge_breakdown"]["year"] = datetime.now().year

                return analysis_result

            except Exception as api_err:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {api_err}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying API call (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return fallback_response

        # If all retries fail, return fallback response
        logger.error("All retry attempts failed")
        return fallback_response

    except Exception as e:
        logger.error(f"Unexpected error in analyze_service_charge: {e}")
        return fallback_response

# File handling functions
def has_selectable_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text().strip()
            if text:
                doc.close()
                return True
        doc.close()
        return False
    except Exception as e:
        logger.error(f"Error checking PDF text: {e}")
        return False

def ocr_space_extract_text(file_path):
    try:
        with open(file_path, 'rb') as file:
            files = {'file': (secure_filename(file_path), file)}
            payload = {
                'apikey': OCR_SPACE_API_KEY,
                'language': 'eng',
                'isOverlayRequired': False
            }
            response = requests.post(OCR_SPACE_API_URL, files=files, data=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('IsErroredOnProcessing', True):
                error_message = result.get('ErrorMessage', ['Unknown error'])[0]
                raise Exception(f"OCR.Space API error: {error_message}")
            
            parsed_text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
            if not parsed_text.strip():
                raise Exception("No text extracted from the document.")
            
            return parsed_text
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR.Space API request failed: {e}")
        raise Exception(f"Failed to connect to OCR: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OCR response: {e}")
        raise Exception(f"Invalid response from OCR.Space API: {str(e)}")
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise Exception(f"Failed to extract text using OCR: {str(e)}")

# JSON file handling functions
def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return []

def save_users(users):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving users: {e}")
        raise Exception("Failed to save user data")

def load_documents():
    try:
        with open(DOCUMENTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def save_documents(documents):
    try:
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump(documents, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving documents: {e}")
        raise Exception("Failed to save document data")

def load_contact_messages():
    try:
        with open(CONTACT_MESSAGES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading contact messages: {e}")
        return []

def save_contact_messages(messages):
    try:
        with open(CONTACT_MESSAGES_FILE, 'w') as f:
            json.dump(messages, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving contact messages: {e}")
        raise Exception("Failed to save contact message data")

def load_properties():
    try:
        with open(PROPERTIES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading properties: {e}")
        return []

def save_properties(properties):
    try:
        with open(PROPERTIES_FILE, 'w') as f:
            json.dump(properties, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving properties: {e}")
        raise Exception("Failed to save property data")

def load_products():
    try:
        with open(PRODUCTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return []

def save_products(products):
    try:
        with open(PRODUCTS_FILE, 'w') as f:
            json.dump(products, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving products: {e}")
        raise Exception("Failed to save product data")

def load_events():
    try:
        with open(EVENTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading events: {e}")
        return []

def save_events(events):
    try:
        with open(EVENTS_FILE, 'w') as f:
            json.dump(events, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving events: {e}")
        raise Exception("Failed to save event data")

def load_reports():
    try:
        with open(REPORTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading reports: {e}")
        return []

def save_reports(reports):
    try:
        with open(REPORTS_FILE, 'w') as f:
            json.dump(reports, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving reports: {e}")
        raise Exception("Failed to save report data")

def load_legislative_updates():
    try:
        with open(LEGISLATIVE_UPDATES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading legislative updates: {e}")
        return []

def save_legislative_updates(updates):
    try:
        with open(LEGISLATIVE_UPDATES_FILE, 'w') as f:
            json.dump(updates, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving legislative updates: {e}")
        raise Exception("Failed to save legislative update data")

# Validation functions
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone_number(phone):
    pattern = r'^\+\d{10,15}$'
    return bool(re.match(pattern, phone))

def validate_text_length(value, max_length=500):
    return len(str(value)) <= max_length

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Email configuration
EMAIL_ADDRESS = 'mosesmaweu46@gmail.com'
EMAIL_PASSWORD = 'Gianna@2024'

def send_confirmation_email(email, username):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg['Subject'] = 'Welcome to IntelliLease!'

    body = f"""
    Dear {username},

    Welcome to IntelliLease! Thank you for signing up.

    We're excited to have you on board. Get started by exploring our leasehold analysis tools and managing your properties.

    Best regards,
    The IntelliLease Team
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
        logger.info(f"Confirmation email sent to {email}")
    except Exception as e:
        logger.error(f"Error sending email: {e}")

# Routes
@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        user = next((u for u in users if u['email'] == email), None)
        
        if user and check_password_hash(user['password'], password):
            session['user_email'] = email
            properties = load_properties()
            user_properties = [prop for prop in properties if prop.get('user_email') == email]
            if not user_properties:
                flash('Welcome! Please add your property details to get started.', 'success')
                return redirect(url_for('properties_page'))
            else:
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/services')
def services():
    if 'user_email' not in session:
        flash('Please log in to view services.', 'error')
        return redirect(url_for('login'))
    return render_template('service.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        phone = request.form.get('phone')
        
        if not validate_phone_number(phone):
            flash('Invalid phone number. Use format +44xxxxxxxxxx (10-15 digits).', 'error')
            return redirect(url_for('signup'))
        
        users = load_users()
        if any(u['email'] == email for u in users):
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        users.append({
            'email': email,
            'username': username,
            'password': hashed_password,
            'phone': phone,
            'subscription': {
                'plan': None,
                'status': 'inactive',
                'start_date': None,
                'expiry_date': None,
                'stripe_customer_id': None,
                'stripe_subscription_id': None
            },
            'notifications': {
                'email_notifications': True,
                'sms_notifications': False,
                'marketing_notifications': True
            }
        })
        save_users(users)
        flash('Sign-up successful! Please log in.', 'success')
        session['terms_accepted'] = False
        session['user_email'] = email
        send_confirmation_email(email, username)
        return redirect(url_for('ticks'))
    
    return render_template('signup.html')

@app.route('/ticks')
def ticks():
    if 'user_email' not in session:
        flash('Please log in to view the dashboard.', 'error')
        return redirect(url_for('login'))
    current_date = date.today().strftime('%B %d, %Y')
    return render_template('ticks.html', current_date=current_date)

@app.route('/accept_ticks', methods=['POST'])
def accept_terms():
    session['ticks_accepted'] = True
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        flash('Please log in to view your profile.', 'error')
        return redirect(url_for('login'))
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    documents = load_documents()
    user_documents = [doc for doc in documents if doc['user_email'] == user_email]
    
    return render_template('dashboard.html', user=user, documents=user_documents)

@app.route('/profile')
def profile():
    if 'user_email' not in session:
        flash('Please log in to view your profile.', 'error')
        return redirect(url_for('login'))
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    documents = load_documents()
    user_documents = [doc for doc in documents if doc['user_email'] == user_email]
    
    return render_template('profile.html', user=user, documents=user_documents)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('All fields are required.', 'error')
            return redirect(url_for('contact'))
        
        if not validate_email(email):
            flash('Invalid email format.', 'error')
            return redirect(url_for('contact'))
        
        try:
            messages = load_contact_messages()
            new_message = {
                'name': name,
                'email': email,
                'subject': subject,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'user_email': session.get('user_email', None)
            }
            messages.append(new_message)
            save_contact_messages(messages)
            flash('Your message has been sent successfully!', 'success')
            return redirect(url_for('contact'))
        except Exception as e:
            logger.error(f"Error saving contact message: {e}")
            flash('Failed to send message. Please try again.', 'error')
            return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    if 'user_email' not in session:
        flash('Please log in to upload documents.', 'error')
        return redirect(url_for('login'))
    return render_template('index.html')
def check_openai_api_key() -> Dict[str, Any]:
    """Check the validity of the OpenAI API key and log details."""
    result = {
        "is_valid": False,
        "details": {},
        "error": None
    }
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        result["error"] = "OPENAI_API_KEY not found in environment variables"
        return result
    truncated_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    logger.debug(f"Checking OpenAI API key: {truncated_key}")
    try:
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        result["is_valid"] = True
        logger.info(f"OpenAI API key is valid: {truncated_key}")
        if hasattr(response, 'headers') and 'openai-organization' in response.headers:
            result["details"]["organization_id"] = response.headers['openai-organization']
            logger.debug(f"Organization ID: {result['details']['organization_id']}")
        result["details"]["checked_at"] = datetime.now(timezone.utc).isoformat()
        logger.debug(f"API key check completed at: {result['details']['checked_at']}")
    except Exception as e:
        logger.error(f"OpenAI API key check failed: {str(e)}", exc_info=True)
        result["error"] = str(e)
        if "authentication" in str(e).lower():
            result["error"] = "Invalid API key: authentication failed"
        elif "rate limit" in str(e).lower():
            result["error"] = "Rate limit exceeded"
    return result
@app.route('/upload', methods=['POST'])
def upload_file():
    api_key_check = check_openai_api_key()

    if not api_key_check["is_valid"]:
        logger.error(f"Aborting upload: {api_key_check['error']}")
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to upload files"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        is_scanned = False
        extracted_text = ""

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            is_scanned = True
            extracted_text = ocr_space_extract_text(file_path)
        elif filename.lower().endswith('.pdf'):
            if not has_selectable_text(file_path):
                is_scanned = True
                extracted_text = ocr_space_extract_text(file_path)
            else:
                doc = fitz.open(file_path)
                extracted_text = ""
                for page in doc:
                    extracted_text += page.get_text().strip()
                doc.close()
        else:
            os.remove(file_path)
            return jsonify({"error": "Only lease documents (PDF, PNG, JPG, JPEG) are allowed."}), 400

        documents = load_documents()
        documents.append({
            'user_email': session['user_email'],
            'filename': filename,
            'upload_date': datetime.utcnow().isoformat(),
            'is_scanned': is_scanned
        })
        save_documents(documents)

        if not is_scanned:
            analysis_result = analyze_lease_document(extracted_text)
            reports = load_reports()
            reports.append({
                'user_email': session['user_email'],
                'title': f"Analysis for {filename}",
                'icon': 'ri-file-chart-line',
                'color': 'blue',
                'date': datetime.utcnow().strftime("%Y-%m-%d")
            })
            save_reports(reports)

        os.remove(file_path)

        if is_scanned:
            return jsonify({
                "is_scanned": True,
                "extracted_text": extracted_text
            })
        else:
            return jsonify(analysis_result), 200

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/analyze_with_ocr', methods=['POST'])
def analyze_with_ocr():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to analyze documents"}), 401
    
    data = request.get_json()
    extracted_text = data.get('extracted_text', '')
    
    if not extracted_text:
        return jsonify({"error": "No extracted text provided"}), 400

    try:
        analysis_result = analyze_lease_document(extracted_text)
        reports = load_reports()
        reports.append({
            'user_email': session['user_email'],
            'title': "OCR Analysis Report",
            'icon': 'ri-file-chart-line',
            'color': 'blue',
            'date': datetime.utcnow().strftime("%Y-%m-%d")
        })
        save_reports(reports)
        return jsonify(analysis_result), 200
    except Exception as e:
        logger.error(f"OCR analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if 'user_email' not in session:
        flash('Please log in to compare documents.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('comparison.html')
    
    if request.method == 'POST':
        if 'original_lease' not in request.files or 'proposed_lease' not in request.files:
            return jsonify({"error": "Both original and proposed lease files are required"}), 400
        
        original_file = request.files['original_lease']
        proposed_file = request.files['proposed_lease']
        
        if original_file.filename == '' or proposed_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        original_filename = secure_filename(original_file.filename)
        proposed_filename = secure_filename(proposed_file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{datetime.utcnow().timestamp()}_{original_filename}")
        proposed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"proposed_{datetime.utcnow().timestamp()}_{proposed_filename}")

        original_file.save(original_path)
        proposed_file.save(proposed_path)

        try:
            original_text = ""
            proposed_text = ""
            
            if original_filename.lower().endswith('.pdf'):
                if not has_selectable_text(original_path):
                    original_text = ocr_space_extract_text(original_path)
                else:
                    doc = fitz.open(original_path)
                    for page in doc:
                        original_text += page.get_text().strip()
                    doc.close()
            
            if proposed_filename.lower().endswith('.pdf'):
                if not has_selectable_text(proposed_path):
                    proposed_text = ocr_space_extract_text(proposed_path)
                else:
                    doc = fitz.open(proposed_path)
                    for page in doc:
                        proposed_text += page.get_text().strip()
                    doc.close()

            if not original_text or not proposed_text:
                raise Exception("Failed to extract text from one or both documents")

            comparison_result = compare_leases(original_text, proposed_text)
            
            documents = load_documents()
            documents.append({
                'user_email': session['user_email'],
                'original_filename': original_filename,
                'proposed_filename': proposed_filename,
                'upload_date': datetime.utcnow().isoformat(),
                'comparison_result': comparison_result
            })
            save_documents(documents)

            reports = load_reports()
            reports.append({
                'user_email': session['user_email'],
                'title': f"Comparison: {original_filename} vs {proposed_filename}",
                'icon': 'ri-scales-line',
                'color': 'indigo',
                'date': datetime.utcnow().strftime("%Y-%m-%d")
            })
            save_reports(reports)

            os.remove(original_path)
            os.remove(proposed_path)
            
            return jsonify(comparison_result), 200

        except Exception as e:
            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(proposed_path):
                os.remove(proposed_path)
            logger.error(f"Comparison error: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/properties', methods=['GET'])
def properties_page():
    if 'user_email' not in session:
        flash('Please log in to view properties.', 'error')
        return redirect(url_for('login'))
    return render_template('properties.html')

@app.route('/submit-property', methods=['POST'])
def submit_property():
    try:
        form_data = request.form.to_dict()
        files = request.files
        property_data = {
            "user_email": session.get('user_email'),
            "leasehold_property_information": {},
            "estate_information": {},
            "ownership_information": {},
            "leasehold_specific_details": {},
            "financial_information": {},
            "insurance_details": {},
            "maintenance_information": {},
            "document_storage": {
                "lease_agreements": [],
                "insurance_certificates": [],
                "safety_certificates": [],
                "property_surveys": [],
                "floor_plans": [],
                "other_documents": []
            }
        }

        required_fields = [
            'address1', 'city', 'postcode', 'property-type', 'bedrooms', 'bathrooms',
            'landlord-name', 'landlord-phone', 'landlord-email', 'landlord-address',
            'lease-start-date', 'lease-length', 'ground-rent', 'service-charge',
            'agent-company', 'agent-phone', 'agent-email',
            'purchase-date', 'purchase-price',
            'buildings-provider', 'buildings-policy', 'buildings-renewal', 'buildings-premium'
        ]
        missing_fields = [field for field in required_fields if field not in form_data or not form_data[field].strip()]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        for key, value in form_data.items():
            if not validate_text_length(value):
                return jsonify({"error": f"Field '{key}' exceeds 500 character limit"}), 400

        property_image_path = None
        if 'property-image' in files and files['property-image']:
            file = files['property-image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                property_image_path = file_path
            else:
                return jsonify({"error": "Invalid or unsupported file type for property image"}), 400

        property_data["leasehold_property_information"] = {
            "address_line_1": form_data.get('address1', ''),
            "address_line_2": form_data.get('address2', ''),
            "city": form_data.get('city', ''),
            "county": form_data.get('county', ''),
            "postcode": form_data.get('postcode', ''),
            "property_type": form_data.get('property-type', ''),
            "floor_number": int(form_data.get('floor-number', 0)),
            "bedrooms": int(form_data.get('bedrooms', 0)),
            "bathrooms": int(form_data.get('bathrooms', 0)),
            "square_footage": int(form_data.get('square-footage', 0)),
            "property_image": property_image_path or "Uncollected"
        }

        property_data["estate_information"] = {
            "estate_name": form_data.get('estate-name', ''),
            "total_properties": int(form_data.get('total-properties', 0)),
            "has_rtm": form_data.get('estate-has-rtm', 'false').lower() == 'true',
            "rtm_details": {
                "rtm_company_name": form_data.get('estate-rtm-company', ''),
                "rtm_acquisition_date": form_data.get('estate-rtm-date', ''),
                "rtm_notes": form_data.get('estate-rtm-notes', '')
            }
        }

        property_data["ownership_information"] = {
            "has_rtm": form_data.get('has-rtm', 'false').lower() == 'true',
            "rtm_details": {},
            "landlord_details": {
                "name": form_data.get('landlord-name', ''),
                "company_name": form_data.get('landlord-company', ''),
                "phone_number": form_data.get('landlord-phone', ''),
                "email_address": form_data.get('landlord-email', ''),
                "address": form_data.get('landlord-address', '')
            },
            "has_head_leaseholder": form_data.get('has-head-leaseholder', 'false').lower() == 'true',
            "head_leaseholder_details": {
                "name": form_data.get('head-leaseholder-name', ''),
                "company_name": form_data.get('head-leaseholder-company', ''),
                "phone_number": form_data.get('head-leaseholder-phone', ''),
                "email_address": form_data.get('head-leaseholder-email', ''),
                "address": form_data.get('head-leaseholder-address', '')
            }
        }

        lease_start_date = form_data.get('lease-start-date', '')
        lease_length = int(form_data.get('lease-length', 0))
        if lease_start_date and lease_length:
            start_date = datetime.strptime(lease_start_date, '%Y-%m-%d')
            current_date = datetime.now()
            years_passed = (current_date - start_date).days / 365.25
            years_remaining = max(0, round((lease_length - years_passed) * 10) / 10)
        else:
            years_remaining = 0

        property_data["leasehold_specific_details"] = {
            "lease_start_date": lease_start_date,
            "lease_length_years": lease_length,
            "years_remaining": years_remaining,
            "ground_rent": {
                "amount": float(form_data.get('ground-rent', 0)),
                "frequency": form_data.get('ground-rent-frequency', 'Annually')
            },
            "service_charge": {
                "amount": float(form_data.get('service-charge', 0)),
                "frequency": form_data.get('service-charge-frequency', 'Annually')
            },
            "managing_agent_details": {
                "company_name": form_data.get('agent-company', ''),
                "contact_person": form_data.get('agent-contact', ''),
                "phone_number": form_data.get('agent-phone', ''),
                "email_address": form_data.get('agent-email', ''),
                "website": form_data.get('agent-website', '')
            }
        }

        property_data["financial_information"] = {
            "purchase_date": form_data.get('purchase-date', ''),
            "purchase_price": float(form_data.get('purchase-price', 0)),
            "current_estimated_value": float(form_data.get('current-value', 0)),
            "has_mortgage": form_data.get('has-mortgage', 'false').lower() == 'true',
            "mortgage_details": {
                "lender_name": form_data.get('mortgage-lender', ''),
                "account_number": form_data.get('mortgage-account', ''),
                "monthly_payment": float(form_data.get('mortgage-payment', 0)),
                "interest_rate": float(form_data.get('interest-rate', 0)),
                "term_remaining_years": int(form_data.get('term-remaining', 0))
            }
        }

        property_data["insurance_details"] = {
            "buildings_insurance": {
                "provider": form_data.get('buildings-provider', ''),
                "policy_number": form_data.get('buildings-policy', ''),
                "renewal_date": form_data.get('buildings-renewal', ''),
                "annual_premium": float(form_data.get('buildings-premium', 0))
            },
            "has_contents_insurance": form_data.get('has-contents', 'false').lower() == 'true',
            "contents_insurance": {
                "provider": form_data.get('contents-provider', ''),
                "policy_number": form_data.get('contents-policy', ''),
                "renewal_date": form_data.get('contents-renewal', ''),
                "annual_premium": float(form_data.get('contents-premium', 0))
            }
        }

        issues = []
        i = 0
        while f'issue-date-{i}' in form_data:
            issues.append({
                "date_reported": form_data.get(f'issue-date-{i}', ''),
                "description": form_data.get(f'issue-description-{i}', ''),
                "status": form_data.get(f'issue-status-{i}', 'In Progress'),
                "resolution": form_data.get(f'issue-resolution-{i}', '')
            })
            i += 1

        repairs = []
        i = 0
        while f'repair-date-{i}' in form_data:
            repairs.append({
                "date": form_data.get(f'repair-date-{i}', ''),
                "description": form_data.get(f'repair-description-{i}', ''),
                "cost": float(form_data.get(f'repair-cost-{i}', 0)),
                "contractor": form_data.get(f'repair-contractor-{i}', '')
            })
            i += 1

        maintenance = []
        i = 0
        while f'maintenance-due-date-{i}' in form_data:
            maintenance.append({
                "due_date": form_data.get(f'maintenance-due-date-{i}', ''),
                "task": form_data.get(f'maintenance-task-{i}', ''),
                "estimated_cost": float(form_data.get(f'maintenance-cost-{i}', 0)),
                "priority": form_data.get(f'maintenance-priority-{i}', 'Medium')
            })
            i += 1

        property_data["maintenance_information"] = {
            "last_inspection_date": form_data.get('last-inspection', ''),
            "next_inspection_date": form_data.get('next-inspection', ''),
            "issues_reported": issues,
            "recent_repairs": repairs,
            "upcoming_maintenance": maintenance
        }

        document_types = ['lease', 'insurance', 'safety', 'survey', 'floorplan', 'other']
        for doc_type in document_types:
            i = 0
            while f'{doc_type}-name-{i}' in form_data:
                file_path = None
                if f'{doc_type}-file-{i}' in files and files[f'{doc_type}-file-{i}']:
                    file = files[f'{doc_type}-file-{i}']
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                property_data["document_storage"][f"{doc_type}_agreements" if doc_type == 'lease' else f"{doc_type}_certificates" if doc_type in ['insurance', 'safety'] else f"{doc_type}_surveys" if doc_type == 'survey' else f"{doc_type}_plans" if doc_type == 'floorplan' else f"other_documents"].append({
                    "name": form_data.get(f'{doc_type}-name-{i}', ''),
                    "file_name": file_path or form_data.get(f'{doc_type}-file-name-{i}', '')
                })
                i += 1

        properties = load_properties()
        properties.append(property_data)
        save_properties(properties)

        return jsonify({"message": "Form submitted successfully!"}), 200

    except Exception as e:
        logger.error(f"Error submitting property: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/service/upload', methods=['POST'])
def service_upload():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to upload files"}), 401

    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    document_types = request.form.getlist('document_types')
    property_address = request.form.get('property_address')
    lease_ref = request.form.get('lease_ref')
    billing_period = request.form.get('billing_period')

    if not all([document_types, property_address, lease_ref, billing_period]):
        return jsonify({"error": "All fields are required"}), 400

    try:
        extracted_texts = []
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({"error": f"File {file.filename} is not a supported type"}), 400
            if file.content_length > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({"error": f"File {file.filename} exceeds 5MB limit"}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = ocr_space_extract_text(file_path)
            elif filename.lower().endswith('.pdf'):
                if not has_selectable_text(file_path):
                    text = ocr_space_extract_text(file_path)
                else:
                    doc = fitz.open(file_path)
                    text = "".join(page.get_text().strip() for page in doc)
                    doc.close()
            else:
                os.remove(file_path)
                return jsonify({"error": "Unsupported file type"}), 400

            extracted_texts.append(text)
            os.remove(file_path)

        documents = load_documents()
        for i, file in enumerate(files):
            documents.append({
                'user_email': session['user_email'],
                'filename': secure_filename(file.filename),
                'upload_date': datetime.utcnow().isoformat(),
                'document_type': document_types[i % len(document_types)],
                'property_address': property_address,
                'lease_ref': lease_ref,
                'billing_period': billing_period
            })
        save_documents(documents)

        combined_text = "\n".join(extracted_texts)
        analysis_result = analyze_service_charge(combined_text)

        reports = load_reports()
        reports.append({
            'user_email': session['user_email'],
            'title': f"Service Charge Analysis for {lease_ref}",
            'icon': 'ri-money-dollar-circle-line',
            'color': 'orange',
            'date': datetime.utcnow().strftime("%Y-%m-%d")
        })
        save_reports(reports)

        return jsonify(analysis_result), 200

    except Exception as e:
        logger.error(f"Service upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/service-data', methods=['GET'])
def get_service_data():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to access data"}), 401

    year = request.args.get('year', '2025')
    period = request.args.get('period', 'Annual')

    try:
        analysis_result = analyze_service_charge("")
        if period == 'Quarterly':
            analysis_result['budget_data']['monthly_data'] = [
                {"month": f"Q{i+1}", "budget": sum(m['budget'] for m in analysis_result['budget_data']['monthly_data'][i*3:(i+1)*3]),
                 "actual": sum(m['actual'] for m in analysis_result['budget_data']['monthly_data'][i*3:(i+1)*3])}
                for i in range(4)
            ]
        elif period == 'Annual':
            analysis_result['budget_data']['monthly_data'] = [
                {"month": year, "budget": analysis_result['budget_data']['total_budget'],
                 "actual": analysis_result['budget_data']['spent_to_date']}
            ]
        analysis_result['service_charge_breakdown']['year'] = int(year)

        return jsonify(analysis_result), 200
    except Exception as e:
        logger.error(f"Error fetching service data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/properties/data', methods=['GET'])
def get_properties():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to access properties"}), 401
    
    user_email = session['user_email']
    try:
        properties = load_properties()
        user_properties = [prop for prop in properties if prop.get('user_email') == user_email]
        return jsonify(user_properties), 200
    except Exception as e:
        logger.error(f"Error retrieving properties: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/products', methods=['GET'])
def get_products():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to access products"}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        products = load_products()
        if user.get('subscription', {}).get('plan') in ['Standard', 'Premium']:
            for product in products:
                if product['name'] == 'Premium Consultation':
                    product['status'] = 'Active'
        return jsonify(products), 200
    except Exception as e:
        logger.error(f"Error retrieving products: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/reports', methods=['GET'])
def get_reports():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to access reports"}), 401
    
    user_email = session['user_email']
    try:
        reports = load_reports()
        user_reports = [report for report in reports if report.get('user_email') == user_email]
        return jsonify(user_reports), 200
    except Exception as e:
        logger.error(f"Error retrieving reports: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/legislative_updates', methods=['GET'])
def get_legislative_updates():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in to access legislative updates"}), 401
    
    try:
        updates = load_legislative_updates()
        return jsonify(updates), 200
    except Exception as e:
        logger.error(f"Error retrieving legislative updates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_user_data')
def get_user_data():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    def get_initials(name):
        return ''.join(word[0] for word in name.split() if word).upper()[:2]

    cohort_members = [
        {"name": "John Doe", "email": "john.doe@example.com", "initials": get_initials("John Doe")},
        {"name": "Emma Wilson", "email": "emma.wilson@example.com", "initials": get_initials("Emma Wilson")},
        {"name": "Michael Brown", "email": "michael.brown@example.com", "initials": get_initials("Michael Brown")}
    ] if user.get('cohort_members') else []

    billing_history = [
        {"date": "08 Jun 2025", "description": "Professional Plan - Monthly", "amount": "£49.99"},
        {"date": "08 May 2025", "description": "Professional Plan - Monthly", "amount": "£49.99"},
        {"date": "08 Apr 2025", "description": "Professional Plan - Monthly", "amount": "£49.99"}
    ] if user.get('subscription') and user['subscription'].get('status') == 'active' else []

    payment_method = {
        "last4": "4242",
        "expiry": "06/2026"
    } if user.get('subscription') and user['subscription'].get('stripe_customer_id') else None

    response = {
        "first_name": user.get('username', '').split()[0] if ' ' in user.get('username', '') else user.get('username', ''),
        "last_name": user.get('username', '').split()[1] if ' ' in user.get('username', '') and len(user.get('username', '').split()) > 1 else '',
        "email": user['email'],
        "phone": user.get('phone', ''),
        "email_notifications": user.get('notifications', {}).get('email_notifications', True),
        "sms_notifications": user.get('notifications', {}).get('sms_notifications', False),
        "marketing_notifications": user.get('notifications', {}).get('marketing_notifications', True),
        "cohort_name": user.get('cohort_name', 'Property Investors UK - 2025'),
        "cohort_type": user.get('cohort_type', 'Property Investment'),
        "cohort_members": cohort_members,
        "subscription": user.get('subscription', {}),
        "payment_method": payment_method,
        "billing_history": billing_history
    }
    return jsonify(response)

@app.route('/update_user_data', methods=['POST'])
def update_user_data():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    user['phone'] = data.get('phone', user.get('phone', ''))
    first_name = data.get('first_name', '')
    last_name = data.get('last_name', '')
    user['username'] = ' '.join(part for part in [first_name, last_name] if part)
    save_users(users)
    return jsonify({"message": "User data updated successfully"})

@app.route('/update_notification_preferences', methods=['POST'])
def update_notification_preferences():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    user.setdefault('notifications', {})
    user['notifications'].update({
        'email_notifications': data.get('email_notifications', True),
        'sms_notifications': data.get('sms_notifications', False),
        'marketing_notifications': data.get('marketing_notifications', True)
    })
    save_users(users)
    return jsonify({"message": "Notification preferences updated successfully"})

@app.route('/update_cohort_settings', methods=['POST'])
def update_cohort_settings():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in"}), 401
    
    user_email = session['user_email']
    users = load_users()
    user = next((u for u in users if u['email'] == user_email), None)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    user['cohort_name'] = data.get('cohort_name', user.get('cohort_name', 'Property Publishers UK - 2025'))
    user['cohort_type'] = data.get('cohort_type', user.get('cohort_type', 'Property Investment'))
    user['cohort_privacy'] = data.get('privacy', user.get('cohort_privacy', 'private'))
    save_users(users)
    return jsonify({"message": "Cohort settings updated successfully"})

@app.route('/get_all_user_data', methods=['GET'])
def get_all_user_data():
    if 'user_email' not in session:
        return jsonify({"error": "Please log in"}), 401

    user_email = session['user_email']

    try:
        users = load_users()
        user = next((u for u in users if u['email'] == user_email), None)
        if not user:
            return jsonify({"error": "User not found"}), 404

        documents = load_documents()
        user_documents = [doc for doc in documents if doc.get('user_email') == user_email]

        properties = load_properties()
        user_properties = [prop for prop in properties if prop.get('user_email') == user_email]

        reports = load_reports()
        user_reports = [report for report in reports if report.get('user_email') == user_email]

        response_data = {
            "user": user,
            "documents": user_documents,
            "properties": user_properties,
            "reports": user_reports
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error retrieving user data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/termz')
def termz():
    current_date = date.today().strftime('%B %d, %Y')
    return render_template('termz.html', current_date=current_date)

if __name__ == '__main__':
    app.run(debug=True)