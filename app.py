from flask import Flask, request, jsonify, make_response, Response, send_file, after_this_request
from flask_cors import CORS
from openai import OpenAI, InvalidWebhookSignatureError
from dotenv import load_dotenv
import os
import PyPDF2
from PIL import Image
import io 
import base64
import pandas as pd
import csv
from datetime import datetime
import requests
from serpapi import GoogleSearch
from firecrawl import FirecrawlApp
from firebase_admin import credentials, firestore, storage, initialize_app
import re
import json
import tiktoken
import hashlib
import torch
import contextlib
import traceback
import time
from urllib.parse import urlparse
# from instagrapi import Client
import base64
from io import BytesIO
import math
import matplotlib
matplotlib.use('Agg')  # Add this line at the top
import threading

load_dotenv()
print("Loaded SERPAPI_KEY:", os.getenv('SERPAPI_KEY'))
app = Flask(__name__)

# Configure CORS with more specific settings
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:3000", "https://notebook-mvp.vercel.app"],
         "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
         "expose_headers": ["Content-Type", "Content-Disposition"],
         "supports_credentials": True,
         "max_age": 86400
     }})

def add_cors_headers(response):
    """Helper function to add CORS headers to any response"""
    origin = request.headers.get('Origin')
    # Always set the Access-Control-Allow-Origin header for preflight requests
    if origin in ["http://localhost:3000", "https://notebook-mvp.vercel.app"]:
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        # For preflight requests, set the header to the actual origin
        response.headers['Access-Control-Allow-Origin'] = origin or '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Content-Disposition'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    return add_cors_headers(response)

@app.before_request
def handle_preflight():
    """Global OPTIONS handler for all routes"""
    if request.method == "OPTIONS":
        response = make_response()
        return add_cors_headers(response), 204
    return None

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Reset counter on application start
api_call_count = 0
user_api_key = None
MAX_FREE_CALLS = 3

# Initialize Firebase
try:
    firebase_creds_str = os.getenv('FIREBASE_CREDENTIALS')
    if not firebase_creds_str:
        raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")
    
    # Parse the JSON string and properly handle the private key
    firebase_creds = json.loads(firebase_creds_str)
    
    # Fix private key formatting
    if isinstance(firebase_creds.get('private_key'), str):
        firebase_creds['private_key'] = firebase_creds['private_key'].replace('\\n', '\n')
    
    cred = credentials.Certificate(firebase_creds)
    firebase_app = initialize_app(cred, {
        'storageBucket': 'notebookmvp.firebasestorage.app'
    })
    db = firestore.client()
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")
    raise

# Add near the top after app initialization
print("Available routes:", [str(rule) for rule in app.url_map.iter_rules()])

# After app initialization
print("\n=== Registered Routes ===")
for rule in app.url_map.iter_rules():
    print(f"Endpoint: {rule.endpoint}, Methods: {rule.methods}, URL: {rule.rule}")
# print("======================\n")

@app.route('/test', methods=['GET', 'OPTIONS'])
def hello_world():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    return add_cors_headers(make_response('Hello, World!'))

@app.route('/test_variable', methods=['GET', 'OPTIONS'])
def test_route():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    variable = request.args.get('variable')
    return add_cors_headers(make_response(f'variable received: {variable}'))

# api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Initialize Firecrawl client
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
if not firecrawl_api_key:
    raise ValueError("FIRECRAWL_API_KEY not found in environment variables")

firecrawl_client = FirecrawlApp(api_key=firecrawl_api_key)

def get_active_api_key():
    """Returns user API key if set, otherwise falls back to env key"""
    global user_api_key
    return user_api_key if user_api_key else os.getenv('OPENAI_API_KEY')

def check_api_key_usage():
    """Tracks API usage and returns whether user needs their own key"""
    global api_call_count
    api_call_count += 1
    if api_call_count > 3 and not user_api_key:
        return False
    return True

# Code for handling files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_file(file, file_type):
    """Process different file types and return their data"""
    if file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    elif file_type == "image":
        # Convert image to base64 for easy transmission
        img = Image.open(file)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    request_id = request.form.get('request_id') or request.args.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        if 'file' not in request.files:
            return add_cors_headers(jsonify({'error': 'No file part'})), 400
        file = request.files['file']
        file_type = request.form.get('type')
        if file.filename == '':
            return add_cors_headers(jsonify({'error': 'No selected file'})), 400
        try:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
            with open(filename, 'rb') as f:
                processed_data = process_file(f, file_type)
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
            response = jsonify({
                'success': True,
                'filename': file.filename,
                'filepath': filename,
                'processed_data': processed_data
            })
            return add_cors_headers(response)
        except Exception as e:
            return add_cors_headers(jsonify({'error': str(e)})), 500
    finally:
        cleanup_request(request_id)

@app.route('/api/set-api-key', methods=['POST', 'OPTIONS'])
def set_api_key():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    global user_api_key
    data = request.json
    user_api_key = data.get('api_key')
    response = jsonify({"success": True})
    return add_cors_headers(response)

def call_model(system_prompt: str, user_prompt: str, sources: dict = None) -> dict:
    """Sends prompts to OpenAI and returns the response"""
    try:
        if not check_api_key_usage():
            return {
                "response": "Please add your own API key to continue using the service.",
                "success": False,
                "needs_api_key": True
            }

        client = OpenAI(api_key=get_active_api_key())
        # Prepare context from sources
        context = ""
        if sources:
            for name, data in sources.items():
                if name in user_prompt:  # Only include referenced sources
                    context += f"\nContent for {name}: {data}\n"
        
        # Modify the content to include context if it exists
        content = f"{system_prompt} {context} {user_prompt}" if context else f"{system_prompt} {user_prompt}"
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4",
        )
        return {
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        return {
            "response": str(e),
            "success": False
        }
    
@app.route('/api/call-model', methods=['POST', 'OPTIONS'])
def api_call_model():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        import time
        time.sleep(5)  # TEMPORARY DELAY FOR TESTING
        system_prompt = data.get('system_prompt', '')
        user_prompt = data.get('user_prompt', '')
        save_as_csv = data.get('save_as_csv', False)
        result = call_model(system_prompt, user_prompt)
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        if save_as_csv and result['success']:
            response_lines = [line.strip() for line in result['response'].split('\n') if line.strip()]
            csv_data = io.StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(['Response'])
            for line in response_lines:
                csv_writer.writerow([line])
            output = make_response(jsonify({
                **result,
                'csv_content': csv_data.getvalue(),
                'filename': 'response.csv'
            }))
            output.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
            return add_cors_headers(output)
        response = make_response(jsonify(result))
        response.set_cookie('session_active', 'true')
        return add_cors_headers(response)
    finally:
        cleanup_request(request_id)

@app.route('/api/call-model-with-source', methods=['POST', 'OPTIONS'])
def api_call_model_with_source():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        system_prompt = data.get('system_prompt', '')
        user_prompt = data.get('user_prompt', '')
        download_url = data.get('download_url', '')
        if not download_url:
            return add_cors_headers(jsonify({"success": False, "error": "No download URL provided"})), 400
        try:
            response = requests.get(download_url)
            response.raise_for_status()
            is_pdf = ('application/pdf' in response.headers.get('Content-Type', '') or download_url.lower().endswith('.pdf'))
            if is_pdf:
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                processed_data = ""
                for page in pdf_reader.pages:
                    processed_data += page.extract_text() + "\n"
            else:
                processed_data = response.text
        except Exception as e:
            return add_cors_headers(jsonify({"success": False, "error": f"Failed to download/process file: {str(e)}"})), 500
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        source_system_prompt = f"You are a helpful assistant. The user has given you the following source to use to answer questions. Please only use this source, and this source only, when helping the user. Source: {processed_data}\n\n{system_prompt}"
        result = call_model(source_system_prompt, user_prompt)
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        response = make_response(jsonify(result))
        response.set_cookie('session_active', 'true')
        return add_cors_headers(response)
    finally:
        cleanup_request(request_id)

@app.route('/oai', methods=['GET', 'OPTIONS'])
def oai_route():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    request_id = request.args.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        system_prompt = request.args.get('system')
        user_prompt = request.args.get('user')
        sources = request.args.get('sources')
        client = OpenAI(api_key=api_key)
        context = ""
        if sources:
            sources_dict = eval(sources)
            for name, data in sources_dict.items():
                if name in user_prompt:
                    context += f"\nContent for {name}: {data}\n"
        content = f"{system_prompt} {context} {user_prompt}" if context else f"{system_prompt} {user_prompt}"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4",
        )
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        return add_cors_headers(jsonify({
            "response": response.choices[0].message.content,
            "success": True
        }))
    except Exception as e:
        return add_cors_headers(jsonify({"error": str(e), "success": False})), 500
    finally:
        cleanup_request(request_id)

@app.route('/api/check-api-key', methods=['GET', 'OPTIONS'])
def check_api_key():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    global user_api_key, api_call_count
    response = jsonify({
        'hasCustomKey': bool(user_api_key),
        'apiKey': user_api_key if user_api_key else '',
        'count': api_call_count
    })
    return add_cors_headers(response)

@app.route('/api/remove-api-key', methods=['POST', 'OPTIONS'])
def remove_api_key():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    global user_api_key, api_call_count
    user_api_key = None
    api_call_count = 0
    return add_cors_headers(jsonify({'success': True}))

@app.route('/api/get-count', methods=['GET', 'OPTIONS'])
def get_count():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    global api_call_count
    return add_cors_headers(jsonify({'count': api_call_count}))


@app.route('/api/process-csv', methods=['POST', 'OPTIONS'])
def process_csv():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        file_path = data.get('filePath')
        filter_criteria = data.get('filterCriteria', [])
        df = pd.read_csv(file_path)
        original_count = len(df)
        for criteria in filter_criteria:
            column = criteria['column']
            operator = criteria['operator']
            value = criteria['value']
            if operator == "equals":
                df = df[df[column] == value]
            elif operator == "not equals":
                df = df[df[column] != value]
            elif operator == "contains":
                df = df[df[column].astype(str).str.contains(value, na=False)]
            elif operator == "starts with":
                df = df[df[column].astype(str).str.startswith(value, na=False)]
            elif operator == "ends with":
                df = df[df[column].astype(str).str.endswith(value, na=False)]
            elif operator == "greater than":
                df = df[pd.to_numeric(df[column], errors='coerce') > float(value)]
            elif operator == "less than":
                df = df[pd.to_numeric(df[column], errors='coerce') < float(value)]
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        processed_data = df.to_string()
        raw_data = df.to_dict('records')
        response_data = {
            'success': True,
            'processedData': processed_data,
            'rawData': raw_data,
            'metadata': {
                'original_row_count': original_count,
                'filtered_row_count': len(df),
                'columns': df.columns.tolist(),
                'applied_filters': filter_criteria
            }
        }
        return add_cors_headers(jsonify(response_data))
    except Exception as e:
        return add_cors_headers(jsonify({'success': False, 'error': str(e)})), 500
    finally:
        cleanup_request(request_id)

def send_checkin_email(to_email=None):
    """Sends a check-in notification email"""
    mailgun_api_key = os.getenv('MAILGUN_API_KEY')
    if not mailgun_api_key:
        print("Mailgun API key not found in environment variables")
        return None
        
    today = datetime.now().strftime("%B %d")
    try:
        response = requests.post(
            "https://api.mailgun.net/v3/robots.yourca.io/messages",
            auth=("api", mailgun_api_key),
            data={
                "from": "Agent Check-Ins <postmaster@robots.yourca.io>",
                "to": to_email or "sahil sinha <sahil@lytix.co>",
                "subject": f"Agent Checkin - {today}",
                "text": """Hey there,
Your agent has hit a check-in and is waiting on you. This is your chance to review and tweak any details before it keeps going.

Take a quick look and continue: https://notebook-mvp.vercel.app/"""
            }
        )
        return response
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return None

@app.route('/api/send-checkin-email', methods=['GET', 'OPTIONS'])
def checkin_email():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    request_id = request.args.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        email = request.args.get('email')
        response = send_checkin_email(email)
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        if response and response.status_code == 200:
            return add_cors_headers(jsonify({
                "success": True,
                "message": "Email sent successfully",
                "sent_to": email or "default email"
            }))
        else:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Failed to send email",
                "status_code": response.status_code if response else None,
                "details": response.text if response else "Failed to send email"
            })), 500
    finally:
        cleanup_request(request_id)

def perform_google_search(query: str = None, engine_type: str = "search", topic_token: str = None, section_token: str = None, window: str = None, trend: str = None, index_market: str = None, num: int = 10) -> dict:
    """Performs a Google search using SerpAPI"""
    try:
        params = {
            "api_key": os.getenv('SERPAPI_KEY'),
            # "api_key": "ec7d8f1da6798c955d5d6af9263843f98c18bd49b1fee485d7e3f25e4c3c1b0d",
            "gl": "us",
            "hl": "en"
        }
        if engine_type == "search":
            params["engine"] = "google"
            params["q"] = query
            params["num"] = num
        elif engine_type == "news":
            params["engine"] = "google_news"
            params["num"] = num  # Add this line
            if query:
                params["q"] = query
            elif topic_token:
                params["topic_token"] = topic_token
                if section_token:
                    params["section_token"] = section_token
            else:
                return {
                    "success": False,
                    "error": "News search requires either a query or topic token"
                }
        elif engine_type == "finance":
            params["engine"] = "google_finance"
            params["q"] = query
            params["num"] = num  # Add this line
            if window:
                params["window"] = window
        elif engine_type == "markets":
            params["engine"] = "google_finance_markets"
            params["num"] = num  # Add this line
            if not trend:
                return {
                    "success": False,
                    "error": "Markets search requires a trend parameter"
                }
            params["trend"] = trend
            if trend == "indexes" and index_market:
                params["index_market"] = index_market
        else:
            return {
                "success": False,
                "error": f"Unsupported engine type: {engine_type}"
            }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Handle different result types based on engine
        if engine_type == "search":
            return {
                "success": True,
                "results": results.get("organic_results", [])  # Do NOT slice here, slice in the endpoint
            }
        elif engine_type == "news":
            return {
                "success": True,
                "results": results.get("news_results", [])
            }
        elif engine_type == "finance":
            return {
                "success": True,
                "results": results
            }
        elif engine_type == "markets":
            return {
                "success": True,
                "results": results.get("market_trends", [])
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/api/search', methods=['GET', 'POST', 'OPTIONS'])
def search():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    if request.method == 'POST':
        data = request.json
        request_id = data.get('request_id')
    else:  # GET
        request_id = request.args.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({
            "success": False,
            "error": "Missing required field: 'request_id'"
        })), 400
    register_request(request_id)
    print(f"Received search request with request_id: {request_id}")
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            })), 499
        import time
        time.sleep(10)  # TEMPORARY DELAY FOR TESTING
        if request.method == 'POST':
            data = request.json
            query = data.get('query')
            engine = data.get('engine', 'search')
            topic_token = data.get('topic_token')
            section_token = data.get('section_token')
            window = data.get('window')
            trend = data.get('trend')
            index_market = data.get('index_market')
            num = int(data.get('num', 10))
        else:  # GET
            query = request.args.get('q')
            engine = request.args.get('engine', 'search')
            topic_token = request.args.get('topic_token')
            section_token = request.args.get('section_token')
            window = request.args.get('window')
            trend = request.args.get('trend')
            index_market = request.args.get('index_market')
            num = int(request.args.get('num', 10))
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            })), 499
        result = perform_google_search(query, engine, topic_token, section_token, window, trend, index_market, num=num)
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            })), 499
        if "results" in result and isinstance(result["results"], list):
            result["results"] = result["results"][:num]
        return add_cors_headers(jsonify(result))
    finally:
        cleanup_request(request_id)

def send_email(to_email, subject, body):
    """Sends an email using Mailgun API
    
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject line
        body (str): Email body text
        
    Returns:
        Response object if successful, None if failed
    """
    mailgun_api_key = os.getenv('MAILGUN_API_KEY')
    if not mailgun_api_key:
        print("Mailgun API key not found in environment variables")
        return None
        
    try:
        response = requests.post(
            "https://api.mailgun.net/v3/robots.yourca.io/messages",
            auth=("api", mailgun_api_key),
            data={
                "from": "Agent Check-Ins <postmaster@robots.yourca.io>",
                "to": to_email,
                "subject": subject,
                "text": body
            }
        )
        return response
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return None

@app.route('/api/send-email', methods=['POST', 'GET', 'OPTIONS'])
def send_email_endpoint():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    if request.method == 'POST':
        data = request.json
        request_id = data.get('request_id')
    else:
        request_id = request.args.get('request_id')
    if not request_id:
        return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
    register_request(request_id)
    try:
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        if request.method == 'POST':
            data = request.json
            email = data.get('email')
            subject = data.get('subject')
            body = data.get('body')
        else:
            email = request.args.get('email')
            subject = request.args.get('subject')
            body = request.args.get('body')
        if not all([email, subject, body]):
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required fields: email, subject, and body are required"
            })), 400
        response = send_email(email, subject, body)
        if is_request_cancelled(request_id):
            return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
        if response and response.status_code == 200:
            return add_cors_headers(jsonify({
                "success": True,
                "message": "Email sent successfully",
                "sent_to": email
            }))
        else:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Failed to send email",
                "status_code": response.status_code if response else None,
                "details": response.text if response else "Failed to send email"
            })), 500
    except Exception as e:
        return add_cors_headers(jsonify({"success": False, "error": str(e)})), 500
    finally:
        cleanup_request(request_id)

# website processing stuff
# initializings and functions 

# Utility Functions

def scrape_website(url: str) -> str:
    """
    Scrapes a website using Firecrawl and returns its content as markdown.
    """
    try:
        scrape_status = firecrawl_client.scrape_url(
            url, params={'formats': ['markdown'], 'timeout': 60000}
        )
        return scrape_status.get('markdown', '')
    except Exception as e:
        raise Exception(f"Firecrawl error: {str(e)}")
    
# Function to chunk content based on tokens
def chunk_text(text: str, max_tokens: int = 512):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # ✅ Explicitly get encoding
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [" ".join(tokenizer.decode(chunk)) for chunk in chunks]

# Function to generate embeddings (Using OpenAI API with text-embedding-3-small)
def get_embeddings(chunks: list) -> list:
    """Generate embeddings for text chunks using OpenAI's `text-embedding-3-small` model."""
    response = client.embeddings.create(
        input=chunks,  # OpenAI now supports batch processing
        model="text-embedding-3-small"
    )
    return [embedding.embedding for embedding in response.data]

# Function to sanitize filenames for Firebase Storage
def sanitize_filename(url: str) -> str:
    """
    Sanitizes URLs consistently for both Firestore document IDs and Storage filenames.
    Removes http(s):// and converts special characters to underscores.
    """
    # Remove http:// or https:// prefix
    url = re.sub(r'^https?://', '', url)
    
    # Replace slashes and other special characters with underscores
    sanitized = re.sub(r'[^\w\-_]', '_', url)
    
    # Remove any duplicate underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized.lower()  # Convert to lowercase for consistency

# Function to upload chunked data to Firebase Cloud Storage (Per User)
def upload_to_firebase(userid, url, chunks, embeddings, filename=None):
    """
    Uploads chunked data + embeddings to Firebase Cloud Storage under:
    gs://notebookmvp.firebasestorage.app/users/{userid}/{filename}.json

    Args:
        userid (str): User ID
        url (str): URL being processed
        chunks (list): Chunked text content
        embeddings (list): Corresponding embeddings
        filename (str, optional): Custom file name (if None, defaults to a hash of the URL)

    Returns:
        str: Firebase Storage file path
    """
    bucket = storage.bucket()  # Gets the configured bucket
    
    # If no filename is provided, default to a hash of the URL
    if filename is None:
        filename = hashlib.sha256(url.encode()).hexdigest()
    
    # Sanitize filename to prevent folder creation issues
    filename = sanitize_filename(filename)

    filename = f"users/{userid}/{filename}.json"  # User-specific path

    blob = bucket.blob(filename)
    data = {"url": url, "chunks": chunks, "embeddings": embeddings}
    
    # Upload data to Firebase Cloud Storage
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    
    # Return the final storage path
    return f"gs://notebookmvp.firebasestorage.app/{filename}"

def process_url(userid: str, url: str, nickname: str = None):
    """
    Processes a URL for a specific user:
    - Fetches website content
    - Chunks the content
    - Generates embeddings
    - Uploads result to Firebase Cloud Storage
    - Stores metadata in Firestore under users/{uid}/files/{sanitized_url}
    """
    print(f"Fetching: {url}")
    content = scrape_website(url)  # Using Firecrawl

    if not content:
        print("No content retrieved.")
        return

    print("Chunking content...")
    chunks = chunk_text(content)

    print("Generating embeddings with OpenAI `text-embedding-3-small`...")
    embeddings = get_embeddings(chunks)

    print("Uploading to Firebase...")
    storage_path = upload_to_firebase(userid, url, chunks, embeddings, filename=nickname or sanitize_filename(url))

    # **Sanitize URL to use as a Firestore document ID**
    sanitized_url = sanitize_filename(url)

    # Store metadata in Firestore
    file_ref = db.collection("users").document(userid).collection("files").document(sanitized_url)
    file_data = {
        "created_at": datetime.utcnow().isoformat(),  # ✅ Now works correctly
        "download_link": storage_path,
        "file_type": "website",
        "full_name": url,
        "nickname": nickname or url,  # Default to full URL if no nickname
        "userID": userid
    }

    # Check if the URL already exists in Firestore
    if file_ref.get().exists:
        file_ref.update(file_data)  # ✅ Update existing record
        print(f"Updated metadata for {url} in Firestore.")
    else:
        file_ref.set(file_data)  # ✅ Create new record
        print(f"Stored new metadata for {url} in Firestore.")

    return storage_path

def load_embeddings_from_firebase(download_url):
    """
    Downloads and loads embeddings from Firebase Storage.
    """
    # Extract file path from the Firebase Storage URL
    file_path = download_url.replace("gs://notebookmvp.firebasestorage.app/", "")

    bucket = storage.bucket()
    blob = bucket.blob(file_path)

    # Download the JSON data
    json_data = blob.download_as_text()
    data = json.loads(json_data)

    return data["chunks"], data["embeddings"]  # Return text chunks & embeddings


def get_query_embedding(query):
    """Generate an embedding for the user's query using OpenAI."""
    response = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    return response.data[0].embedding  #

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two lists using PyTorch."""
    tensor1 = torch.tensor(vec1)
    tensor2 = torch.tensor(vec2)
    
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0).item()

def retrieve_top_chunks(query, chunks, stored_embeddings, top_n=3):
    """
    Finds the most relevant chunks based on cosine similarity with the query.

    Args:
        query (str): The user's question.
        chunks (list): List of website text chunks.
        stored_embeddings (list): List of stored embeddings.
        top_n (int): Number of relevant chunks to retrieve.

    Returns:
        list: The top-N most relevant text chunks.
    """
    query_embedding = get_query_embedding(query)  # Get query embedding

    # Compute cosine similarity manually
    similarities = [
        cosine_similarity(query_embedding, stored_embedding)
        for stored_embedding in stored_embeddings
    ]

    # Get top-N most relevant chunk indices
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]

    return [chunks[i] for i in top_indices]  # Return top chunks


def process_rag_query(user_id, url, user_query, top_n=3):
    """
    Looks up Firestore for a URL's stored embeddings, retrieves relevant chunks, and answers a query.
    """
    print("Looking up Firestore for stored embeddings...")

    # **Sanitize URL to match Firestore document ID**
    sanitized_url = sanitize_filename(url)  # ✅ Fix: Ensure Firestore lookup matches stored data

    file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_url)
    file_data = file_ref.get().to_dict()

    if not file_data or "download_link" not in file_data:
        return "No embeddings found for this URL."

    download_url = file_data["download_link"]

    print("Loading website data from Firebase...")
    chunks, embeddings = load_embeddings_from_firebase(download_url)

    print("Retrieving most relevant chunks...")
    relevant_chunks = retrieve_top_chunks(user_query, chunks, embeddings, top_n)

    # Construct the system message to enforce strict context usage
    system_prompt = (
        "You are an AI assistant that strictly answers user questions based only on the provided context.\n"
        "Do not use any external knowledge or make assumptions. If the answer is not in the provided context, "
        "reply with 'I don't have enough information to answer that based on the provided data.'\n\n"
        f"Context:\n{relevant_chunks}"
    )

    print("Generating response...")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},  # ✅ Context is enforced in system prompt
            {"role": "user", "content": user_query}  # ✅ User question remains separate
        ],
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

# API Endpoints

@app.route("/api/process_url", methods=["GET", "POST", "OPTIONS"])
def process_url():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    
    if request.method == "GET":
        # Get metadata of a stored URL
        user_id = request.args.get("user_id")
        url = request.args.get("url")

        if not user_id or not url:
            return add_cors_headers(jsonify({"error": "Missing required parameters"})), 400

        sanitized_url = sanitize_filename(url)
        file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_url)
        file_data = file_ref.get().to_dict()

        if not file_data:
            return add_cors_headers(jsonify({"message": "No data found for this URL"})), 404

        return add_cors_headers(jsonify(file_data))

    elif request.method == "POST":
        # Process and store a new URL
        data = request.json
        request_id = data.get('request_id')
        if not request_id:
            return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
        register_request(request_id)
        try:
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
            user_id = data.get("user_id")
            url = data.get("url")
            nickname = data.get("nickname", None)
            if not user_id or not url:
                return add_cors_headers(jsonify({"error": "Missing required parameters"})), 400
            print(f"Processing URL: {url}")
            content = scrape_website(url)
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
            if not content:
                return add_cors_headers(jsonify({"error": "Failed to retrieve content"})), 500
            chunks = chunk_text(content)
            embeddings = get_embeddings(chunks)
            storage_path = upload_to_firebase(user_id, url, chunks, embeddings, filename=nickname)
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
            sanitized_url = sanitize_filename(url)
            file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_url)
            file_data = {
                "created_at": datetime.utcnow().isoformat(),
                "download_link": storage_path,
                "file_type": "website",
                "full_name": url,
                "nickname": nickname or url,
                "userID": user_id
            }
            if file_ref.get().exists:
                file_ref.update(file_data)
            else:
                file_ref.set(file_data)
            return add_cors_headers(jsonify({
                "success": True,
                "message": "URL processed successfully", 
                "download_link": storage_path,
                "content": content
            }))
        finally:
            cleanup_request(request_id)

@app.route("/api/answer_with_rag", methods=["GET", "POST", "OPTIONS"])
def answer_with_rag():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
        
    if request.method == "GET":
        # Get stored content for a URL (debugging)
        user_id = request.args.get("user_id")
        url = request.args.get("url")

        if not user_id or not url:
            return add_cors_headers(jsonify({
                "error": "Missing required parameters"
            })), 400

        sanitized_url = sanitize_filename(url)
        file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_url)
        file_data = file_ref.get().to_dict()

        if not file_data:
            return add_cors_headers(jsonify({
                "message": "No data found for this URL"
            })), 404

        return add_cors_headers(jsonify(file_data))

    elif request.method == "POST":
        try:
            data = request.json
            request_id = data.get('request_id')
            if not request_id:
                return add_cors_headers(jsonify({'error': 'Missing required field: request_id'})), 400
            register_request(request_id)
            try:
                if is_request_cancelled(request_id):
                    return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
                user_id = data.get("user_id")
                url = data.get("url")
                user_query = data.get("query")
                if not user_id or not url or not user_query:
                    return add_cors_headers(jsonify({"error": "Missing required parameters"})), 400
                if url.startswith("gs://"):
                    try:
                        chunks, embeddings = load_embeddings_from_firebase(url)
                    except Exception as e:
                        print(f"Error loading from storage URL: {str(e)}")
                        return add_cors_headers(jsonify({"success": False, "error": f"Failed to load embeddings: {str(e)}"})), 500
                else:
                    sanitized_url = sanitize_filename(url)
                    file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_url)
                    file_data = file_ref.get().to_dict()
                    if not file_data or "download_link" not in file_data:
                        return add_cors_headers(jsonify({"success": False, "error": "No embeddings found for this URL"})), 404
                    chunks, embeddings = load_embeddings_from_firebase(file_data["download_link"])
                relevant_chunks = retrieve_top_chunks(user_query, chunks, embeddings)
                if is_request_cancelled(request_id):
                    return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
                system_prompt = (
                    "You are an AI assistant that strictly answers user questions based only on the provided context.\n"
                    "Do not use any external knowledge or make assumptions. If the answer is not in the provided context, "
                    "reply with 'I don't have enough information to answer that based on the provided data.'\n\n"
                    f"Context:\n{relevant_chunks}"
                )
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens=300
                )
                if is_request_cancelled(request_id):
                    return add_cors_headers(jsonify({'error': 'Request was cancelled', 'cancelled': True, 'success': False})), 499
                return add_cors_headers(jsonify({
                    "success": True,
                    "response": response.choices[0].message.content.strip()
                }))
            finally:
                cleanup_request(request_id)
        except Exception as e:
            print(f"Error in answer_with_rag: {str(e)}")
            return add_cors_headers(jsonify({"success": False, "error": str(e)})), 500

@app.route('/api/run_code_local', methods=['POST', 'OPTIONS'])
def run_code_local():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    
    try:
        data = request.json
        code = data.get('code')
        
        if not code:
            error_response = jsonify({
                "success": False,
                "error": "No code provided"
            })
            return add_cors_headers(error_response), 400
        
        # Set up execution environment with useful libraries
        exec_globals = {
            "__builtins__": __builtins__,
            "requests": __import__("requests"),
            "pandas": __import__("pandas"),
            "json": __import__("json"),
            "datetime": __import__("datetime"),
            "os": __import__("os"),
            "sys": __import__("sys"),
            "io": __import__("io"),
            "base64": __import__("base64"),
            "xlsxwriter": __import__("xlsxwriter"),
        }
        
        # Create a BytesIO buffer to capture output
        output_buffer = io.BytesIO()
        
        # Create a custom print function that writes to our buffer
        def custom_print(*args, **kwargs):
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            output = sep.join(str(arg) for arg in args) + end
            output_buffer.write(output.encode('utf-8'))
        
        # Add our custom print to the execution globals
        exec_globals['print'] = custom_print
        
        try:
            # Execute the code with our custom print function
            exec(code, exec_globals)
            
            # Get the output and decode it
            output_buffer.seek(0)
            result = output_buffer.getvalue().decode('utf-8').strip()
            
            # Create response with proper headers
            api_response = jsonify({
                "success": True,
                "output": result
            })
            return add_cors_headers(api_response)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_response = jsonify({
                "success": False,
                "error": error_traceback
            })
            return add_cors_headers(error_response), 400
            
    except Exception as e:
        error_response = jsonify({
            "success": False,
            "error": str(e)
        })
        return add_cors_headers(error_response), 500


# EXCEL AGENT STUFF BELOW

# API link configuration - when deployed on Render, this will be the same URL as the app itself
# API_LINK = 'http://localhost:5000'  # Always use the Render URL

EXCEL_SYSTEM_PROMPT = """
You are a Python code generation agent that creates Excel spreadsheets based on user instructions. You use the pandas and xlsxwriter libraries to generate .xlsx files.

Your role is to translate natural language instructions into Python code that creates spreadsheets with the requested data, formatting, and charts.

Your behavior:
- You ALWAYS save the file as "output.xlsx" with no path or subdirectories. Never use os.makedirs() or create folders.
- You always import pandas and use pd.ExcelWriter(..., engine="xlsxwriter") to write Excel files.
- If using a with block, you use: with pd.ExcelWriter(output.xlsx, engine="xlsxwriter") as writer:
- If not using a with block, you must call writer.close() at the end. Never use writer.save().
- You always use workbook and worksheet variables to add formatting and charts.
- You never call workbook.add_chart(chart). Only use workbook.add_chart({...}) to create a chart.
- You always return clean, working Python code with no explanations or comments.
- You use clear variable names and write readable, maintainable code.
- You do not generate markdown, commentary, or error messages.
- You only use features supported by the xlsxwriter engine.
- You never include pivot tables or use libraries other than pandas and xlsxwriter.

You understand how to apply the following formatting and visualization techniques:

<Add a column chart />
chart = workbook.add_chart({'type': 'column'})
chart.add_series({
    'name': 'Revenue',
    'categories': '=Sheet1!$A$2:$A$10',
    'values': '=Sheet1!$B$2:$B$10',
})
chart.set_title({'name': 'Monthly Revenue'})
chart.set_x_axis({'name': 'Month'})
chart.set_y_axis({'name': 'Revenue ($)'})
chart.set_legend({'position': 'bottom'})
worksheet.insert_chart('E2', chart)

<Add a line chart />
chart = workbook.add_chart({'type': 'line'})
chart.add_series({
    'name': 'Units Sold',
    'categories': '=Sheet1!$A$2:$A$10',
    'values': '=Sheet1!$C$2:$C$10',
    'marker': {'type': 'circle'},
})
chart.set_title({'name': 'Units Sold Over Time'})
worksheet.insert_chart('F2', chart)

<Add a pie chart />
chart = workbook.add_chart({'type': 'pie'})
chart.add_series({
    'name': 'Sales by Region',
    'categories': '=Sheet1!$A$2:$A$5',
    'values': '=Sheet1!$B$2:$B$5',
})
chart.set_title({'name': 'Sales Breakdown'})
worksheet.insert_chart('E10', chart)

<Format header row />
header_format = workbook.add_format({
    'bold': True,
    'bg_color': '#DCE6F1',
    'border': 1
})
worksheet.set_row(0, None, header_format)

<Format currency columns />
currency_format = workbook.add_format({'num_format': '$#,##0.00'})
worksheet.set_column("C:C", 15, currency_format)

<Format date columns />
date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
worksheet.set_column("A:A", 15, date_format)

<Set column widths />
worksheet.set_column("A:A", 20)
worksheet.set_column("B:B", 15)

<Create multiple sheets />
df1.to_excel(writer, sheet_name="Data", index=False)
df2.to_excel(writer, sheet_name="Summary", index=False)

<Insert image />
worksheet.insert_image('G2', 'logo.png')

<Write a formula />
worksheet.write_formula('D2', '=B2*C2')

<Write a bold, centered title />
title_format = workbook.add_format({'bold': True, 'font_size': 16, 'align': 'center'})
worksheet.merge_range('A1:D1', 'Q1 Sales Summary', title_format)

<Write a basic table />
data = [['Region', 'Revenue', 'Units Sold'],
        ['North', 1200, 30],
        ['South', 1500, 45]]
for row_num, row in enumerate(data):
    worksheet.write_row(row_num, 0, row)

<Highlight a cell />
highlight_format = workbook.add_format({'bg_color': '#FFD966'})
worksheet.write('B2', 1500, highlight_format)

<Center-align text />
center_format = workbook.add_format({'align': 'center'})
worksheet.set_column('A:A', 20, center_format)

<Freeze panes />
worksheet.freeze_panes(1, 0)

<Add autofilter />
worksheet.autofilter('A1:D1')

<Format as percent />
percent_format = workbook.add_format({'num_format': '0.00%'})
worksheet.set_column('E:E', 12, percent_format)

You know how to combine these formatting and charting techniques in response to user requests. 
Always return your answer as pure Python code.
Do not include JSON, markdown, comments, explanations, or surrounding text.
Only return raw Python code that will be executed as-is.
Your entire response will be passed directly into a Python exec() call. If your output includes anything other than valid Python code, it will break.
Name the final output file "output.xlsx". Make sure the output file is named "output.xlsx". It will not work if the file is named anything else.
You always use absolute paths: os.path.join("/Users/sahilsinha/Documents/caio/test_backend", "output.xlsx")
"""

def strict_code_cleaner(text: str) -> str:
    if text.startswith("```python") and text.endswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    lines = text.strip().splitlines()
    filtered = [
        line for line in lines
        if not line.strip().startswith(("Here", "This code", "```", "!", "It seems", "Let's"))
    ]
    return "\n".join(filtered).strip()


def is_syntax_valid(code: str) -> tuple[bool, str | None]:
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e.msg} (line {e.lineno})"


def generate_code_from_prompt(prompt: str) -> str:
    messages = [
        {"role": "developer", "content": EXCEL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return strict_code_cleaner(response.choices[0].message.content)


def fix_code_with_llm(code: str, error: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You're a Python code repair assistant. Return only valid, corrected Python code. "
                "Do not explain, comment, include markdown, or use pip installs. "
                "Your entire response will be executed with exec()."
            )
        },
        {
            "role": "user",
            "content": f"Broken code:\n\n{code}\n\nError:\n\n{error}"
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return strict_code_cleaner(response.choices[0].message.content)


def run_code(code: str) -> dict:
    """Internal code execution without HTTP calls"""
    try:
        # Set up execution environment with useful libraries
        exec_globals = {
            "__builtins__": __builtins__,
            "requests": __import__("requests"),
            "pandas": __import__("pandas"),
            "json": __import__("json"),
            "datetime": __import__("datetime"),
            "os": __import__("os"),
            "sys": __import__("sys"),
            "io": __import__("io"),
            "base64": __import__("base64"),
            "xlsxwriter": __import__("xlsxwriter"),
        }
        
        # Create a BytesIO buffer to capture output
        output_buffer = io.BytesIO()
        
        # Create a custom print function that writes to our buffer
        def custom_print(*args, **kwargs):
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            output = sep.join(str(arg) for arg in args) + end
            output_buffer.write(output.encode('utf-8'))
        
        # Add our custom print to the execution globals
        exec_globals['print'] = custom_print
        
        # Execute the code with our custom print function
        exec(code, exec_globals)
        
        # Get the output and decode it
        output_buffer.seek(0)
        result = output_buffer.getvalue().decode('utf-8').strip()
        
        return {
            "success": True,
            "output": result
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def validate_and_clean_code(code: str, max_syntax_retries: int = 2) -> str:
    cleaned = strict_code_cleaner(code)
    for attempt in range(max_syntax_retries + 1):
        is_valid, error = is_syntax_valid(cleaned)
        if is_valid:
            return cleaned
        cleaned = fix_code_with_llm(cleaned, error)
    raise ValueError("Code could not be fixed to pass Python syntax checks.")


def generate_and_run_code_from_prompt(prompt: str, max_syntax_retries: int = 2, max_exec_retries: int = 2):
    raw_code = generate_code_from_prompt(prompt)

    try:
        code = validate_and_clean_code(raw_code, max_syntax_retries)
    except Exception as syntax_fail:
        return {
            "success": False,
            "error": f"Syntax fix failed: {syntax_fail}",
            "code": raw_code,
            "stage": "syntax_validation"
        }

    for attempt in range(max_exec_retries + 1):
        result = run_code(code)
        if result.get("success"):
            return {
                "success": True,
                "output": result.get("output", ""),
                "code": code,
                "attempts": attempt + 1
            }
        error = result.get("error", "Unknown error")
        code = fix_code_with_llm(code, error)
        try:
            code = validate_and_clean_code(code, max_syntax_retries)
        except Exception as recheck_fail:
            return {
                "success": False,
                "error": f"Execution fix failed after retry: {recheck_fail}",
                "code": code,
                "stage": "exec_fix"
            }

    return {
        "success": False,
        "error": "Max retries exceeded during execution fixing.",
        "code": code,
        "stage": "final"
    }


def extract_excel_filename(code: str) -> str | None:
    match = re.search(r"pd\.ExcelWriter\(['\"]([^'\"]+\.xlsx)['\"]", code)
    return match.group(1) if match else None


@app.route('/api/excel_agent', methods=['POST', 'GET', 'OPTIONS'])
def excel_agent():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        if request.method == 'POST':
            data = request.json
            prompt = data.get('prompt')
            user_id = data.get('user_id')
            request_id = data.get('request_id')
        else:
            prompt = request.args.get('prompt')
            user_id = request.args.get('user_id')
            request_id = request.args.get('request_id')

        if not prompt:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'prompt' is required"
            })), 400

        if not user_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'user_id' is required"
            })), 400

        if not request_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'request_id' is required"
            })), 400

        register_request(request_id)
        try:
            print("📝 Received prompt:", prompt)
            print("👤 User ID:", user_id)
            print("🆔 Request ID:", request_id)

            # Check for cancellation before each major step
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Request was cancelled",
                    "cancelled": True
                })), 499

            result = generate_and_run_code_from_prompt(prompt)
            print("🔍 Generation result:", result)

            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Request was cancelled",
                    "cancelled": True
                })), 499

            generated_code = result.get("code", "")
            print("📦 Generated code:\n", generated_code)

            if not result.get("success"):
                print("❌ Code generation failed:", result.get("error"))
                return add_cors_headers(jsonify(result)), 500

            # Check for cancellation before file operations
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Request was cancelled",
                    "cancelled": True
                })), 499

            expected_output_path = os.path.join(os.getcwd(), "output.xlsx")
            print("🔍 Looking for file at:", expected_output_path)

            if not os.path.exists(expected_output_path):
                error_msg = f"Expected Excel file not found at: {expected_output_path}"
                print("❌", error_msg)
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": error_msg,
                    "debug_info": {
                        "generated_code": generated_code,
                        "working_directory": os.getcwd()
                    }
                })), 500

            # Check for cancellation before Firebase upload
            if is_request_cancelled(request_id):
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Request was cancelled",
                    "cancelled": True
                })), 499

            print("📤 Uploading Excel file to Firebase...")
            storage_path = upload_excel_to_firebase(user_id, expected_output_path)
            download_url = get_firebase_download_url(storage_path)

            if not download_url:
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Failed to generate download URL"
                })), 500

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            sanitized_filename = f"excel_{timestamp}"
            file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_filename)
            file_data = {
                "created_at": datetime.utcnow().isoformat(),
                "download_link": storage_path,
                "file_type": "excel",
                "full_name": f"Generated Excel - {prompt[:50]}...",
                "nickname": f"Excel Spreadsheet",
                "userID": user_id,
                "prompt": prompt
            }
            if file_ref.get().exists:
                file_ref.update(file_data)
            else:
                file_ref.set(file_data)
            try:
                os.remove(expected_output_path)
                print(f"🧹 Deleted local file: {expected_output_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete local file: {e}")
            print(f"✅ Successfully uploaded Excel file and generated download URL")
            return add_cors_headers(jsonify({
                "success": True,
                "download_url": download_url,
                "storage_path": storage_path,
                "message": "Done! Here's the link to your spreadsheet -",
                "request_id": request_id
            }))
        finally:
            cleanup_request(request_id)
    except Exception as e:
        print("❌ Unexpected error:", str(e))
        print("📜 Traceback:", traceback.format_exc())
        return add_cors_headers(jsonify({ 
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        })), 500

# Function to upload Excel files to Firebase Cloud Storage
def upload_excel_to_firebase(userid, excel_path, filename=None):
    """
    Uploads an Excel file to Firebase Cloud Storage under:
    gs://notebookmvp.firebasestorage.app/users/{userid}/excel/{filename}.xlsx

    Args:
        userid (str): User ID
        excel_path (str): Local path to the Excel file
        filename (str, optional): Custom file name (if None, defaults to timestamp)

    Returns:
        str: Firebase Storage file path
    """
    bucket = storage.bucket()
    
    # If no filename is provided, default to timestamp
    if filename is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"excel_{timestamp}"
    
    # Sanitize filename to prevent folder creation issues
    filename = sanitize_filename(filename)
    filename = f"users/{userid}/excel/{filename}.xlsx"  # User-specific path

    blob = bucket.blob(filename)
    
    # Upload the Excel file
    with open(excel_path, 'rb') as f:
        blob.upload_from_file(f, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Return the final storage path
    return f"gs://notebookmvp.firebasestorage.app/{filename}"

#INSTAGRAM AGENT STUFF BELOW

def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"

class SimplifiedPost:
    def __init__(self, pk, id, taken_at_str, image_url, comment_count, like_count, play_count, has_liked, caption):
        self.pk = pk
        self.id = id
        self.taken_at = taken_at_str
        self.image_url = str(image_url)  # Ensure JSON serializable
        self.comment_count = comment_count
        self.like_count = like_count
        self.play_count = play_count
        self.has_liked = has_liked
        self.caption = caption

    def __repr__(self):
        return f"<Post {self.id} | Likes: {self.like_count}, Comments: {self.comment_count}>"

    def to_dict(self):
        return {
            "pk": self.pk,
            "id": self.id,
            "taken_at": self.taken_at,
            "image_url": self.image_url,
            "comment_count": self.comment_count,
            "like_count": self.like_count,
            "play_count": self.play_count,
            "has_liked": self.has_liked,
            "caption": self.caption
        }

def simplify_post(media_obj):
    dt = media_obj.taken_at
    taken_at_str = f"{ordinal(dt.day)} {dt.strftime('%B %Y')}"

    # image_url = media_obj.thumbnail_url
    # if not image_url and media_obj.resources:
    #     image_url = media_obj.resources[0].thumbnail_url
    # image_url = str(image_url) if image_url else ""

    return SimplifiedPost(
        pk=media_obj.pk,
        id=media_obj.id,
        taken_at_str=taken_at_str,
        image_url=media_obj.image_url,
        comment_count=media_obj.comment_count,
        like_count=media_obj.like_count,
        play_count=getattr(media_obj, "play_count", None),
        has_liked=media_obj.has_liked,
        caption=media_obj.caption_text or ""
    )

# def get_instagram_posts_from_url(profile_url, post_limit=20):
#     # Extract username from URL
#     parsed_url = urlparse(profile_url)
#     match = re.match(r'^/?(?P<username>[\w\.]+)', parsed_url.path.strip('/'))
#     if not match:
#         raise ValueError("Invalid Instagram profile URL")
#     username = match.group("username")
# 
#     # Authenticate with hardcoded credentials
#     cl = Client()
#     cl.login("sahilsinha854", "Fax94!Leg")
# 
#     # Get user ID and latest media
#     user_id = cl.user_id_from_username(username)
#     medias = cl.user_medias(user_id, post_limit)
# 
#     # Convert to simplified format
#     return [simplify_post(media) for media in medias]

@app.route('/api/instagram_agent', methods=['POST', 'GET', 'OPTIONS'])
def instagram_agent():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    
    try:
        # Create hardcoded posts
        posts = [
            SimplifiedPost(
                pk="3603393380995525262",
                id="3603393380995525262_2319062",
                taken_at_str="4th April 2025",
                image_url="https://i.imgur.com/bY1F7gU.jpeg",
                comment_count=174,
                like_count=1984,
                play_count=0,
                has_liked=False,
                caption="Throw your coffee a party with our newest limited batch. Introducing Chobani® Coffee Creamer: Confetti Birthday Cake. It's marvelously rich, and made with real cream and only natural ingredients. Tag a friend you'd love to grab coffee and cake with."
            ),
        ]
        
        # Convert posts to dictionary format
        posts_data = [post.to_dict() for post in posts]
        
        return add_cors_headers(jsonify({
            "success": True,
            "posts": posts_data
        }))
            
    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500
    
# google image search stuff below
def search_google_images(query: str, num: int = 5) -> dict:
    """Performs a Google image search using Custom Search API"""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": os.getenv('GOOGLE_CSE_ID'),
            "key": os.getenv('GOOGLE_API_KEY'),
            "searchType": "image",
            "num": num,
            "safe": "active"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        results = response.json()
        return {
            "success": True,
            "results": [
                {
                    "url": item["link"],
                    "title": item.get("title"),
                    "contextLink": item.get("image", {}).get("contextLink")
                }
                for item in results.get("items", [])
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# existing function you already have
def download_image(url):
    response = requests.get(url, stream=True, timeout=5)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def create_simple_image_collage(image_urls, images_per_row=3, image_size=(300, 300)):
    images = []
    for url in image_urls:
        try:
            img = download_image(url)
            img = img.resize(image_size)
            images.append(img)
        except Exception as e:
            print(f"Could not load {url}: {e}")

    if not images:
        raise Exception("No images could be loaded.")

    rows = math.ceil(len(images) / images_per_row)
    collage_width = images_per_row * image_size[0]
    collage_height = rows * image_size[1]

    collage = Image.new("RGB", (collage_width, collage_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * image_size[0]
        y = row * image_size[1]
        collage.paste(img, (x, y))

    return collage

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_image_for_llm(image_url: str) -> str:
    """Convert image URL to base64 for LLM processing"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/jpeg;base64,{image_data}"
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

def analyze_image_with_llm(image_data: str | list[str], prompt: str) -> str:
    """Analyze image(s) using OpenAI's GPT-4 Vision"""
    try:
        # Handle single image vs multiple images
        if isinstance(image_data, str):
            image_payload = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]
        else:
            # For multiple images
            image_payload = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_data
                    }
                }
                for img_data in image_data
            ]

        # Add the text prompt after the image(s)
        payload = image_payload + [{"type": "text", "text": prompt}]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": payload
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {str(e)}"

@app.route('/api/image_search', methods=['GET', 'POST', 'OPTIONS'])
def image_search():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    
    try:
        if request.method == 'POST':
            if not request.is_json:
                return add_cors_headers(jsonify({
                    "success": False,
                    "error": "Content-Type must be application/json"
                })), 400
                
            data = request.get_json(force=True)
        else:  # GET
            data = request.args.to_dict()

        query = data.get('query')
        num = int(data.get('num', 5))
        make_collage = data.get('make_collage', False) or data.get('combine', False)
        image_prompt = data.get('image_prompt')  # Only handle individual image prompts for now

        if not query:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: 'query'"
            }), 400

        # Perform the image search
        search_result = search_google_images(query, num)
        
        if not search_result.get("success"):
            return jsonify(search_result), 500

        # Process and analyze individual images if prompt is provided
        if image_prompt:
            processed_images = []
            for item in search_result["results"]:
                image_data = process_image_for_llm(item["url"])
                if image_data:
                    analysis = {}
                    analysis["individual_analysis"] = analyze_image_with_llm(
                        image_data,
                        image_prompt
                    )
                    processed_images.append({
                        "url": item["url"],
                        "title": item.get("title"),
                        "contextLink": item.get("contextLink"),
                        "analysis": analysis
                    })
            search_result["results"] = processed_images

        # Create collage if requested
        if make_collage and search_result.get("success") and search_result.get("results"):
            try:
                image_urls = [img["url"] for img in search_result["results"]]
                collage_image = create_simple_image_collage(image_urls)
                search_result["collage"] = f"data:image/jpeg;base64,{image_to_base64(collage_image)}"
            except Exception as e:
                print(f"Error creating collage: {str(e)}")
                search_result["collage_error"] = str(e)

        response = jsonify(search_result)
        return add_cors_headers(response)

    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500

def strict_code_cleaner(text: str) -> str:
    if text.startswith("```python") and text.endswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    lines = text.strip().splitlines()
    filtered = [
        line for line in lines
        if not line.strip().startswith(("Here", "This code", "```", "!", "It seems", "Let's"))
    ]
    return "\n".join(filtered).strip()

def is_syntax_valid(code: str):
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e.msg} (line {e.lineno})"

def fix_code_with_llm(code: str, error: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You're a Python code repair assistant. Return only valid, corrected Python code. "
                "Do not explain, comment, include markdown, or use pip installs. "
                "Your entire response will be executed with exec()."
            )
        },
        {
            "role": "user",
            "content": f"Broken code:\n\n{code}\n\nError:\n\n{error}"
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return strict_code_cleaner(response.choices[0].message.content)

def validate_and_clean_code(code: str, max_syntax_retries: int = 2) -> str:
    cleaned = strict_code_cleaner(code)
    for attempt in range(max_syntax_retries + 1):
        is_valid, error = is_syntax_valid(cleaned)
        if is_valid:
            return cleaned
        cleaned = fix_code_with_llm(cleaned, error)
    raise ValueError("Code could not be fixed to pass Python syntax checks.")

# --- Code Execution Endpoint ---

@app.route('/api/run_plot_code_local', methods=['POST', 'OPTIONS'])
def run_plot_code_local():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    
    try:
        data = request.json
        code = data.get('code')
        if not code:
            return add_cors_headers(jsonify({"success": False, "error": "No code provided"})), 400
        
        exec_globals = {
            "__builtins__": __builtins__,
            "requests": __import__("requests"),
            "pandas": __import__("pandas"),
            "numpy": __import__("numpy"),
            "json": __import__("json"),
            "matplotlib": __import__("matplotlib"),
            "plt": __import__("matplotlib.pyplot"),
            "seaborn": __import__("seaborn"),
            "os": __import__("os"),
            "io": __import__("io"),
            "sys": __import__("sys"),
            "base64": __import__("base64"),
        }

        output_buffer = io.BytesIO()
        def custom_print(*args, **kwargs):
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            output = sep.join(str(arg) for arg in args) + end
            output_buffer.write(output.encode('utf-8'))
        exec_globals['print'] = custom_print
        
        try:
            exec(code, exec_globals)
            output_buffer.seek(0)
            result = output_buffer.getvalue().decode('utf-8').strip()
            return add_cors_headers(jsonify({"success": True, "output": result}))
        except Exception:
            return add_cors_headers(jsonify({
                "success": False,
                "error": traceback.format_exc()
            })), 400
    except Exception as e:
        return add_cors_headers(jsonify({"success": False, "error": str(e)})), 500

# --- LLM Prompt Generation ---

PLOT_SYSTEM_PROMPT = """
You are a Python code generation agent that creates data visualizations based on user instructions. You can use pandas, matplotlib, and seaborn libraries to generate plots and save them as PNG image files.

You always:
# - Save the plot to "output.png" in the current directory  # Production
# - Save the plot to "/Users/sahilsinha/Documents/caio/test_backend/output.png"  # Local
# - Use matplotlib.pyplot.savefig('output.png') at the end of plotting.  # Production
# - Use matplotlib.pyplot.savefig('/Users/sahilsinha/Documents/caio/test_backend/output.png') at the end of plotting.  # Local
- Do not display plots (never use plt.show()).
- Return pure runnable Python code. No markdown, no explanations.
- Use pandas for data handling.
- Use numpy if you need random or synthetic data generation.
- Always import all required libraries at the top.
- Do not use any other libraries beyond pandas, numpy, matplotlib, seaborn.
- No markdown, no commentary, no pip installs.

Your code will be executed directly by Python's exec().

Here are examples:

<Bar Chart>
import pandas as pd
import matplotlib.pyplot as plt

data = {'Region': ['North', 'South', 'East', 'West'], 'Sales': [300, 450, 200, 500]}
df = pd.DataFrame(data)

plt.figure(figsize=(8,6))
plt.bar(df['Region'], df['Sales'])
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales ($)")
# plt.savefig('output.png')  # Production
# plt.savefig('/Users/sahilsinha/Documents/caio/test_backend/output.png')  # Local
plt.savefig('output.png')
"""

def build_plot_system_prompt(chart_type: str) -> str:
    base_prompt = PLOT_SYSTEM_PROMPT
    if chart_type == "smart":
        return base_prompt
    chart_type_instruction = f"""
IMPORTANT:
- The user wants a {chart_type} chart.
- You must generate a {chart_type} chart.
- Use pandas, matplotlib, and seaborn to generate a {chart_type} plot.
- Do not generate any other chart types.
"""
    return chart_type_instruction + "\n" + base_prompt

def generate_plot_code_from_prompt(prompt: str, chart_type: str) -> str:
    system_prompt = build_plot_system_prompt(chart_type)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return strict_code_cleaner(response.choices[0].message.content)

# --- Full Visual Agent API ---

# Function to upload image to Firebase Cloud Storage
def upload_image_to_firebase(userid, image_path, filename=None):
    """
    Uploads an image file to Firebase Cloud Storage under:
    gs://notebookmvp.firebasestorage.app/users/{userid}/images/{filename}.png

    Args:
        userid (str): User ID
        image_path (str): Local path to the image file
        filename (str, optional): Custom file name (if None, defaults to timestamp)

    Returns:
        str: Firebase Storage file path
    """
    bucket = storage.bucket()
    
    # If no filename is provided, default to timestamp
    if filename is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}"
    
    # Sanitize filename to prevent folder creation issues
    filename = sanitize_filename(filename)
    filename = f"users/{userid}/images/{filename}.png"  # User-specific path

    blob = bucket.blob(filename)
    
    # Upload the image file
    with open(image_path, 'rb') as f:
        blob.upload_from_file(f, content_type="image/png")
    
    # Return the final storage path
    return f"gs://notebookmvp.firebasestorage.app/{filename}"

# Function to get public download URL from Firebase Storage
def get_firebase_download_url(storage_path):
    """
    Gets a public download URL for a file in Firebase Storage.
    
    Args:
        storage_path (str): Firebase Storage path (gs://...)
        
    Returns:
        str: Public download URL
    """
    try:
        # Extract file path from the Firebase Storage URL
        file_path = storage_path.replace("gs://notebookmvp.firebasestorage.app/", "")
        
        bucket = storage.bucket()
        blob = bucket.blob(file_path)
        
        # Generate a signed URL that expires in 1 hour
        url = blob.generate_signed_url(
            version="v4",
            expiration=3600,  # 1 hour
            method="GET"
        )
        
        return url
    except Exception as e:
        print(f"Error generating download URL: {str(e)}")
        return None

@app.route('/api/visual_agent', methods=['POST', 'OPTIONS'])
def visual_agent():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.json
        prompt = data.get('prompt')
        chart_type = data.get('chart_type', 'smart').lower()
        user_id = data.get('user_id')  # Add user_id parameter

        if not prompt:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'prompt'"
            })), 400

        if not user_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'user_id'"
            })), 400

        print(f"📝 Prompt: {prompt}")
        print(f" Chart type: {chart_type}")
        print(f" User ID: {user_id}")

        raw_code = generate_plot_code_from_prompt(prompt, chart_type)
        code = validate_and_clean_code(raw_code)

        # RUN_PLOT_ENDPOINT = "http://localhost:5000/api/run_plot_code_local"
        RUN_PLOT_ENDPOINT = "https://caio-backend.onrender.com/api/run_plot_code_local"
        response = requests.post(RUN_PLOT_ENDPOINT, json={"code": code})
        result = response.json()

        if not result.get("success"):
            return add_cors_headers(jsonify(result)), 500

        # Local version (comment out for production)
        # expected_output_path = os.path.join("/Users/sahilsinha/Documents/caio/test_backend", "output.png")
        
        # Production version (uncomment for deployment)
        expected_output_path = os.path.join(os.getcwd(), "output.png")
        if not os.path.exists(expected_output_path):
            return add_cors_headers(jsonify({
                "success": False,
                "error": f"Expected plot file not found at: {expected_output_path}"
            })), 500

        # Upload image to Firebase
        print(" Uploading image to Firebase...")
        storage_path = upload_image_to_firebase(user_id, expected_output_path)
        
        # Generate download URL
        download_url = get_firebase_download_url(storage_path)
        
        if not download_url:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Failed to generate download URL"
            })), 500

        # Store metadata in Firestore
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sanitized_filename = f"plot_{timestamp}"
        
        file_ref = db.collection("users").document(user_id).collection("files").document(sanitized_filename)
        file_data = {
            "created_at": datetime.utcnow().isoformat(),
            "download_link": storage_path,
            "file_type": "image",
            "full_name": f"Generated plot - {prompt[:50]}...",
            "nickname": f"Plot - {chart_type.title()}",
            "userID": user_id,
            "prompt": prompt,
            "chart_type": chart_type
        }

        if file_ref.get().exists:
            file_ref.update(file_data)
        else:
            file_ref.set(file_data)

        # Clean up local file
        try:
            os.remove(expected_output_path)
            print(f" Deleted local file: {expected_output_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete local file: {e}")

        return add_cors_headers(jsonify({
            "success": True,
            "download_url": download_url,
            "storage_path": storage_path,
            "message": "Image generated and uploaded successfully"
        }))

    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })), 500

#coffee agent stuff below
# Load dataset globally
# sheet_url = "https://docs.google.com/spreadsheets/d/1xA2ToifiDFcGHBRSl9nLlMl_I2KIb2JVDWF06iatFnw/export?format=csv&gid=1953223582"
# sheet_url = "https://docs.google.com/spreadsheets/d/1AxOk1qF-Q-bTqqmtTT__AcoF0yUMuHK5COaPX3Kb6tw/export?format=csv"
# df_coffee = pd.read_csv(sheet_url)

# @app.route("/query_coffee", methods=["GET"])
# def query_coffee():
#     user_query = request.args.get("q", default="", type=str)
#     if not user_query:
#         return jsonify({"error": "Missing required query parameter 'q'"}), 400

#     valid_values = {
#         "nyc_neighborhood": [
#             "East Village", "SoHo", "Flatiron District", "Gowanus", "Kips Bay",
#             "Chelsea", "NoMad", "Downtown", "Midtown", "Williamsburg",
#             "Garment District", "Hell's Kitchen", "Turtle Bay", "Boerum Hill",
#             "Greenwich Village", "South Street Seaport", "Long Island City",
#             "Bedford-Stuyvesant", "Lower East Side", "Meatpacking District",
#             "Koreatown", "Clinton Hill"
#         ],
#         "wifi": ["Yes", "Yes - limited", "Unknown", "No"],
#         "outlets": ["Unknown", "Few", "Enough", "No", "Many"],
#         "bathrooms": ["No", "Yes", "Unknown"],
#         "laptops_on_weekends": ["No", "Limited", "Yes", "Unknown"]
#     }

#     valid_options_str = "\n".join([f"- {col}: {', '.join(vals)}" for col, vals in valid_values.items()])
#     system_msg = (
#     "You are a helpful assistant. Based on a user query, extract structured filters to apply "
#     "to a coffee shop dataset with the following columns:\n\n"
#     f"{valid_options_str}\n\n"
#     "Translate natural phrases into structured filters using the columns and values above.\n"
#     "Use these mappings when you see common language:\n"
#     "- \"with outlets\" → filter for outlets ≠ \"No\" (i.e., value is one of: \"Few\", \"Enough\", \"Many\")\n"
#     "- \"with many outlets\" → {\"column\": \"outlets\", \"value\": \"Many\"}\n"
#     "- \"no outlets\" → {\"column\": \"outlets\", \"value\": \"No\"}\n"
#     "- \"with wifi\" → {\"column\": \"wifi\", \"value\": \"Yes\"}\n"
#     "- \"wifi with password\" → {\"column\": \"wifi\", \"value\": \"Yes - limited\"}\n"
#     "- \"no wifi\" → {\"column\": \"wifi\", \"value\": \"No\"}\n"
#     "- \"laptop friendly on weekends\" → {\"column\": \"laptops_on_weekends\", \"value\": \"Yes\"}\n"
#     "- \"no laptops on weekends\" → {\"column\": \"laptops_on_weekends\", \"value\": \"No\"}\n"
#     "- \"with bathrooms\" → {\"column\": \"bathrooms\", \"value\": \"Yes\"}\n"
#     "- \"no bathrooms\" → {\"column\": \"bathrooms\", \"value\": \"No\"}\n\n"
#     "You may return arrays of values if appropriate. For example:\n"
#     '{"filters": [{"column": "outlets", "value": ["Few", "Enough", "Many"]}]}\n\n'
#     "Only use values from the allowed lists. Return a JSON object with a \"filters\" array."
# )

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": user_query}
#         ],
#         temperature=0
#     )
#     content = response.choices[0].message.content

#     try:
#         filters = json.loads(content)["filters"]
#     except json.JSONDecodeError:
#         filters = []

#     # Apply hard filters
#     filtered_df = df_coffee.copy()
#     pandas_code = ["# Start with a copy of the original dataframe", "filtered_df = df_coffee.copy()"]
    
#     for f in filters:
#         col, val = f["column"], f["value"]
#         if col in filtered_df.columns:
#             filtered_df = filtered_df[filtered_df[col] == val]
#             pandas_code.append(f"# Filter for {col} == {val}")
#             pandas_code.append(f"filtered_df = filtered_df[filtered_df['{col}'] == '{val}']")

#     # Get final results
#     pandas_code.append("# Select final columns and get top 10 results")
#     pandas_code.append("locations = filtered_df[['name', 'address', 'nyc_neighborhood', 'rating', 'wifi', 'outlets', 'laptops_on_weekends']].head(10).to_dict(orient='records')")
    
#     locations = filtered_df[["name", "address", "nyc_neighborhood", "rating", "wifi", "outlets", "laptops_on_weekends"]].head(10).to_dict(orient="records")

#     # Generate a follow-up question
#     follow_up_prompt = f"Given this user request: '{user_query}', what is one follow-up question I could ask to help refine or narrow down their coffee shop search?"
#     follow_up_response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant who asks one smart follow-up question to refine a coffee shop search."},
#             {"role": "user", "content": follow_up_prompt}
#         ],
#         temperature=0.7
#     )
#     follow_up_question = follow_up_response.choices[0].message.content.strip()

#     return jsonify({
#         "locations": locations,
#         "follow_up_question": follow_up_question,
#         "pandas_code": "\n".join(pandas_code)
#     })

# def apply_manual_filters(df, filters):
#     filtered_df = df.copy()

#     if wifi := filters.get("wifi"):
#         if isinstance(wifi, list) and wifi:
#             filtered_df = filtered_df[filtered_df["wifi"].isin(wifi)]

#     if outlets := filters.get("outlets"):
#         if isinstance(outlets, list) and outlets:
#             filtered_df = filtered_df[filtered_df["outlets"].isin(outlets)]

#     if laptops := filters.get("laptops"):
#         if isinstance(laptops, list) and laptops:
#             filtered_df = filtered_df[filtered_df["laptops_on_weekends"].isin(laptops)]

#     if neighborhoods := filters.get("neighborhood"):
#         if isinstance(neighborhoods, list) and neighborhoods:
#             filtered_df = filtered_df[filtered_df["nyc_neighborhood"].isin(neighborhoods)]

#     if rating := filters.get("rating"):
#         try:
#             rating_float = float(rating)
#             filtered_df = filtered_df[filtered_df["rating"] >= rating_float]
#         except ValueError:
#             pass

#     return filtered_df.reset_index(drop=True)

# @app.route("/apply_filters", methods=["POST"])
# def filter_endpoint():
#     filters = request.get_json()
#     filtered = apply_manual_filters(df_coffee, filters)

#     locations = filtered.to_dict(orient="records")
#     return jsonify({"locations": locations})

@app.route('/', defaults={'path': ''}, methods=['GET', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'OPTIONS'])
def catch_all(path):
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204
    print(f"\nCaught unhandled request: {path}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    return add_cors_headers(jsonify({
        "error": "Route not found",
        "requested_path": path,
        "available_routes": [str(rule) for rule in app.url_map.iter_rules()]
    })), 404

research_client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")
# Extractor function
def extract_message_and_search_results(response):
    message = response.choices[0].message.content
    search_results = []
    if hasattr(response, "search_results"):
        search_results = response.search_results
    return {
        "message": message,
        "search_results": search_results
    }

# # API route
@app.route("/deepresearch", methods=["POST"])
def ask():
    data = request.get_json()
    request_id = data.get("request_id")
    if not request_id:
        return jsonify({"error": "Missing 'request_id' in request body"}), 400

    register_request(request_id)
    try:
        # Check for cancellation before starting
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        user_prompt = data["prompt"]
        search_engine = data["search_engine"]

        try:
            if search_engine == "perplexity":
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an artificial intelligence assistant and you need to "
                            "engage in a helpful, detailed, polite conversation with a user."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]

                # Check for cancellation before API call
                if is_request_cancelled(request_id):
                    return jsonify({
                        "success": False,
                        "error": "Request was cancelled",
                        "cancelled": True
                    }), 499

                response = research_client.chat.completions.create(
                    # model="sonar-deep-research",
                    model="sonar-pro",
                    messages=messages,
                )

                # Check for cancellation after API call
                if is_request_cancelled(request_id):
                    return jsonify({
                        "success": False,
                        "error": "Request was cancelled",
                        "cancelled": True
                    }), 499

                result = extract_message_and_search_results(response)
                response = jsonify(result)
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response

            elif search_engine == "firecrawl":
                # Check for cancellation before API call
                if is_request_cancelled(request_id):
                    return jsonify({
                        "success": False,
                        "error": "Request was cancelled",
                        "cancelled": True
                    }), 499
                response = firecrawl_client.search(user_prompt)
                # Check for cancellation after API call
                if is_request_cancelled(request_id):
                    return jsonify({
                        "success": False,
                        "error": "Request was cancelled",
                        "cancelled": True
                    }), 499
                result = {
                    "message": response.get('success', False),
                    "search_results": response.get('data', [])
                }
                return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    finally:
        cleanup_request(request_id)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.get_json()
    request_id = data.get("request_id")
    if not request_id:
        return jsonify({"error": "Missing 'request_id' in request body"}), 400

    print(f"Received scrape request with request_id: {request_id}")
    register_request(request_id)
    try:
        # Check for cancellation before starting
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        import time
        time.sleep(5)  # TEMPORARY DELAY FOR TESTING

        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' in request body"}), 400

        url = data["url"]
        prompt = data.get("prompt")  # Optional parameter

        # === Blocked URL logic ===
        BLOCKED_DOMAINS = ["x.com", "youtube.com", "cnbc.com"]
        try:
            result = urlparse(url)
            domain = result.netloc.lower()
            # Remove 'www.' prefix for comparison
            domain = domain[4:] if domain.startswith("www.") else domain
            if any(domain == blocked or domain.endswith("." + blocked) for blocked in BLOCKED_DOMAINS):
                return jsonify({
                    "success": False,
                    "error": "Sorry, this website cannot be scraped by the agent.",
                    "blocked": True
                }), 400
            if not all([result.scheme, result.netloc]):
                return jsonify({"error": "URL isn't valid try again"}), 400
        except Exception:
            return jsonify({"error": "URL isn't valid try again"}), 400

        # Check for cancellation before scraping
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        try:
            scrape_result = scrape_website(url)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Check for cancellation after scraping
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        if not prompt:
            return jsonify(scrape_result)

        # If prompt exists, analyze the content
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant analyzing web content. "
                    "Use the provided content to answer the user's question accurately. "
                    "If the answer cannot be found in the content, say so."
                    "Please be concise, and limit LLM preamble. Avoid phraises like 'Here is the data' or 'Here is the analysis'."
                ),
            },
            {
                "role": "user",
                "content": f"Content to analyze: {scrape_result}\n\nQuestion: {prompt}",
            },
        ]

        # Check for cancellation before LLM analysis
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        try:
            response = research_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                timeout=30
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Check for cancellation after LLM analysis
        if is_request_cancelled(request_id):
            return jsonify({
                "success": False,
                "error": "Request was cancelled",
                "cancelled": True
            }), 499

        return jsonify({
            "markdown": scrape_result,
            "analysis": response.choices[0].message.content
        })
    finally:
        cleanup_request(request_id)

@app.route('/api/apollo_enrich', methods=['POST', 'OPTIONS'])
def apollo_enrich():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.json
        name = data.get('name')
        company = data.get('company')
        prompt = data.get('prompt')  # Optional

        if not name or not company:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required fields: 'name' and 'company' are required"
            })), 400

        # Prepare Apollo API request
        apollo_url = (
            "https://api.apollo.io/api/v1/people/match"
            f"?name={requests.utils.quote(name)}"
            f"&organization_name={requests.utils.quote(company)}"
            "&reveal_personal_emails=false"
            "&reveal_phone_number=false"
        )

        apollo_api_key = os.getenv("APOLLO_API_KEY")
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "x-api-key": apollo_api_key
        }

        response = requests.post(apollo_url, headers=headers)
        try:
            apollo_result = response.json()
        except Exception:
            apollo_result = {"raw": response.text}

        # If no prompt, just return Apollo data
        if not prompt:
            return add_cors_headers(jsonify(apollo_result)), response.status_code

        # If prompt, run OpenAI client over the Apollo data
        system_message = (
            "You are an AI assistant analyzing Apollo enrichment data. "
            "Use the provided data to answer the user's question accurately. "
            "If the answer cannot be found in the data, say so."
            "Be very concise with your answers. Minimize preamble such as 'Here is the data' or 'Here is the analysis'."
        )
        user_message = f"Data to analyze: {json.dumps(apollo_result)}\n\nQuestion: {prompt}"

        openai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300
        )

        return add_cors_headers(jsonify({
            "apollo_data": apollo_result,
            "analysis": openai_response.choices[0].message.content
        })), response.status_code

    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500

@app.route('/api/cancel-request', methods=['POST', 'OPTIONS'])
def cancel_request_endpoint():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.json
        request_id = data.get('request_id')
        if not request_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'request_id'"
            })), 400

        cancelled = cancel_request(request_id)
        return add_cors_headers(jsonify({
            "success": True,
            "cancelled": cancelled,
            "message": "Request cancelled successfully" if cancelled else "Request not found"
        }))
    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500

# --- Request Cancellation Infrastructure ---
active_requests = {}
request_lock = threading.Lock()

def register_request(request_id):
    with request_lock:
        active_requests[request_id] = {
            'started_at': datetime.utcnow(),
            'cancelled': False
        }

def cancel_request(request_id):
    print(f"Cancelling request: {request_id}")
    print("Current running requests:", list(active_requests.keys()))
    with request_lock:
        if request_id in active_requests:
            active_requests[request_id]['cancelled'] = True
            return True
        return False

def is_request_cancelled(request_id):
    with request_lock:
        return active_requests.get(request_id, {}).get('cancelled', False)

def cleanup_request(request_id):
    with request_lock:
        active_requests.pop(request_id, None)

@app.route('/api/oai_deep_research', methods=['POST', 'OPTIONS'])
def test_responses():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.json
        input_text = data.get('input')
        if not input_text:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required field: 'input'"
            })), 400

        # Use the OpenAI responses API via direct HTTP request
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": "o3-deep-research",
            "input": input_text,
            "tools": [
                { "type": "web_search_preview" }
            ]
        }
        api_response = requests.post(url, headers=headers, json=payload)
        try:
            response_data = api_response.json()
        except Exception:
            response_data = api_response.text

        return add_cors_headers(jsonify({
            "success": api_response.status_code == 200,
            "response": response_data
        })), api_response.status_code
    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500

# --- Deep Research Async Integration ---
# 1. Start Deep Research (POST)
@app.route('/api/start_deep_research', methods=['POST', 'OPTIONS'])
def start_deep_research():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response()), 204

    try:
        data = request.json
        prompt = data.get('prompt')
        user_id = data.get('user_id')
        request_id = data.get('request_id')
        variable = data.get('variable')

        if not prompt or not user_id or not request_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing required fields: 'prompt', 'user_id', 'request_id'"
            })), 400

        # Make request to OpenAI Deep Research API
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": "o3-deep-research",
            "input": prompt,
            "tools": [
                { "type": "web_search_preview" }
            ],
            "background": True,
        }

        api_response = requests.post(url, headers=headers, json=payload)
        try:
            response_data = api_response.json()
        except Exception:
            response_data = {"raw": api_response.text}

        thread_id = response_data.get("id") or response_data.get("thread_id")
        if not thread_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "No thread_id returned from OpenAI",
                "response": response_data
            })), 500

        # Save to Firebase
        doc_data = {
            "thread_id": thread_id,
            "status": "in_progress",
            "created_at": datetime.utcnow().isoformat(),
            "prompt": prompt
        }

        if variable:
            db.collection("users").document(user_id).collection("variables").document(variable).set(doc_data)
        else:
            db.collection("users").document(user_id).collection("deep_research_calls").document(request_id).set(doc_data)

        return add_cors_headers(jsonify({
            "success": True,
            "thread_id": thread_id
        }))

    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500
    
    # ✅ Updated `/api/deep-research-callback`
@app.route('/api/deep-research-callback', methods=['POST'])
def deep_research_callback():
    try:
        webhook_secret = os.getenv('OPENAI_WEBHOOK_SECRET')

        if webhook_secret:
            webhook_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), webhook_secret=webhook_secret)
            try:
                event = webhook_client.webhooks.unwrap(request.data, request.headers)
            except InvalidWebhookSignatureError as e:
                print("Invalid signature", e)
                return Response("Invalid signature", status=400)
        else:
            event = request.get_json()

        thread_id = event.data.id if hasattr(event.data, 'id') else event.get("data", {}).get("id")
        status = event.type

        # Find matching document
        updated = False
        users_ref = db.collection("users")
        for user_doc in users_ref.stream():
            user_id = user_doc.id
            ref = users_ref.document(user_id).collection("deep_research_calls")
            for doc in ref.stream():
                if doc.to_dict().get("thread_id") == thread_id:
                    update_data = {
                        "updated_at": datetime.utcnow().isoformat(),
                        "status": "complete" if status == "response.completed" else "error"
                    }
                    if status == "response.failed":
                        update_data["error_msg"] = "OpenAI marked this request as failed"
                    ref.document(doc.id).update(update_data)
                    updated = True

        return Response(status=200 if updated else 404)

    except Exception as e:
        print(f"Webhook error: {e}")
        return Response("Webhook error", status=500)

# @app.route('/api/deep-research-callback', methods=['POST'])
# def deep_research_callback():
#     try:
#         webhook_secret = os.getenv('OPENAI_WEBHOOK_SECRET')
        
#         if webhook_secret:
#             # Create a temporary client with webhook secret for verification
#             webhook_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), webhook_secret=webhook_secret)
#             try:
#                 event = webhook_client.webhooks.unwrap(request.data, request.headers)
#             except InvalidWebhookSignatureError as e:
#                 print("Invalid signature", e)
#                 return Response("Invalid signature", status=400)
#         else:
#             print("⚠️ OPENAI_WEBHOOK_SECRET not set - skipping signature verification")
#             event = request.get_json()

#         print("✅ Webhook received from OpenAI:")
#         print(f"Event type: {event.type}")
#         print(f"Event data: {event.data}")

#         if event.type == "response.completed":
#             response_id = event.data.id
#             print(f"🎉 Deep research completed for response_id: {response_id}")
            
#             # Optional: Auto-retrieve and log the response
#             try:
#                 response = client.responses.retrieve(response_id)
#                 print("Response status:", response.status)
#                 # Extract main text like we did in the test
#                 main_text = None
#                 if hasattr(response, 'output') and response.output:
#                     for item in response.output:
#                         if hasattr(item, 'content'):
#                             for content in item.content:
#                                 if hasattr(content, 'text'):
#                                     main_text = content.text
#                                     break
#                             if main_text:
#                                 break
#                 print("Response text preview:", main_text[:200] + "..." if main_text and len(main_text) > 200 else main_text)
#             except Exception as e:
#                 print(f"Error retrieving response: {e}")

#         # Save webhook event to Firebase for debugging
#         from uuid import uuid4
#         db.collection("webhook_events").document(str(uuid4())).set({
#             "received_at": datetime.utcnow().isoformat(),
#             "event_type": event.type,
#             "response_id": event.data.id if hasattr(event.data, 'id') else None,
#             "full_event": str(event)  # Convert to string for JSON serialization
#         })

#         return Response(status=200)
        
#     except Exception as e:
#         print(f"❌ Webhook error: {str(e)}")
#         return Response(f"Webhook error: {str(e)}", status=500)

@app.route('/api/finalize_deep_result', methods=['POST'])
def finalize_deep_result():
    try:
        data = request.json
        thread_id = data.get("thread_id")
        if not thread_id:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Missing thread_id"
            })), 400

        # Retrieve the result from OpenAI using direct API call
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        url = f"https://api.openai.com/v1/responses/{thread_id}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        api_response = requests.get(url, headers=headers)
        try:
            response_data = api_response.json()
        except Exception:
            response_data = {"raw": api_response.text}

        if api_response.status_code != 200:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "Failed to retrieve result from OpenAI",
                "response": response_data
            })), 500

        # Extract main text and URLs from the response
        main_text = None
        urls = set()
        
        output = response_data.get("output", [])
        for item in output:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        main_text = content.get("text")
                        for ann in content.get("annotations", []):
                            if ann.get("type") == "url_citation" and ann.get("url"):
                                urls.add(ann["url"])
            elif item.get("type") == "web_search_call":
                url = item.get("action", {}).get("url")
                if url:
                    urls.add(url)

        update_data = {
            "status": "complete",
            "result": response_data,
            "result_text": main_text,
            "result_urls": list(urls),
            "value": main_text,
            "completed_at": datetime.utcnow().isoformat()
        }

        # Update matching document in either collection
        updated = False
        users_ref = db.collection("users")
        for user_doc in users_ref.stream():
            user_id = user_doc.id
            for coll in ["variables", "deep_research_calls"]:
                col_ref = users_ref.document(user_id).collection(coll)
                for doc in col_ref.stream():
                    if doc.to_dict().get("thread_id") == thread_id:
                        col_ref.document(doc.id).update(update_data)
                        updated = True

        if updated:
            return add_cors_headers(jsonify({"success": True}))
        else:
            return add_cors_headers(jsonify({
                "success": False,
                "error": "thread_id not found"
            })), 404

    except Exception as e:
        return add_cors_headers(jsonify({
            "success": False,
            "error": str(e)
        })), 500

# 3. Fetch Result (GET)
@app.route('/api/get_deep_result', methods=['GET'])
def get_deep_result():
    thread_id = request.args.get("thread_id")
    if not thread_id:
        return jsonify({"success": False, "error": "Missing thread_id"}), 400

    users_ref = db.collection("users")
    for user_doc in users_ref.stream():
        user_id = user_doc.id
        # Check variables
        vars_ref = users_ref.document(user_id).collection("variables")
        for var_doc in vars_ref.stream():
            doc = var_doc.to_dict()
            if doc.get("thread_id") == thread_id:
                if doc.get("status") == "complete":
                    return jsonify({"success": True, "result": doc.get("result")})
                elif doc.get("status") == "error":
                    return jsonify({"success": False, "error": doc.get("result")}), 500
                else:
                    return jsonify({"success": False, "error": "results aren't ready, check back later"}), 202
        # Check deep_research_calls
        drc_ref = users_ref.document(user_id).collection("deep_research_calls")
        for drc_doc in drc_ref.stream():
            doc = drc_doc.to_dict()
            if doc.get("thread_id") == thread_id:
                if doc.get("status") == "complete":
                    return jsonify({"success": True, "result": doc.get("result")})
                elif doc.get("status") == "error":
                    return jsonify({"success": False, "error": doc.get("result")}), 500
                else:
                    return jsonify({"success": False, "error": "results aren't ready, check back later"}), 202

    return jsonify({"success": False, "error": "thread_id not found"}), 404

@app.route('/test-retrieve-response', methods=['GET'])
def test_retrieve_response():
    try:
        # Use the thread_id from your webhook
        response_id = "resp_687eac86ecdc819abbf785f7ce0c110207c6df137d095cc2"
        response = client.responses.retrieve(response_id)
        print("=== RETRIEVED RESPONSE ===")
        print(response)
        
        # Extract the main text like we did before
        main_text = None
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            main_text = content.text
                            break
                    if main_text:
                        break
        
        return jsonify({
            "success": True,
            "response_id": response.id,
            "status": response.status,
            "main_text_preview": main_text[:500] + "..." if main_text and len(main_text) > 500 else main_text
        })
        
    except Exception as e:
        print(f"Error retrieving response: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    # Local version (comment out for production)
    # app.run(debug=False, port=5000, host='0.0.0.0')
    
    # Production version (uncomment for deployment)
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
