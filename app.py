from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
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


load_dotenv()

app = Flask(__name__)
CORS(app)

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Reset counter on application start
api_call_count = 0
user_api_key = None
MAX_FREE_CALLS = 3

@app.route('/test')
def hello_world():
    return 'Hello, World!'

@app.route('/test_variable', methods=['GET'])
def test_route():
    variable = request.args.get('variable')
    return f'variable received: {variable}'

# api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


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
    
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Process file based on type
        with open(filename, 'rb') as f:
            processed_data = process_file(f, file_type)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'filepath': filename,
            'processed_data': processed_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set-api-key', methods=['POST'])
def set_api_key():
    """Endpoint to set user's API key"""
    global user_api_key
    data = request.json
    user_api_key = data.get('api_key')
    return jsonify({"success": True})

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
    
@app.route('/api/call-model', methods=['POST'])
def api_call_model():
    """Standard LLM call without source"""
    data = request.json
    system_prompt = data.get('system_prompt', '')
    user_prompt = data.get('user_prompt', '')
    save_as_csv = data.get('save_as_csv', False)
    
    result = call_model(system_prompt, user_prompt)
    
    if save_as_csv and result['success']:
        # Parse response into lines (split by newlines)
        response_lines = [line.strip() for line in result['response'].split('\n') if line.strip()]
        
        # Convert the response to a CSV string
        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)
        csv_writer.writerow(['Response'])  # Header
        
        # Write each line as a separate row
        for line in response_lines:
            csv_writer.writerow([line])
        
        # Create response with CSV file
        output = make_response(jsonify({
            **result,
            'csv_content': csv_data.getvalue(),
            'filename': 'response.csv'
        }))
        output.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return output
    
    # Regular JSON response if not saving as CSV
    response = make_response(jsonify(result))
    response.set_cookie('session_active', 'true')
    return response

@app.route('/api/call-model-with-source', methods=['POST'])
def api_call_model_with_source():
    """LLM call with a source attached"""
    data = request.json
    system_prompt = data.get('system_prompt', '')
    user_prompt = data.get('user_prompt', '')
    download_url = data.get('download_url', '')

    if not download_url:
        return jsonify({
            "success": False,
            "error": "No download URL provided"
        }), 400

    try:
        # Download the file content from the URL
        response = requests.get(download_url)
        response.raise_for_status()

        # Check if it's a PDF by looking at content type or URL
        is_pdf = ('application/pdf' in response.headers.get('Content-Type', '') or 
                 download_url.lower().endswith('.pdf'))

        if is_pdf:
            # Read PDF content
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            processed_data = ""
            for page in pdf_reader.pages:
                processed_data += page.extract_text() + "\n"
        else:
            # Regular text content
            processed_data = response.text

        # Prepend the source context to system prompt
        source_system_prompt = f"You are a helpful assistant. The user has given you the following source to use to answer questions. Please only use this source, and this source only, when helping the user. Source: {processed_data}\n\n{system_prompt}"
        
        result = call_model(source_system_prompt, user_prompt)
        response = make_response(jsonify(result))
        response.set_cookie('session_active', 'true')
        return response

    except requests.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Failed to download file: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to process file: {str(e)}"
        }), 500

@app.route('/oai', methods=['GET'])
def oai_route():
    system_prompt = request.args.get('system')
    user_prompt = request.args.get('user')
    sources = request.args.get('sources')

    try:
        if not api_key:
            return {
                "response": "Please add your own API key to continue using the service.",
                "success": False,
                "needs_api_key": True
            }

        client = OpenAI(api_key=api_key)
        
        # Prepare context from sources
        context = ""
        if sources:
            sources_dict = eval(sources)  # Convert string to dictionary
            for name, data in sources_dict.items():
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

@app.route('/api/check-api-key', methods=['GET'])
def check_api_key():
    """Check if custom API key exists and return count"""
    global user_api_key, api_call_count
    return jsonify({
        'hasCustomKey': bool(user_api_key),
        'apiKey': user_api_key if user_api_key else '',
        'count': api_call_count
    })

@app.route('/api/remove-api-key', methods=['POST'])
def remove_api_key():
    """Remove custom API key and reset count"""
    global user_api_key, api_call_count
    user_api_key = None
    api_call_count = 0
    return jsonify({'success': True})

@app.route('/api/get-count', methods=['GET'])
def get_count():
    """Get current API call count"""
    global api_call_count
    return jsonify({'count': api_call_count})


@app.route('/api/process-csv', methods=['POST'])
def process_csv():
    try:
        data = request.json
        print("Received request data:", data)
        
        file_path = data.get('filePath')
        filter_criteria = data.get('filterCriteria', [])
        
        print(f"Processing CSV with path: {file_path}")
        print(f"Applying filters: {filter_criteria}")
        
        # Read the CSV
        df = pd.read_csv(file_path)
        original_count = len(df)
        print(f"Original row count: {original_count}")
        
        # Apply each filter
        for criteria in filter_criteria:
            column = criteria['column']
            operator = criteria['operator']
            value = criteria['value']
            
            print(f"Applying filter: {column} {operator} {value}")
            
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
                
            print(f"Rows remaining after filter: {len(df)}")

        # Convert to different formats
        processed_data = df.to_string()  # For prompts
        raw_data = df.to_dict('records')  # For JSON
        
        print(f"Final row count: {len(df)}")
        
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
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        print(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

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

@app.route('/api/send-checkin-email', methods=['GET'])
def checkin_email():
    """Endpoint to trigger check-in email"""
    # Get email from query parameters
    email = request.args.get('email')
    print("Email being used:", email)  # Debug print
    
    response = send_checkin_email(email)
    
    if response and response.status_code == 200:
        return jsonify({
            "success": True,
            "message": "Email sent successfully",
            "sent_to": email or "default email"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to send email",
            "status_code": response.status_code if response else None,
            "details": response.text if response else "Failed to send email"
        }), 500

def perform_google_search(query: str = None, engine_type: str = "search", topic_token: str = None, section_token: str = None, window: str = None, trend: str = None, index_market: str = None) -> dict:
    """Performs a Google search using SerpAPI"""
    try:
        # Base params
        params = {
            "api_key": os.getenv('SERPAPI_KEY'),
            "gl": "us",  # Location set to US
            "hl": "en"   # Language set to English
        }
        
        # Set engine type and specific parameters
        if engine_type == "search":
            params["engine"] = "google"
            params["q"] = query
        elif engine_type == "news":
            params["engine"] = "google_news"
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
            if window:
                params["window"] = window
        elif engine_type == "markets":
            params["engine"] = "google_finance_markets"
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
                "results": results.get("organic_results", [])
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

@app.route('/api/search', methods=['GET', 'POST'])
def search():
    """Endpoint for Google search"""
    # Get parameters from either query string or JSON body
    if request.method == 'POST':
        data = request.json
        query = data.get('query')
        engine = data.get('engine', 'search')
        topic_token = data.get('topic_token')
        section_token = data.get('section_token')
        window = data.get('window')
        trend = data.get('trend')
        index_market = data.get('index_market')
    else:  # GET
        query = request.args.get('q')
        engine = request.args.get('engine', 'search')
        topic_token = request.args.get('topic_token')
        section_token = request.args.get('section_token')
        window = request.args.get('window')
        trend = request.args.get('trend')
        index_market = request.args.get('index_market')
    
    if engine == "news" and not (query or topic_token):
        return jsonify({
            "success": False,
            "error": "News search requires either a query or topic token"
        }), 400
    elif engine in ["search", "finance"] and not query:
        return jsonify({
            "success": False,
            "error": f"{engine} requires a query"
        }), 400
    elif engine == "markets" and not trend:
        return jsonify({
            "success": False,
            "error": "Markets search requires a trend parameter"
        }), 400
        
    result = perform_google_search(query, engine, topic_token, section_token, window, trend, index_market)
    return jsonify(result)

if __name__ == '__main__':
    # Change to Flase or just remove when you deploy
    app.run(debug=True, port=5000)

    # app.run(debug=True)
    # app.run()
