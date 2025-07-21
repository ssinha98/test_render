import os
from flask import Flask, request, jsonify, make_response
from openai import OpenAI
from firecrawl import FirecrawlApp
from flask_cors import CORS
from datetime import datetime
import threading


firecrawl_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

app = Flask(__name__)

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
                    model="sonar-deep-research",
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

# Add the cancel request endpoint
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

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "deep-research"})

# Add the main execution block
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')