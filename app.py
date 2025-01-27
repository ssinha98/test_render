from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

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
    
    result = call_model(system_prompt, user_prompt)
    response = make_response(jsonify(result))
    response.set_cookie('session_active', 'true')
    return response

@app.route('/api/call-model-with-source', methods=['POST'])
def api_call_model_with_source():
    """LLM call with a source attached"""
    data = request.json
    system_prompt = data.get('system_prompt', '')
    user_prompt = data.get('user_prompt', '')
    processed_data = data.get('processed_data', '')
    
    # Prepend the source context to system prompt
    source_system_prompt = f"You are a helpful assistant. The user has given you the following source to use to answer questions. Please only use this source, and this source only, when helping the user. Source: {processed_data}\n\n{system_prompt}"
    
    result = call_model(source_system_prompt, user_prompt)
    response = make_response(jsonify(result))
    response.set_cookie('session_active', 'true')
    return response

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


if __name__ == '__main__':
    # Change to Flase or just remove when you deploy
    # app.run(debug=True)
    app.run()
