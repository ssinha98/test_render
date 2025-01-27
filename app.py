from flask import Flask, request
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/test', methods=['GET'])
def test_route():
    variable = request.args.get('variable')
    return f'variable received: {variable}'

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

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




if __name__ == '__main__':
    # Change to Flase or just remove when you deploy
    # app.run(debug=True)
    app.run()
