from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/test', methods=['GET'])
def test_route():
    variable = request.args.get('variable')
    return f'variable received: {variable}'


if __name__ == '__main__':
    # Change to Flase or just remove when you deploy
    # app.run(debug=True)
    app.run()
