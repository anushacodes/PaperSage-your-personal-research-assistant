from flask import Flask, render_template, request
from query import query_rag

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if request.method == 'POST':
        user_query = request.form['query_text']
        response_text = query_rag(user_query)  # This will call the function from query.py
        return render_template('index.html', query=user_query, response=response_text)


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change the port to 5001 or any other available port if 5000 doesn't work

