from flask import Flask, request, jsonify
from flask_cors import CORS

from nst_service import nst_apply

app = Flask(__name__)

CORS(app)
CORS(app, resources={r'/_api/*': {'origins': 'http://127.0.0.1:8000/'}})

@app.route('/', methods=['POST'])
def file_upload():
    file = request.files['file']
    file_url = nst_apply(file)
    return jsonify({"result": file_url})


if __name__ == '__main__':
    app.run()

