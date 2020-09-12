from flask import Flask, request, redirect, jsonify
import urllib.request
from werkzeug.utils import secure_filename
from sudoku_solver import solver
import os

UPLOAD_FOLDER = "images/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED = set(['jpg', 'jpeg'])

def allowed(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED

@app.route("/", methods=['POST'])
def upload():
    if 'file' not in request.files:
        response = jsonify({'Message' : 'Error'})
        response.status_code = 400
        return response
    file = request.files['file']
    if file.filename == '':
        response = jsonify({'Message': 'No file selected'})
        response.status_code = 400
        return response
    if file and allowed(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        response = solver("".join([UPLOAD_FOLDER, filename]))
        return response
    else:
        response = jsonify({"Message": "Not allowed"})
        response.status_code = 400
        return response

if __name__ == "__main__":
    app.run()
