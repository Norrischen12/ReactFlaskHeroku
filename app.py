import json
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from flask_cors import CORS, cross_origin

app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def index():
    # if request.method == "POST":
    return send_from_directory(app.static_folder, 'index')


@app.route("/members", methods=["POST"])
@cross_origin()
def members():
    arg1 = request.form.get('arg1')
    arg2_file = request.files['arg2']

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    arg2_path = os.path.join(upload_folder, arg2_file.filename)
    arg2_file.save(arg2_path)

    # Execute the Python script with the provided arguments
    result = subprocess.check_output(
        ['python', 'test.py', arg1, arg2_path], text=True)
    html_table = result.strip()
    response_data = {
        #"arg1": arg1,
        #'arg2': arg2_file,
         #Include any other data that you want to send back to the frontend
        'message': 'hello',
        'status': 'passed',
        'arg1_result': arg1,
        'html_table': html_table
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
