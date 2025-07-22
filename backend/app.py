from flask import Flask, request, jsonify, session, send_from_directory
import os
import pandas as pd
import sweetviz
from werkzeug.utils import secure_filename
from src.data_upload.uploader import save_and_analyze_file
from src.data_analysis.analyzer import generate_report
from src.data_preprocessing.processor import preprocess_data
from src.model_training.trainer import train_model
from src.predictor.predictor import predict_from_model

app = Flask(__name__)
app.secret_key = "your-secret-key"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "reports")
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    df, info = save_and_analyze_file(file)
    if df is None:
        return jsonify(info), 400
    session["last_uploaded_data"] = df.to_json(orient="split")
    session["last_uploaded_filename"] = secure_filename(file.filename)
    return jsonify({"info": info, "filename": file.filename})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "last_uploaded_data" not in session:
        return jsonify({"error": "No data uploaded"}), 400
    df = pd.read_json(session["last_uploaded_data"], orient="split")
    report_filename = f"report_{session.get('last_uploaded_filename', 'data')}.html"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    generate_report(df, report_path)
    session["last_report"] = report_filename
    return jsonify({"report_path": report_filename})


@app.route("/get_report", methods=["GET"])
def get_report():
    if "last_report" not in session:
        return jsonify({"error": "No report generated"}), 400
    report_filename = session["last_report"]
    return send_from_directory(REPORTS_FOLDER, report_filename)


@app.route("/preprocess", methods=["POST"])
def preprocess():
    if "last_uploaded_data" not in session:
        return jsonify({"error": "No data uploaded"}), 400
    df = pd.read_json(session["last_uploaded_data"], orient="split")
    options = request.json
    processed_df = preprocess_data(df, options)
    session["last_uploaded_data"] = processed_df.to_json(orient="split")
    return jsonify({"columns": processed_df.columns.tolist()})


@app.route("/train", methods=["POST"])
def train():
    if "last_uploaded_data" not in session:
        return jsonify({"error": "No data uploaded"}), 400
    df = pd.read_json(session["last_uploaded_data"], orient="split")
    params = request.json
    result = train_model(df, params)
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    model_file = request.form.get("model_file")
    if not model_file:
        return jsonify({"error": "No model file specified"}), 400
    model_path = os.path.join(MODELS_FOLDER, model_file)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 404
    if "file" in request.files:
        file = request.files["file"]
        df = pd.read_csv(file)
    else:
        data = request.json.get("data")
        df = pd.DataFrame(data)
    predictions = predict_from_model(model_path, df)
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
