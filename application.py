from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from config import configure_logging
from utils.predictions import run_prediction , log1p_transform

def create_app():
    """Application Factory"""
    configure_logging()  # Initialize logging
    logger = logging.getLogger(__name__)

    application = Flask(__name__)
    CORS(application)  # Enable CORS if needed

    @application.route("/", methods=["GET"])
    def index():
        return "Used Car Price Prediction API is up and running!"

    @application.route("/predict", methods=["POST"])
    def predict():
        try:
            payload = request.get_json(force=True)
            result = run_prediction(payload)
            # If there's an error key, return a 400 status code
            if "error" in result:
                return jsonify(result), 400
            elif "warning" in result:
                return jsonify(result), 200
            else:
                return jsonify(result), 200
        except Exception as e:
            logger.exception("Unhandled exception in /predict")
            return jsonify({"error": str(e)}), 500

    return application

# For local development
if __name__ == "__main__":

    application = create_app()
    application.run(debug=True, host="0.0.0.0", port=5000)
