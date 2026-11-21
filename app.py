import os
import sys
from pathlib import Path

for parent in [Path.cwd(), *Path.cwd().parents]:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from flask import Flask, render_template, request
from src.audio.exception import CustomException
from src.audio.pipeline.prediction_pipeline import SinglePrediction
from src.audio.constants import *

app = Flask(__name__)


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/index", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            upload_file = request.files["fileup"]

            filename = "upload_file.wav"
            upload_dir = os.path.join(os.getcwd(), STATIC_DIR, UPLOAD_SUB_DIR)
            os.makedirs(upload_dir, exist_ok=True)
            upload_audio_path = os.path.join(upload_dir, filename)

            upload_file.save(upload_audio_path)
            pred = SinglePrediction()
            result = pred.predict()

            return render_template(
                "index.html",
                upload=True,
                upload_audio=filename,
                text=result.upper(),
            )

        return render_template("index.html")
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/training", methods=["GET"])
def training():
    return render_template("training.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    # Disabled on live demo so visitors don't crash Render's 512MB free tier
    return "Live retraining is disabled on the hosted demo. Model is pre-trained at 95.5% test accuracy."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5030))
    print(f" * Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)