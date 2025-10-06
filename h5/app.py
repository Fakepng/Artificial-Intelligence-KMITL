import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "https://tafern.consolutechcloud.com"}}) # แก้ไข url
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mnist_cnn_model.h5")
model = load_model(MODEL_PATH)

def preprocess(img, target_size=(28, 28)):
    img = img.convert("L")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, target_size[1], target_size[0], 1)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    data = request.get_json()

    if "image_base64" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # แปลง base64 -> Image
        image_data = data["image_base64"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))

        # Preprocess
        input_data = preprocess(img)

        # Predict
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return jsonify({
            "prediction": prediction.tolist(),
            "predicted_class": predicted_class
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True) # ให้ทำการแก้ไข port