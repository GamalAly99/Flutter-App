
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os



app = Flask(__name__)

#      Firebase
#cred = credentials.Certificate("C:/Users/jaser/AndroidStudioProjects/gamal_project/corn-22-firebase-adminsdk-like-token.json")
#firebase_admin.initialize_app(cred)

#    Firestore
#db = firestore.client()


MODEL_PATH = "C:/Users/jaser/AndroidStudioProjects/gamal_project/best_modell.keras"


class_names = ["(Cercospora) Gray leaf spot", "Common rust", "Northern Leaf Blight", "healthy"]


model = tf.keras.models.load_model(MODEL_PATH)


IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    try:

        img = Image.open(image_path).convert("RGB")

        img = img.resize(IMG_SIZE)

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error in processing image: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded. Please upload an image file."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:

        temp_path = "temp_image.jpg"
        file.save(temp_path)


        img_array = preprocess_image(temp_path)


        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100


        os.remove(temp_path)


        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000)



