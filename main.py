from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('model/model_klasifikasi_kulit.h5')
class_labels = ['Combination', 'Dry', 'Normal', 'Oily']
recommendations = {
    'Normal': ['Niacinamide', 'Vitamin C', 'Peptides', 'Retinol', 'Green Tea Extract'],
    'Oily': ['Salicylic Acid', 'Niacinamide', 'Zinc', 'Tea Tree Oil', 'Benzoyl Peroxide', 'Retinol'],
    'Dry': ['Hyaluronic Acid', 'Ceramides', 'Squalane Panthenol', 'Vitamin E', 'Rosehip Oil', 'Glycerin', 'Ceramides', 'Sodium PCA', 'Allantoin'],
    'Combination': ['Salicylic Acid', 'Niacinamide', 'Witch Hazel', 'Retinol', 'Green Tea Extract', 'Glycolic Acid', 'Lactic Acid']
}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = class_labels[np.argmax(pred)]
    recs = recommendations.get(label, [])

    return jsonify({
        'jenis_kulit': label,
        'rekomendasi': recs
    })

if __name__ == '__main__':
    app.run()
